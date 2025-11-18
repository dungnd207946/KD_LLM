# multi_teacher_kd_fixed.py
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

import torch
from torch import nn
from datasets import Dataset, DatasetDict

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -----------------------------
# Data utilities
# -----------------------------
class DataProcessor:
    """
    - split_data: HF split into train/valid/test
    - prepare_data: normalize list -> HF Dataset with 'input_text', 'target_text'
    - tokenize_function: build prompt, mask prompt tokens with -100, keep targets to train on
    """

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512, user_assistant_template: Optional[Tuple[str, str]] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if user_assistant_template is None:
            self.user_prefix = "### User:\n"
            self.asst_prefix = "\n### Assistant:\n"
        else:
            self.user_prefix, self.asst_prefix = user_assistant_template

    @staticmethod
    def split_data(
        dataset: Dataset,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetDict:
        assert 0 < train_ratio < 1 and 0 <= valid_ratio < 1 and train_ratio + valid_ratio < 1
        test_ratio = 1.0 - train_ratio - valid_ratio

        interim = dataset.train_test_split(test_size=(1.0 - train_ratio), seed=seed, shuffle=True)
        valid_test = interim["test"].train_test_split(test_size=(test_ratio / (valid_ratio + test_ratio)), seed=seed, shuffle=True)

        return DatasetDict(
            train=interim["train"],
            validation=valid_test["train"],
            test=valid_test["test"],
        )

    @staticmethod
    def prepare_data(data_list: List[Dict]) -> Dataset:
        normalized = []
        for item in data_list:
            if isinstance(item, dict):
                if "input_text" in item and "target_text" in item:
                    normalized.append({"input_text": item["input_text"], "target_text": item["target_text"]})
                elif "input" in item and "target" in item:
                    normalized.append({"input_text": item["input"], "target_text": item["target"]})
                else:
                    raise ValueError(f"Unsupported dict keys in item: {list(item.keys())}")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.append({"input_text": item[0], "target_text": item[1]})
            else:
                raise ValueError("Each item must be dict with input/target or a 2-tuple/list.")
        return Dataset.from_list(normalized)

    def _build_prompt(self, input_text: str, target_text: str) -> Tuple[str, str]:
        prompt = f"{self.user_prefix}{input_text}{self.asst_prefix}"
        full_text = f"{prompt}{target_text}"
        return prompt, full_text

    def tokenize_function(self, examples: Dict) -> Dict[str, List[int]]:
        """
        Tokenizes prompt + target together; labels mask ONLY the prompt with -100 so the loss is computed on targets.
        Guarantees at least one non -100 label to avoid zero-loss batches.
        """
        input_texts = examples["input_text"]
        target_texts = examples["target_text"]

        full_inputs = []
        prompt_lens = []

        for inp, tgt in zip(input_texts, target_texts):
            prompt, full_text = self._build_prompt(inp, tgt)
            # get prompt length (no special tokens)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_inputs.append(full_text)
            prompt_lens.append(len(prompt_ids))

        tok = self.tokenizer(
            full_inputs,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )

        labels = []
        for input_ids, p_len in zip(tok["input_ids"], prompt_lens):
            p_len = min(p_len, len(input_ids))
            cur_labels = [-100] * p_len + input_ids[p_len:]
            # Ensure at least one supervised position
            if all(x == -100 for x in cur_labels):
                # If truncation ate the target, supervise the last token
                cur_labels[-1] = input_ids[-1]
            labels.append(cur_labels)

        tok["labels"] = labels
        return tok


# -----------------------------
# Collator for decoder-only with precomputed labels
# -----------------------------
@dataclass
class DataCollatorForCausalLMWithLabels:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        batch = self.tokenizer.pad(
            {k: [f[k] for f in features] for k in features[0] if k != "labels"},
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if lab.numel() < max_len:
                pad = torch.full((max_len - lab.numel(),), self.label_pad_token_id, dtype=torch.long)
                lab = torch.cat([lab, pad], dim=0)
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
        batch["labels"] = torch.stack(padded_labels, dim=0)

        # Safety: ensure not all labels are -100 in the batch
        if (batch["labels"] != self.label_pad_token_id).sum().item() == 0:
            batch["labels"][-1, -1] = batch["input_ids"][-1, -1]

        return batch


# -----------------------------
# Multi-teacher KD Trainer
# -----------------------------
class MultipleTeacherTrainer(Trainer):
    """
    total_loss = student_loss + alpha * teacher1_loss (teacher term is detached)
    Teachers are frozen; same inputs/labels.
    """

    def __init__(self, *args, teacher1_model: nn.Module, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher1_model = teacher1_model.eval()
        for p in self.teacher1_model.parameters():
            p.requires_grad = False
        self.alpha = float(alpha)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Standard student forward pass
        outputs = model(**inputs)
        student_loss = outputs.loss  # tensor with grad

        # Align teacher device
        device = inputs["input_ids"].device
        if next(self.teacher1_model.parameters()).device != device:
            self.teacher1_model.to(device)

        with torch.no_grad():
            t1_outputs = self.teacher1_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                labels=inputs["labels"],
            )
            # Detach to keep teacher out of the graph
            t1_loss = t1_outputs.loss.detach()

        total = student_loss + self.alpha * t1_loss
        return (total, outputs) if return_outputs else total


# -----------------------------
# Utilities
# -----------------------------
def count_trainable_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def get_tokenized_data(data_file_path, tokenizer):
    # --- Load data ---
    with open(data_file_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)
    if not isinstance(raw_samples, list) or not raw_samples or "input_text" not in raw_samples[0]:
        raise ValueError("Dataset must be a non-empty list of {'input_text', 'target_text'} dicts.")

    print("Example sample:", raw_samples[0])

    processor = DataProcessor(tokenizer=tokenizer, max_length=256)
    dataset_all = processor.prepare_data(raw_samples)
    splits = processor.split_data(dataset_all, train_ratio=0.75, valid_ratio=0.125, seed=42)

    tokenized_data = DatasetDict(
        train=splits["train"].map(processor.tokenize_function, batched=True, remove_columns=splits["train"].column_names),
        validation=splits["validation"].map(processor.tokenize_function, batched=True, remove_columns=splits["validation"].column_names),
        test=splits["test"].map(processor.tokenize_function, batched=True, remove_columns=splits["test"].column_names),
    )
    return tokenized_data

def get_teacher_model(teacher1_ckpt):
    # --- Teacher model (frozen) ---
    teacher1 = AutoModelForCausalLM.from_pretrained(
        teacher1_ckpt,
        # quantization_config=bnb_config,
        device_map={"": 0},
    )
    teacher1.config.use_cache = False
    for p in teacher1.parameters():
        p.requires_grad = False
    teacher1.eval()
    return teacher1

def get_student_model(student_ckpt):
    # --- Student model ---
    student = AutoModelForCausalLM.from_pretrained(
        student_ckpt,
        # quantization_config=bnb_config,
        device_map={"": 0},
    )
    student.config.use_cache = False

    # --- LoRA ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=None,  # let PEFT find common modules; customize if needed
    )

    # Attach LoRA
    student = get_peft_model(student, peft_config)
    student.print_trainable_parameters()
    trn, tot = count_trainable_parameters(student)
    print(f"Trainable params: {trn:,} / {tot:,} ({100 * trn / tot:.4f}%)")
    return student

def get_tokenizer(student_ckpt):
    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(student_ckpt, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

# -----------------------------
# Main
# -----------------------------
def train_and_evaluate(training_args, tokenized_data, teacher1, student, tokenizer, save_model_path):
    set_seed(42)

    # --- Quantization (optional but common) ---
    # use_bnb = None
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=False,
    # ) if use_bnb else None

    # Prepare for k-bit training BEFORE adding LoRA
    # if use_bnb:
    #     student = prepare_model_for_kbit_training(student, use_gradient_checkpointing=True)
    #     # For some archs, enabling input requires grad helps with PEFT on k-bit
    #     student.enable_input_require_grads()

    collator = DataCollatorForCausalLMWithLabels(tokenizer)

    trainer = MultipleTeacherTrainer(
        model=student,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        teacher1_model=teacher1,
        alpha=0.5,
    )

    # Sanity: ensure there are supervised tokens in a sample batch
    sample = collator([tokenized_data["train"][0]])
    assert (sample["labels"] != -100).any(), "Collator produced a batch with all -100 labels."

    # --- Train & evaluate ---
    train_out = trainer.train()
    print("Train output:", train_out)

    metrics = trainer.evaluate(tokenized_data["test"])
    print("Test metrics:", metrics)

    # Lưu lại mô hình đã huấn luyện
    trainer.save_model(save_model_path)
    tokenizer.save_pretrained(save_model_path) 


if __name__ == "__main__":
    # --- Training tham số ---
    training_args = TrainingArguments(
        output_dir="./kd_fixed_outputs",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4, # -> batch size sẽ là 2x4=8
        learning_rate=1e-4,
        num_train_epochs=10,  # tune as needed
        logging_steps=10,
        save_steps=200,
        eval_steps=100,
        max_grad_norm=1.0,
        save_total_limit=2,
        fp16=True,
        report_to=[],  # no external logger by default
        remove_unused_columns=False,  # keep custom fields
        # gradient_checkpointing=True,     # helps w/ memory; works with prepare_model_for_kbit_training
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
    )

    teacher1_ckpt = "./bigscience/bloomz-560m"
    teacher1 = get_teacher_model(teacher1_ckpt)

    student_ckpt = "./kd_v1_finetuned_model"
    if os.path.exists(student_ckpt) and os.path.isdir(student_ckpt):
        student_ckpt = student_ckpt
    else:
        student_ckpt = "./google/gemma-3-270m-it"
    student = get_student_model(student_ckpt)

    tokenizer = get_tokenizer(student_ckpt)

    data_file_path = "./my_datasets/raw_samples_3000.json"
    tokenized_data = get_tokenized_data(data_file_path, tokenizer)

    save_model_path = "./kd_v1_finetuned_model"

    train_and_evaluate(training_args, tokenized_data, teacher1, student, tokenizer, save_model_path)
