import torch
import os
import shutil
import logging
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
from model_utils import MultiTeacherDataCollator, MultiTeacherTrainer

def get_config_dir(args):
    """
    Constructs a directory path based on training arguments for model configurations.

    Args:
        args: Command-line arguments or any arguments object with necessary attributes.

    Returns:
        A string representing the path to the configuration directory.
    """
    return f'{args.dataset}/{args.from_pretrained.split("/")[1]}_{args.gamma}_{args.alpha}_{args.beta}_{args.max_input_length}_{args.grad_steps*args.batch_size}_{args.optimizer_name}_{args.lr}'

def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    """
    Sets up and runs the training and evaluation process for a sequence-to-sequence model.

    Args:
        args: Training configuration arguments.
        run: Integer representing the current run or seed for reproducibility.
        tokenizer: Tokenizer object for text preprocessing.
        tokenized_datasets: Tokenized datasets for training and evaluation.
        compute_metrics: Function to compute metrics during evaluation.

    This function initializes the model, sets up training arguments, data collator, and trainer,
    and then starts the training process.
    """
    set_seed(run)  # Ensure reproducibility

    model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)

    # Configuration directories for output and logging
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'
    logging_dir = f'logs/{config_dir}/{run}'

    # Adjust logging strategy based on arguments
    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # Clear existing checkpoint directory for a fresh start
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    # Setup training arguments for the Seq2SeqTrainer
    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns=False,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='no',
        save_steps=args.eval_steps,
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=args.eval_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        seed=run,
        local_rank=args.local_rank,
        bf16=args.bf16,
        generation_max_length=args.gen_max_len,
        prediction_loss_only=False,
    )

    # Initialize the data collator for handling batching and tokenization
    data_collator = MultiTeacherDataCollator(tokenizer=tokenizer, model=model)

    # Trainer setup with custom arguments for the training process
    trainer_kwargs = {
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': args.gamma,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["test"], },
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }

    # Initialize and run the trainer
    trainer = MultiTeacherTrainer(**trainer_kwargs)

    trainer.train()