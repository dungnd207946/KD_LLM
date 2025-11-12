import os
import torch
import numpy as np
import argparse
from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from data_utils import OBQADatasetLoader, ARCDatasetLoader, PIQADatasetLoader, RiddleDatasetLoader, PubMedQADatasetLoader, BioASQDatasetLoader
from train_utils import train_and_evaluate

def compute_metrics_text(tokenizer):
    """
    Defines a function for computing custom evaluation metrics.

    Args:
        tokenizer: The tokenizer used for decoding model predictions.

    Returns:
        A function that computes metrics based on predictions and labels.
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions[0] = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
        tokenizer.batch_decode(predictions[0], skip_special_tokens=True)
        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Compute accuracy
        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))
        return {'accuracy': acc}

    return compute_metrics

def run(args):
    """
    Main function to run the training and evaluation process based on provided arguments.

    Args:
        args: Command-line arguments specifying dataset, model configuration, and training parameters.
    """
    # Load dataset using the appropriate loader
    if args.dataset == 'obqa':
        dataset_loader = OBQADatasetLoader()
        max_input_length = 100
    elif args.dataset == 'arc':
        dataset_loader = ARCDatasetLoader()
        max_input_length = 200
    elif args.dataset == 'piqa':
        dataset_loader = PIQADatasetLoader()
        max_input_length = 100
    elif args.dataset == 'riddle':
        dataset_loader = RiddleDatasetLoader()
        max_input_length = 100
    elif args.dataset == 'pubmedqa':
        dataset_loader = PubMedQADatasetLoader()
        max_input_length = 500
    elif args.dataset == 'bioasq':
        dataset_loader = BioASQDatasetLoader()
        max_input_length = 500
    else:
        raise ValueError
    
    datasets = dataset_loader.load_from_json()

    # Load rationales from language models
    train_llm_rationales, train_llm_labels, train_llm_llamarationales = dataset_loader.load_llm_preds(split='train')
    test_llm_rationales, test_llm_labels, test_llm_llamarationales = dataset_loader.load_llm_preds(split='test')

    # Add loaded information as columns to the datasets
    datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
    datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
    datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
    datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)
    datasets['train'] = datasets['train'].add_column('llm_llama_rationale', train_llm_llamarationales)
    datasets['test'] = datasets['test'].add_column('llm_llama_rationale', test_llm_llamarationales)

    if dataset_loader.has_valid:
        valid_llm_rationales, valid_llm_labels, valid_lamma_rationales = dataset_loader.load_llm_preds(split='valid')
        datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
        datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)
        datasets['valid'] = datasets['valid'].add_column('llm_llama_rationale', valid_lamma_rationales)
    else:
        train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)

        datasets = DatasetDict({
            'train': train_valid_datasets['train'],
            'valid': train_valid_datasets['test'],
            'test': datasets['test'],
        })
    if 'rationale' in datasets['train'].column_names:
        datasets = datasets.remove_columns('rationale')
        datasets = datasets.remove_columns('llamarationale')
    datasets = datasets.rename_column('llm_rationale', 'rationale')
    datasets = datasets.rename_column('llm_llama_rationale', 'llama_rationale')

    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained, device_map='auto', torch_dtype=torch.bfloat16)

    # Tokenization function to prepare data for training
    def tokenize_function(examples):
        # Tokenization logic for inputs, labels, and rationales
        model_inputs = tokenizer(['predict: ' + text for text in examples['input']],
                                 max_length=args.max_input_length,
                                 padding="max_length",
                                 truncation=True)
        
        # Tokenize the t5 explanations
        t5_model_inputs = tokenizer(['explain: ' + text for text in examples['input']],
                                      max_length=args.max_input_length,
                                      padding="max_length",
                                      truncation=True)

        # Tokenize the llama explanations
        llama_model_inputs = tokenizer(['rationale: ' + text for text in examples['input']],
                                            max_length=args.max_input_length,
                                            padding="max_length",
                                            truncation=True)

        # Assign the tokenized explanations to the model inputs
        model_inputs['t5_input_ids'] = t5_model_inputs['input_ids']
        model_inputs['t5_attention_mask'] = t5_model_inputs['attention_mask']
        model_inputs['llama_input_ids'] = llama_model_inputs['input_ids']
        model_inputs['llama_attention_mask'] = llama_model_inputs['attention_mask']

        # Tokenize the labels and rationales
        with tokenizer.as_target_tokenizer():
            label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
            rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)
            llamarationale_output_encodings = tokenizer(examples['llama_rationale'], max_length=256, truncation=True)

        # Assign the tokenized labels and rationales to the model inputs
        model_inputs['labels'] = label_output_encodings['input_ids']
        model_inputs['t5_labels'] = rationale_output_encodings['input_ids']
        model_inputs['llama_labels'] = llamarationale_output_encodings['input_ids']

        return model_inputs
        # If multiple GPUs are available, wrap the model with nn.DataParallel

    # Apply tokenization function to datasets
    tokenized_datasets = datasets.map(
        tokenize_function,
        remove_columns=['input', 'rationale', 'label', 'llm_label', 'llama_rationale'],
        batched=True
    )

    compute_metrics = compute_metrics_text(tokenizer)

    # Start training and evaluation
    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/flan-t5-large')
    parser.add_argument('--max_input_length', type=int, default=100)
    parser.add_argument('--grad_steps', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    args = parser.parse_args()
    run(args)