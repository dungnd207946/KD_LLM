import argparse
import re
import json
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

# Define the root directory where datasets are stored.
DATASET_ROOT = 'datasets'

class DatasetLoader:
    """Base class for loading and processing specific datasets."""

    def __init__(self, dataset_name, has_valid, split_map, batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None):
        """
        Initializes the dataset loader.

        Args:
            dataset_name (str): Name of the dataset.
            has_valid (bool): Indicates if there's a validation set.
            split_map (dict): Maps dataset split names to their identifiers.
            batch_size (int): Size of each data batch.
            train_batch_idxs (list): Batch indices for training data.
            test_batch_idxs (list): Batch indices for test data.
            valid_batch_idxs (list, optional): Batch indices for validation data, if available.
        """
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.has_valid = has_valid
        self.split_map = split_map
        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs
        assert self.split_map is not None, "Split map cannot be None."

    def load_from_source(self):
        """Loads the dataset directly from its source."""
        datasets = load_dataset(self.dataset_name)
        return datasets

    def to_json(self, datasets):
        """Exports datasets to JSON files according to the split map."""
        for split_name, split_id in self.split_map.items():
            file_path = f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{split_name}.json'
            datasets[split_id].to_json(file_path)
        
    def load_from_json(self):
        """Loads the dataset from pre-exported JSON files."""
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }
        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json', })
        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets)
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx * self.batch_size, (idx + 1) * self.batch_size)
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])
        return datasets

    def load_llm_preds(self, split):
        labels = list()
        rationales = list()
        llamarationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                outputs = json.load(f)
            for output in outputs:
                rationale, label, llamarationale = self._parse_llm_output(output)
                rationales.append(rationale)
                labels.append(label)
                llamarationales.append(llamarationale)
        return rationales, labels, llamarationales

class OBQADatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'obqa'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(10)
        test_batch_idxs = range(1)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]
            c_3 = example['choices'][3]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}\n(d) {c_3}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale


class ARCDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'arc'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(3)
        test_batch_idxs = range(3)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]
            c_3 = example['choices'][3]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}\n(d) {c_3}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale


class PIQADatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'piqa'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(33)
        test_batch_idxs = range(2)
        valid_batch_idxs = range(2)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale


class RiddleDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'riddle'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(8)
        test_batch_idxs = range(2)
        valid_batch_idxs = range(2)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]
            c_3 = example['choices'][3]
            c_4 = example['choices'][4]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}\n(d) {c_3}\n(e) {c_4}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale


class PubMedQADatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'pubmedqa'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(1)
        test_batch_idxs = range(1)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale


class BioASQDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'bioasq'
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}'

            example['input'] = input
            example['label'] = example['answer']
            print("example['label']")
            print(example['label'])
            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(
            ['id', 'question', 'choices', 'answer'])

        return datasets

    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        rationale, label = rationale_label.split('Thus, the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
        except:
            label = ' '

        start_index = output.find("llama rationale: ")

        if start_index != -1:
            llamarationale = output[start_index + len("llama rationale: "):].strip()
        else:
            llamarationale = ""

        return rationale, label, llamarationale

# Main execution block for running as a script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    # Initialize the appropriate dataset loader based on the command line argument.
    if args.dataset == 'obqa':
        dataset_loader = OBQADatasetLoader()
    if args.dataset == 'arc':
        dataset_loader = ARCDatasetLoader()
    if args.dataset == 'piqa':
        dataset_loader = PIQADatasetLoader()
    if args.dataset == 'riddle':
        dataset_loader = RiddleDatasetLoader()
    if args.dataset == 'pubmedqa':
        dataset_loader = PubMedQADatasetLoader()
    if args.dataset == 'bioasq':
        dataset_loader = BioASQDatasetLoader()

    # Load dataset from source and export to JSON.
    datasets = dataset_loader.load_from_source()
    dataset_loader.to_json(datasets)
