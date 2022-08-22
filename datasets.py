import math

from torch.utils.data import Dataset
import csv
import glob
import numpy as np


class DatasetHolder:
    def __init__(self, table_name, path=None, row_names=None, col_names=None, table_content=None):
        self.table_name = table_name
        if table_content is not None:
            self.row_names = row_names
            self.col_names = col_names
            self.table_content = table_content
        else:
            self.row_names = []
            self.col_names = []
            self.table_content = []
            with open(path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for i, row in enumerate(reader):
                    if i == 0:
                        self.col_names = row[1:]
                    else:
                        self.row_names.append(row[0])
                        self.table_content.append(row[1:])

    def __len__(self):
        return len(self.row_names)

    def shuffle(self):
        permutation = np.random.permutation(len(self.row_names))
        self.table_content = [self.table_content[i] for i in permutation]
        self.row_names = [self.row_names[i] for i in permutation]

    def get_dataset_holder_crop(self, start, end):
        cropped_row_names = self.row_names[start: end]
        cropped_table_content = self.table_content[start: end]
        return DatasetHolder(self.table_name, row_names=cropped_row_names, col_names=self.col_names,
                             table_content=cropped_table_content)


class DatasetCropper(Dataset):
    def __init__(self, dataset_holder, number_of_records_per_crop, is_shuffle=True):
        self.number_of_records_per_crop = number_of_records_per_crop
        self.dataset_holder = dataset_holder
        if is_shuffle:
            self.dataset_holder.shuffle()

    def __len__(self):
        return math.ceil(len(self.dataset_holder) / self.number_of_records_per_crop)

    def __getitem__(self, idx):
        start = idx * self.number_of_records_per_crop
        end = start + self.number_of_records_per_crop
        return self.dataset_holder.get_dataset_holder_crop(start, end)


class DatasetsWrapper(Dataset):
    def __init__(self, datasets_path, number_of_records_per_crop, is_shuffle=True):
        self.datasets = []
        self.read_datasets(datasets_path)
        self.number_of_records_per_crop = number_of_records_per_crop
        self.is_shuffle = is_shuffle
        self.datasets_lens = []
        lens = 0
        for dataset in self.datasets:
            lens += len(dataset)
            self.datasets_lens.append(lens)

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        for i, dataset_len in enumerate(self.datasets_lens):
            if idx < dataset_len:
                return self.datasets[i][idx - (self.datasets_lens[i - 1] if i != 0 else 0)]

    def read_datasets(self, datasets_path):
        dataset_files_dirs = glob.glob(datasets_path)
        for dataset_dir in dataset_files_dirs:
            for file_path in glob.glob(dataset_dir):
                if file_path.endswith('.csv'):
                    dataset_holder = DatasetHolder(table_name=dataset_dir.split('/')[-1], path=file_path)
                    dataset_cropper = DatasetCropper(dataset_holder, self.number_of_records_per_crop, self.is_shuffle)
                    self.datasets.append(dataset_cropper)
