import math
import os
import random

from torch.utils.data import Dataset
import csv
import pandas as pd
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
        label = mask_table_content(cropped_table_content, number_of_masks=len(cropped_row_names))
        return {
            'table_name': self.table_name,
            'row_names': cropped_row_names,
            'col_names': self.col_names,
            'table_content': cropped_table_content,
            'label': label
        }
        # return DatasetHolder(self.table_name, row_names=cropped_row_names, col_names=self.col_names,
        #                      table_content=cropped_table_content)


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
    def __init__(self, datasets_path, number_of_records_per_crop, train_or_val, is_shuffle=True):
        self.datasets = []
        self.number_of_records_per_crop = number_of_records_per_crop
        self.is_shuffle = is_shuffle
        self.train_or_val = train_or_val

        self.read_datasets(datasets_path)

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
        for current_dir, dirs, _ in os.walk(datasets_path):
            for dataset_dir in dirs:
                for prefix_dir, _, files in os.walk(os.path.join(current_dir, dataset_dir)):
                    for file_name in files:
                        if file_name.endswith('.csv') and self.train_or_val in file_name:
                            dataset_holder = DatasetHolder(table_name=dataset_dir.replace('-', ' '),
                                                           path=os.path.join(prefix_dir, file_name))
                            dataset_cropper = DatasetCropper(dataset_holder=dataset_holder,
                                                             number_of_records_per_crop=self.number_of_records_per_crop,
                                                             is_shuffle=self.is_shuffle)
                            self.datasets.append(dataset_cropper)


def mask_table_content(cropped_table_content, number_of_masks):
    num_rows, num_cols = len(cropped_table_content), len(cropped_table_content[0])
    mask_idxs = random.sample([(i, j) for i in range(num_rows) for j in range(num_cols)], number_of_masks)
    mask_idxs.sort()
    label = ''
    mask_counter = 0
    for i, j in mask_idxs:
        mask_name = '[extra_id_' + str(mask_counter) + ']'
        gold = cropped_table_content[i][j]
        cropped_table_content[i][j] = mask_name
        label += mask_name + ' ' + gold + ' '
        mask_counter += 1
    return label


def split_train_val_test(datasets_path, frac=0.8):
    for current_dir, dirs, _ in os.walk(datasets_path):
        for dataset_dir in dirs:
            for prefix_dir, _, files in os.walk(os.path.join(current_dir, dataset_dir)):
                for file_path in files:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(os.path.join(prefix_dir, file_path), encoding='utf-8')
                        df['split'] = np.random.randn(df.shape[0], 1)

                        msk = np.random.rand(len(df)) <= frac

                        train = df[msk]
                        val = df[~msk]
                        train.to_csv(os.path.join(prefix_dir, 'train.csv'), encoding='utf-8')
                        val.to_csv(os.path.join(prefix_dir, 'val.csv'), encoding='utf-8')


if __name__ == '__main__':
    # split_train_val_test('train-data/csvs')
    a = [['hi', 'bi'], ['roi', 'ben'], ['hfdsaaaaaaai', 'bcdsi']]
    d = DatasetHolder('tab', row_names=['row1', 'row2', 'row3'], col_names=['1', '2'], table_content=a)
    print(d.get_dataset_holder_crop(0, 2))
