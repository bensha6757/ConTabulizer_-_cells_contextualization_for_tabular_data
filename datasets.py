from torch.utils.data import Dataset
import pandas as pd
from sklearn.utils import shuffle
import glob


class DatasetCropper(Dataset):
    def __init__(self, path, number_of_records_per_crop, table_name, is_shuffle=True):
        self.number_of_records_per_crop = number_of_records_per_crop
        self.df = pd.read_csv(path, encoding='utf-8')
        if is_shuffle:
            self.df = shuffle(self.df)
        self.table_name = table_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        start = idx * self.number_of_records_per_crop
        end = start + self.number_of_records_per_crop
        return self.table_name, self.df[start: end]


class DatasetsWrapper:
    def __init__(self, datasets_path, number_of_records_per_crop, is_shuffle=True):
        self.datasets = []
        self.read_datasets(datasets_path)
        self.number_of_records_per_crop = number_of_records_per_crop
        self.is_shuffle = is_shuffle

    def read_datasets(self, datasets_path):
        dataset_files = glob.glob(datasets_path)
        self.datasets = [DatasetCropper(dataset_path, self.number_of_records_per_crop, '', self.is_shuffle) for dataset_path in dataset_files]


