import copy
import numpy as np
import torch
import json
from torch.utils.data import Dataset

def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

class ETADataset(Dataset):
    def __init__(self, data, opt):
        # id, M1, feature, M2, M3, M4, ata, M5 = data
        id, feature, static, top_5, ata = data

        self.id = np.array(id)
        self.feature = np.array(feature)
        self.static = np.array(static)
        self.top_5 = np.array(top_5)
        self.target = np.array(ata)

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        id = self.id[index]
        feature = self.feature[index]
        static = self.static[index]
        top_5 = self.top_5[index]
        target = self.target[index]
        return [torch.tensor(id),
                torch.tensor(feature),
                torch.tensor(static),
                torch.tensor(top_5),
                torch.tensor(target)]

class TaskBatchGenerator(object):
    def __init__(self, train_data):
        super(TaskBatchGenerator).__init__()
        self.train_data = train_data
        self.data_num = len(self.train_data)
        self.batch_num = 18
        self.batch_size = int(self.data_num / self.batch_num)

    def getTaskBatch(self):
        train_data = self.train_data.copy()
        np.random.shuffle(train_data)
        task_batches = []
        for b in range(self.batch_num):
            if b < self.batch_num - 1:
                task_batches.append(train_data[b * self.batch_size: (b+1) * self.batch_size])
            else:
                task_batches.append(train_data[b * self.batch_size:])
        return task_batches

