import copy
import numpy as np
import torch
import json
from torch.utils.data import Dataset


# Function to load JSON data from a file
def load_json(file):
    """
    Load JSON data from a specified file.

    Args:
        file (str): Path to the JSON file.

    Returns:
        data (dict or list): The loaded JSON data.
    """
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e  # Raise an exception if loading fails
    return data


# Custom dataset class inheriting from PyTorch's Dataset
class ETADataset(Dataset):
    """
    ETADataset for loading and managing training/testing data.

    Args:
        data (tuple): A tuple containing the following elements:
            id (list): Sample IDs.
            feature (list): Dynamic feature data.
            static (list): Static feature data.
            top_5 (list): Top 5 related data.
            ata (list): Target values.
        opt (dict): Optional parameters (not used here).
    """

    def __init__(self, data, opt):
        # Unpack the input data
        id, feature, static, top_5, ata = data

        # Convert data to numpy arrays
        self.id = np.array(id)
        self.feature = np.array(feature)
        self.static = np.array(static)
        self.top_5 = np.array(top_5)
        self.target = np.array(ata)

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.id)

    def __getitem__(self, index):
        """
        Retrieve a single sample based on the index.

        Args:
            index (int): Index of the sample.

        Returns:
            list[torch.Tensor]: A list of tensors for the sample components.
        """
        # Retrieve individual components of the sample
        id = self.id[index]
        feature = self.feature[index]
        static = self.static[index]
        top_5 = self.top_5[index]
        target = self.target[index]

        # Convert the data to PyTorch tensors and return
        return [
            torch.tensor(id),
            torch.tensor(feature),
            torch.tensor(static),
            torch.tensor(top_5),
            torch.tensor(target)
        ]


# Task batch generator class
class TaskBatchGenerator(object):
    """
    TaskBatchGenerator for creating batches of training data.

    Args:
        train_data (list): List of training data.
    """

    def __init__(self, train_data):
        # Initialize the parent class
        super(TaskBatchGenerator).__init__()

        # Store training data
        self.train_data = train_data

        # Total number of data samples
        self.data_num = len(self.train_data)

        # Define the number of batches (fixed to 18)
        self.batch_num = 18

        # Calculate batch size (last batch may have fewer samples)
        self.batch_size = int(self.data_num / self.batch_num)

    def getTaskBatch(self):
        """
        Generate random shuffled batches of training tasks.

        Returns:
            list: A list containing data for each batch.
        """
        # Create a copy of the training data and shuffle it
        train_data = self.train_data.copy()
        np.random.shuffle(train_data)

        # Initialize the list of task batches
        task_batches = []

        # Split data into batches
        for b in range(self.batch_num):
            if b < self.batch_num - 1:
                # Generate the first batch_num-1 complete batches
                task_batches.append(train_data[b * self.batch_size: (b + 1) * self.batch_size])
            else:
                # The last batch may contain the remaining samples
                task_batches.append(train_data[b * self.batch_size:])

        return task_batches
