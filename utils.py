from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import json
from imblearn.over_sampling import SMOTE

aa_to_int = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,
             'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}


class AminoAcidDataset(Dataset):
    """
    A PyTorch Dataset class for amino acid sequences.
    """

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def one_hot_encode(sequence, aa_to_int):
    """
    One-hot encodes an amino acid sequence.

    Parameters:
    sequence (str): The amino acid sequence to encode.
    aa_to_int (dict): Dictionary mapping amino acids to integers.

    Returns:
    np.ndarray: Flattened one-hot encoded vector.
    """
    encoding = np.zeros((len(sequence), len(aa_to_int)))
    for i, aa in enumerate(sequence):
        encoding[i, aa_to_int[aa]] = 1
    return encoding.flatten()


def load_and_encode(file_path, label):
    """
    Loads sequences from a file and encodes them.

    Parameters:
    file_path (str): Path to the file containing sequences.
    label (int): Label to assign to all sequences in the file.

    Returns:
    tuple: Arrays of encoded sequences and corresponding labels.
    """
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    encoded = [one_hot_encode(line, aa_to_int) for line in lines]
    labels = [label] * len(lines)
    return np.array(encoded), np.array(labels)


def create_data_loaders(pos_file_path, neg_file_path, train_size=0.9, batch_size=64, seed=42):
    """
    Creates training and testing data loaders with SMOTE for class imbalance.

    Parameters:
    pos_file_path (str): Path to the file containing positive sequences.
    neg_file_path (str): Path to the file containing negative sequences.
    train_size (float): Proportion of data to use for training.
    batch_size (int): Number of samples per batch.
    seed (int): Random seed for reproducibility.

    Returns:
    tuple: Training and testing data loaders.
    """
    X_pos, y_pos = load_and_encode(pos_file_path, 1)
    X_neg, y_neg = load_and_encode(neg_file_path, 0)

    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.concatenate((y_pos, y_neg), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed)

    smote = SMOTE(random_state=seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    train_dataset = AminoAcidDataset(X_train_resampled, y_train_resampled)
    test_dataset = AminoAcidDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def val_train_split(train_loader, split_ratio):
    """
    Splits a training data loader into training and validation sets.

    Parameters:
    train_loader (DataLoader): The data loader to split.
    split_ratio (float): Proportion of data to use for training.

    Returns:
    tuple: Validation and training data loaders.
    """
    dataset_size = len(train_loader.dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, sampler=val_sampler)
    return val_loader, train_loader


def save_hyperparameters(hidden_layers, lr, file_path):
    """
    Saves hyperparameters to a JSON file.

    Parameters:
    hidden_layers (list): List of hidden layer sizes.
    lr (float): Learning rate.
    file_path (str): Path to the file to save hyperparameters.
    """
    data = {'hidden_layers': hidden_layers, 'lr': lr}
    with open(file_path, 'w') as file:
        json.dump(data, file)


def load_hyperparameters(file_path):
    """
    Loads hyperparameters from a JSON file.

    Parameters:
    file_path (str): Path to the file to load hyperparameters from.

    Returns:
    tuple: Loaded hidden layers and learning rate.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    hidden_layers = data['hidden_layers']
    lr = data['lr']
    return hidden_layers, lr
