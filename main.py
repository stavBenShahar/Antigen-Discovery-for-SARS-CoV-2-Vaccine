import os.path
import matplotlib.pyplot as plt
import torch.cuda
from torch import nn
from utils import create_data_loaders, val_train_split, save_hyperparameters, load_hyperparameters, one_hot_encode, \
    aa_to_int
from model import MLP
import numpy as np
from sklearn.metrics import precision_score, recall_score
import random

default_lr = 0.001
default_hidden_layers = [180, 180]
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(dataloader, model, criterion, optimizer):
    """
    Trains the model for one epoch.

    Parameters:
    dataloader (DataLoader): DataLoader for training data.
    model (nn.Module): Model to be trained.
    criterion (nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimizer for training.

    Returns:
    float: Average training loss for the epoch.
    """
    model.train()
    train_loss = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).float().squeeze()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits.squeeze(), y)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(train_loss).item()


def test_epoch(dataloader, model, criterion):
    """
    Evaluates the model on the test data.

    Parameters:
    dataloader (DataLoader): DataLoader for test data.
    model (nn.Module): Model to be evaluated.
    criterion (nn.Module): Loss function.
    device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
    dict: Dictionary containing test loss, accuracy, precision, and recall.
    """
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_preds = []
    num_batches = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).float().squeeze()
            logits = model(X).squeeze()
            loss = criterion(logits, y).item()
            test_loss += loss

            preds = torch.round(torch.sigmoid(logits))
            correct_predictions += (preds == y).sum().item()
            total_predictions += len(y)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            num_batches += 1

    test_loss /= num_batches
    accuracy = correct_predictions / total_predictions
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return {"loss": test_loss, "accuracy": accuracy, "precision": precision, "recall": recall}


def train(train_dataloader, test_dataloader, model, criterion, optimizer, epochs):
    """
    Trains and evaluates the model for a given number of epochs.

    Parameters:
    train_dataloader (DataLoader): DataLoader for training data.
    test_dataloader (DataLoader): DataLoader for test data.
    model (nn.Module): Model to be trained.
    criterion (nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    epochs (int): Number of epochs to train.

    Returns:
    tuple: Lists of training and test losses per epoch.
    """
    train_loss_arr = []
    test_loss_arr = []
    for t in range(epochs):
        train_loss_arr.append(train_epoch(train_dataloader, model, criterion, optimizer))
        test_loss_arr.append(test_epoch(test_dataloader, model, criterion)['loss'])
        print(f"Epoch: {t + 1} | Train Loss: {train_loss_arr[t]:0.3f} | Test Loss: {test_loss_arr[t]:0.3f}")
    test_matrices = test_epoch(test_dataloader, model, criterion)
    print(
        f"Loss : {test_matrices['loss']:0.3f} | Accuracy: {test_matrices['accuracy'] * 100: 0.2f}% |"
        f" Precision: {test_matrices['precision']} | Recall: {test_matrices['recall']}")
    return train_loss_arr, test_loss_arr


def random_hidden_layer_options(num_options):
    """
    Generates random configurations of hidden layers.

    Parameters:
    num_options (int): Number of random configurations to generate.

    Returns:
    list: List of randomly generated hidden layer configurations.
    """
    hidden_layers_options = []
    sizes = [64, 128, 256, 512]
    layers = range(2, 5)
    for _ in range(num_options):
        hidden_layers = [random.choice(sizes) for _ in range(random.choice(layers))]
        hidden_layers_options.append(hidden_layers)
    return hidden_layers_options


def find_best_hyperparameters(criterion, train_loader, epochs=10):
    """
    Finds the best hyperparameters using random search.

    Parameters:
    criterion (nn.Module): Loss function.
    train_loader (DataLoader): DataLoader for training data.
    epochs (int): Number of epochs for each configuration.

    Returns:
    tuple: Best learning rate and hidden layer configuration.
    """
    val_loader, train_loader = val_train_split(train_loader, 0.8)

    best_test_loss = float('inf')
    best_params = None

    hidden_layers_options = random_hidden_layer_options(num_options=10)
    lr_options = [round(x, 3) for x in [0.001 + 0.01*i for i in range(10)]]

    for hidden_layers in hidden_layers_options:
        lr = random.choice(lr_options)
        model = MLP(hidden_layers=hidden_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            train_epoch(train_loader, model, criterion, optimizer)

        test_results = test_epoch(val_loader, model, criterion)
        test_loss = test_results['loss']

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_params = {'learning_rate': lr, 'hidden_layers': hidden_layers}

        print(f"Current config: LR= {lr:.3f} | Layers= {hidden_layers} | Test Loss: {test_loss:.3f}")

    print(f"Best Test Loss: {best_test_loss}")
    print(f"Best Parameters: {best_params}")

    return best_params['learning_rate'], best_params['hidden_layers']


def plot(train_loss_arr: list[float], test_loss_arr: list[float], title: str, save_path: str) -> None:
    """
    Plots training and test losses and saves the plot.

    Parameters:
    train_loss_arr (list): List of training losses.
    test_loss_arr (list): List of test losses.
    title (str): Title of the plot.
    save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_arr, label='Train Loss', color='orange')
    plt.plot(test_loss_arr, label='Test Loss', color='blue')
    plt.xlabel('Epoch Num')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def run_experiment(train_dataloader, test_dataloader, criterion, epochs, hyperparameters_path, with_activation_func,
                   use_default=False):
    """
    Runs an experiment with given hyperparameters and plots the results.

    Parameters:
    train_dataloader (DataLoader): DataLoader for training data.
    test_dataloader (DataLoader): DataLoader for test data.
    criterion (nn.Module): Loss function.
    epochs (int): Number of epochs to train.
    hyperparameters_path (str): Path to save/load hyperparameters.
    with_activation_func (bool): Whether to use activation functions in the model.
    use_default (bool): Whether to use default hyperparameters.

    Returns:
    tuple: Lists of training and test losses, trained model.
    """
    if not use_default:
        if not os.path.exists(hyperparameters_path):
            optimal_lr, optimal_hidden_layers = find_best_hyperparameters(criterion, train_dataloader)
            save_hyperparameters(optimal_hidden_layers, optimal_lr, hyperparameters_path)
        else:
            optimal_hidden_layers, optimal_lr = load_hyperparameters(hyperparameters_path)
    else:
        optimal_lr = default_lr
        optimal_hidden_layers = default_hidden_layers
    model = MLP(input_size=input_size, hidden_layers=optimal_hidden_layers,
                with_activation_func=with_activation_func).to(device)
    optimizer = torch.optim.SGD(model.parameters(), optimal_lr)
    train_loss_arr, test_loss_arr = train(train_dataloader, test_dataloader, model, criterion, optimizer, epochs)
    return train_loss_arr, test_loss_arr, model


def read_spike(filename, peptide_size=9):
    """
    Reads spike protein sequence from a file and encodes it into peptides.

    Parameters:
    filename (str): Path to the file containing spike protein sequence.
    peptide_size (int): Size of each peptide.

    Returns:
    tuple: List of peptides and their one-hot encoded representations.
    """
    with open(filename, 'r') as file:
        sequence = file.read().replace('\n', '').strip()

    peptides = [sequence[i:i + peptide_size] for i in range(0, len(sequence) - peptide_size + 1, peptide_size)]
    peptides_encoding = [one_hot_encode(peptide, aa_to_int) for peptide in peptides]

    return peptides, peptides_encoding


def top_peptides_from_spike(model, filename, amount_of_peptides=3):
    """
    Identifies top-scoring peptides from a spike protein sequence.

    Parameters:
    model (nn.Module): Trained model.
    filename (str): Path to the file containing spike protein sequence.
    amount_of_peptides (int): Number of top peptides to identify.

    Returns:
    list: List of top-scoring peptides.
    """
    peptides, peptides_encoding = read_spike(filename)
    peptides_encoding_tensor = torch.tensor(peptides_encoding, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(peptides_encoding_tensor)
    scores = outputs.squeeze().tolist()
    peptides_scores = list(zip(peptides, scores))
    sorted_peptides_scores = sorted(peptides_scores, key=lambda x: x[1], reverse=True)
    top_peptides = sorted_peptides_scores[:amount_of_peptides]
    return top_peptides


if __name__ == '__main__':
    # Variables
    split_ratio = 0.9
    epochs = 40
    batch_size = 64
    input_size = 180
    train_dataloader, test_dataloader = create_data_loaders("pos_A0201.txt", "neg_A0201.txt", split_ratio,
                                                            batch_size=batch_size)
    criterion = nn.BCEWithLogitsLoss()
    with_activation_func = True

    # Question 2b - first two inner layers are the input's dim
    question_2b_hyperparameters_path = "question_2b_hyperparameters.json"
    train_loss_arr, test_loss_arr, model0 = run_experiment(train_dataloader, test_dataloader, criterion, epochs,
                                                           question_2b_hyperparameters_path, with_activation_func,
                                                           use_default=True)
    plot(train_loss_arr, test_loss_arr, "MLP (2-Inner Layers - Input Dimension) ", "plots/question_2_b")

    # Question 2c - New network architecture
    question_2c_hyperparameters_path = "question_2c_hyperparameters.json"
    train_loss_arr, test_loss_arr, model1 = run_experiment(train_dataloader, test_dataloader, criterion, epochs,
                                                           question_2c_hyperparameters_path, with_activation_func)
    plot(train_loss_arr, test_loss_arr, "Regular MLP", "plots/question_2_c")

    # Question 2d - Removing non-linearity from the network
    with_activation_func = False
    train_loss_arr, test_loss_arr, model2 = run_experiment(train_dataloader, test_dataloader, criterion, epochs,
                                                           question_2c_hyperparameters_path, with_activation_func)
    plot(train_loss_arr, test_loss_arr, "MLP (No Activation Functions)", "plots/question_2_d")

    # Question 2e
    amount_of_peptides = 3
    top_peptides = top_peptides_from_spike(model1, "spike_protein_sequence", amount_of_peptides=amount_of_peptides)
    print(f"Top {amount_of_peptides} peptides from the spike are:")
    for peptide in top_peptides:
        print(f"    {peptide} ")
