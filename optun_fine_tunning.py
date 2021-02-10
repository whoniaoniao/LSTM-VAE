"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from vrae.utils import *
import optuna
from vrae.vrae import VRAE
from torch.utils.data import DataLoader, TensorDataset




def define_vrae_model(trial, sequence_length, number_of_features):
    hidden_layer_depth = trial.suggest_int("n_hidden_layers", 1, 2)
    hidden_size = trial.suggest_int("hidden_size", 30, 60, step=10)
    latent_length = trial.suggest_int("latent_length",2, 10, step=2)
    batch_size = trial.suggest_int("batch_size", 1, 10, step=1)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.0005, step=0.0002)
    n_epochs = trial.suggest_int("n_epochs", 200, 1600, step=200)
    dropout_rate = trial.suggest_float("learning_rate", 0.1, 0.2, step=0.1)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    loss = trial.suggest_categorical("loss", ["SmoothL1Loss", "MSELoss"])
    # block = trial.suggest_categorical("block_choose", ["LSTM", "GRU"])

    # hidden_layer_depth = 1
    # hidden_size = 30
    # latent_length = 10
    # batch_size = 64
    # learning_rate = 0.00005
    # n_epochs = 5
    # dropout_rate = 0.0005
    # n_epochs = 2
    # optimizer = "Adam"
    # loss = "SmoothL1Loss"
    block = "LSTM"
    dload = './ECG200_model_dir/'

    print_every = 30
    cuda = True
    clip = True
    max_grad_norm = 5
    vrae = VRAE(sequence_length=sequence_length,
                number_of_features=number_of_features,
                hidden_size=hidden_size,
                hidden_layer_depth=hidden_layer_depth,
                latent_length=latent_length,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                cuda=cuda,
                print_every=print_every,
                clip=clip,
                max_grad_norm=max_grad_norm,
                loss=loss,
                block=block,
                dload=dload)
    return vrae


def objective(trial):
    # Generate the model.
    # Load data and preprocess
    DEVICE = torch.device("cuda")
    pwd = os.getcwd()
    # X_train, X_val, y_train, y_val = open_newdata_ED("ElectricDevices", ratio_train=0.9)
    X_train, X_val, y_train, y_val = open_newdata("ECG200", ratio_train=0.9)


    train_dataset = TensorDataset(torch.from_numpy(X_train))
    test_dataset = TensorDataset(torch.from_numpy(X_val))
    sequence_length = X_train.shape[1]
    number_of_features = X_train.shape[2]
    drae_model = define_vrae_model(trial, sequence_length, number_of_features).to(DEVICE)

    drae_model.fit(train_dataset, save=True)
    average_loss = drae_model.test_loss_compute(test_dataset, save=False)

    return average_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")  # Minimize the loss
    study.optimize(objective, n_trials=10000, timeout=None)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))