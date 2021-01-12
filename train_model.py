from vrae.vrae import VRAE
from vrae.utils import *
import numpy as np
import torch

import plotly
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import yaml
from attrdict import AttrDict
# plotly.offline.init_notebook_mode()
# Input parameters
index = 4
dload = './model_dir/' + str(index)  # download directory
image_save = "./images/"+ str(index)
dload_path = Path(dload)
if not Path.exists(dload_path):
    Path.mkdir(dload_path)

# Hyper parameters
cwd = Path.cwd()
with open(str(cwd/"config.yml")) as handle:
    config = yaml.load(handle, Loader=yaml.FullLoader)
    config_dict = config.copy()
    config = AttrDict(config)

with open(str(dload +"/"+"config.yml"), "w") as handle:
    yaml.dump(config_dict, handle)
    
hidden_size = config.hidden_size
hidden_layer_depth = config.hidden_layer_depth
latent_length = config.latent_length
batch_size = config.batch_size
learning_rate = config.learning_rate
n_epochs = config.n_epochs
dropout_rate = config.dropout_rate
optimizer = config.optimizer  # options: ADAM, SGD
cuda = config.cuda  # options: True, False
print_every = config.print_every
clip = config.clip  # options: True, False
max_grad_norm = config.max_grad_norm
loss = config.loss  # options: SmoothL1Loss, MSELoss
block = config.block  # options: LSTM, GRU

# Load data and preprocess
X_train, X_val, y_train, y_val = open_data('data', ratio_train=0.9)

num_classes = len(np.unique(y_train))
base = np.min(y_train)  # Check if data is 0-based
if base != 0:
    y_train -= base
y_val -= base

train_dataset = TensorDataset(torch.from_numpy(X_train))
test_dataset = TensorDataset(torch.from_numpy(X_val))

sequence_length = X_train.shape[1]

number_of_features = X_train.shape[2]

vrae = VRAE(sequence_length=sequence_length,
            number_of_features=number_of_features,
            hidden_size=hidden_size,
            hidden_layer_depth=hidden_layer_depth,
            latent_length=latent_length,
            batch_size=batch_size, \
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

# Fit the model onto dataset
vrae.fit(train_dataset, save=True)

# Transform the input timeseries to encoded latent vectors
z_run = vrae.transform(test_dataset, save=True)

# Save the model to be fetched later
vrae.save('vrae.pth')
print("Save Model successfully")
# Visualize using PCA and tSNE
plot_clustering(z_run, y_val, engine='matplotlib', download=True, folder_name=image_save)
