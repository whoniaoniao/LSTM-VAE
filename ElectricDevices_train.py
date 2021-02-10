from vrae.vrae import VRAE
from vrae.utils import *
import numpy as np
import torch
import plotly
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import yaml
from attrdict import AttrDict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
def main(config):
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
    dload = config.dload

    sequence_length = X_train.shape[1]

    number_of_features = X_train.shape[2]

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

if __name__ == '__main__':
    # plotly.offline.init_notebook_mode()
    # Input parameters
    index = 1
    dload = './ElectricDevices_model_dir/' + str(index)  # download directory
    image_save = "./ElectricDevices_images/" + str(index)
    dload_path = Path(dload)
    plots = Path(dload + "/plots")
    if not Path.exists(dload_path):
        Path.mkdir(dload_path)
    if not Path.exists(plots):
        Path.mkdir(plots)

    torch.manual_seed(0)


    # Hyper parameters
    cwd = Path.cwd()
    with open(str(cwd / "config.yml")) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
        config_dict = config.copy()
        config = AttrDict(config)

    with open(str(dload + "/" + "config.yml"), "w") as handle:
        yaml.dump(config_dict, handle)

    # Load data and preprocess
    X_train, X_val, y_train, y_val = open_newdata_ED('data', ratio_train=0.9)

    num_classes = len(np.unique(y_train))
    print(num_classes
          )
    base = np.min(y_train)  # Check if data is 0-based
    if base != 0:
        y_train -= base
    y_val -= base
    train_dataset = TensorDataset(torch.from_numpy(X_train))
    test_dataset = TensorDataset(torch.from_numpy(X_val))
    #Save the model to be fetched later
    # To load a presaved model, execute:
    Model_Num = index
    # # Fit the model onto dataset
    config.dload = dload
    vrae = main(config)
    vrae.fit(train_dataset,save=True)
    vrae.save('vrae.pth')
    print("Save Model successfully")
    # vrae.load('./model_dir/{}/vrae.pth'.format(str(Model_Num)))
    # # loss = vrae.compute_loss(test_dataset)
    # # # Transform the input timeseries to encoded latent vectors
    #z_run = vrae.transform(test_dataset)
    # # # Visualize using PCA and tSNE
    #plot_clustering(z_run, y_val, engine='matplotlib', download=True, folder_name=image_save)
    # ##Transform the input dataset to encoded latent vectors
    # z_run_train = vrae.transform(train_dataset, save=False)
    # labels = y_train[:z_run_train.shape[0]]
    # clf = svm.SVC()
    # clf.fit(z_run_train, labels)
    # y_pred = clf.predict(z_run)
    # labels_for_z_run = y_train[:z_run.shape[0]]
    # print("accuracy of the svm", accuracy_score(labels_for_z_run,y_pred))

    # Visulalizing the input time series vs the Output time series
    # X_decoded = vrae.reconstruct(test_dataset)
    # for i in range(100):
    #     # Plotting test_dataset
    #     test_data_sample = test_dataset.tensors[0].numpy()[i,:]
    #     test_data_sample_time_steps = np.arange(0, test_data_sample.shape[0])
    #     plt.plot(test_data_sample_time_steps, test_data_sample)
    #     X_decoded_sample = np.squeeze(X_decoded)[:,i]
    #     X_decoded_sample_time_steps = np.arange(0, X_decoded_sample.shape[0])
    #     plt.plot(X_decoded_sample_time_steps, X_decoded_sample)
    #     plt.savefig("./model_dir/{}/plots/{}_compare.png".format(str(Model_Num), str(i)))
    #     plt.show()
    #     mean_error = np.mean(np.abs(X_decoded_sample - np.squeeze(test_data_sample)))
    #     print(mean_error)
    # print("After")
