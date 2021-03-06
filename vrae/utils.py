from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt

from plotly.graph_objs import *
import plotly
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, MeanShift, DBSCAN
from sklearn import metrics
from sklearn import svm
from sklearn import mixture
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score


# def svm_latent_space(data, label):
#     clf = svm.SVC()
#     clf.fit(data, label)
#     return clf

def plot_clustering(z_run, labels, engine='plotly', download=False, folder_name='clustering'):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """

    def plot_clustering_plotly(z_run, labels):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        trace = Scatter(
            x=z_run_pca[:, 0],
            y=z_run_pca[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )

        data = Data([trace])
        layout = Layout(
            title='PCA on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        trace = Scatter(
            x=z_run_tsne[:, 0],
            y=z_run_tsne[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = Data([trace])
        layout = Layout(
            title='tSNE on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, download, folder_name):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        print("there are {} clusters".format(n_clusters_))
        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))
        colors = [hex_colors[int(i)] for i in labels]
        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)
        # SpectralClustering
        clustering_name = "GMM"
        # sc_model = SpectralClustering(n_clusters=n_clusters_)
        # y_pred = sc_model.fit_predict(z_run)

        # Kmean
        # kmeans_model = KMeans(n_clusters=n_clusters_, random_state=1)
        # y_pred = kmeans_model.fit_predict(z_run) #Labels of each point

        # # Mean Shift
        # meanshift_model = MeanShift(bandwidth=n_clusters_)
        # y_pred = meanshift_model.fit_predict(z_run)

        # GMM
        GMM_model = mixture.GaussianMixture(n_components=5, covariance_type="full")
        y_pred = GMM_model.fit_predict(z_run)

        # DBSCAN
        # DBSCAN_model = DBSCAN(eps = 0.1)
        # y_pred = DBSCAN_model.fit_predict(z_run)

        # # OPTICS
        # OPTICS_model = OPTICS(eps=0.8, min_samples=10)
        # y_pred = OPTICS_model.fit_predict(z_run)

        # cls = svm_latent_space(z_run, labels)
        # cls.predict()
        # clustering latent variable
        latent_dataset = {"z_run_pca": z_run_pca, "z_run_tsne": z_run_tsne}
        for name, z_run_sep in latent_dataset.items():

            # Kmean
            # kmeans_model = KMeans(n_clusters=n_clusters_, random_state=1)
            # y_pred = kmeans_model.fit_predict(z_run_sep) #Labels of each point

            # name
            # # Mean Shift
            # meanshift_model = MeanShift(bandwidth=n_clusters_)
            # y_pred = meanshift_model.fit_predict(z_run_sep)
            # SVM

            metrics.silhouette_score(z_run_sep, labels, metric='euclidean')
            labels_for_metrics = np.squeeze(labels)
            accuracy = metrics.adjusted_mutual_info_score(labels_for_metrics, y_pred)  # 完全一样则为1，也可能为0
            f1_final_score = f1_score(y_true=labels, y_pred=y_pred, average='weighted')
            recall_final_score = recall_score(y_true=labels, y_pred=y_pred, average='weighted')
            accuracy_final_score = accuracy_score(y_true=labels, y_pred=y_pred)
            precision_final_score = precision_score(y_true=labels, y_pred=y_pred, average='weighted')

            accuracy = round(accuracy, 2)
            f1_final_score = round(f1_final_score, 2)
            accuracy_final_score = round(accuracy_final_score, 2)
            precision_final_score = round(precision_final_score, 2)
            recall_final_score = round(recall_final_score, 2)
            print(
                "***************the accuracy of the clustering {} and dim_red {} is: {}************************".format(
                    "Kmeans", name, str(accuracy)))

            plt.scatter(z_run_sep[:, 0], z_run_sep[:, 1], c=y_pred)
            title = "predict_clustering_{} on {} ".format(clustering_name, name) + " Acc: " + str(
                accuracy) + " f1_score: {}, recall_score: {}, accuracy_score: {}, precision_score: {}.png".format(
                f1_final_score, recall_final_score, accuracy_final_score, precision_final_score)
            plt.title(title)
            if download:
                if os.path.exists(folder_name):
                    pass
                else:
                    os.mkdir(folder_name)
                plt.savefig(folder_name + "/" + title)
            else:
                plt.show()
            plt.scatter(z_run_sep[:, 0], z_run_sep[:, 1], c=colors, marker='*', linewidths=0)
            title = "Groundtruth on " + name + ".png"
            plt.title(title)
            if download:
                if os.path.exists(folder_name):
                    pass
                else:
                    os.mkdir(folder_name)
                plt.savefig(folder_name + "/" + title)
            else:
                plt.show()

        # if download:
        #     if os.path.exists(folder_name):
        #         pass
        #     else:
        #         os.mkdir(folder_name)
        #     plt.savefig(folder_name + "/pca.png")
        # else:
        #     plt.show()

        # plt.scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=colors, marker='*', linewidths=0)
        # plt.title('tSNE on z_run')
        # if download:
        #     if os.path.exists(folder_name):
        #         pass
        #     else:
        #         os.mkdir(folder_name)
        #     plt.savefig(folder_name + "/tsne.png")
        # else:
        #     plt.show()

    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name)


def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]

def open_newdata(direc, ratio_train=0.8, dataset="ECG200"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN.csv', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST.csv', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], :-1, :], data[ind[ind_cut:], :-1, :], data[ind[:ind_cut], -1, :], data[ind[ind_cut:], -1, :]

def open_newdata_ED(direc, ratio_train=0.8, dataset="ElectricDevices"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN.csv', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST.csv', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], :-1, :], data[ind[ind_cut:], :-1, :], data[ind[:ind_cut], -1, :], data[ind[ind_cut:], -1, :]

def cvs_to_numpy(direc, ratio_train=0.8, dataset="ECG5000"):
    datadir = direc + '/' + dataset + '/' + dataset


if __name__ == "__main__":
    cvs_to_numpy(direc='data', ratio_train=0.9, dataset="normalized")
