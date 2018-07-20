from __future__ import print_function, division

from sklearn.decomposition.pca import PCA
from sklearn.decomposition.nmf import NMF
import numpy as np

def pca(data, whiten_bool, components):
    # Set PCA parameters
    pca = PCA(n_components=components, whiten=whiten_bool, svd_solver="full")
    # Fit PCA to data
    pca.fit(data)
    np.set_printoptions(suppress=True)
    print("PCA Components Explained Variance Ratio: " + str(np.around(pca.explained_variance_ratio_ * 100, 2)))
    # Calculate loading matrix
    loadings_matrix = (pca.components_.T * np.sqrt(pca.explained_variance_)).T
    # Transform data
    data_transformed = pca.transform(data)
    return data_transformed


def nmf(data, components):
	nmf = NMF(n_components=components, init="random", random_state=0)
	W = nmf.fit_transform(data)
	H = nmf.components_
	R = np.dot(W,H)
	return W