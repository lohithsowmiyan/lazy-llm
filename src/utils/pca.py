import numpy as np


def PCA(X):
    X_meaned = X - np.mean(X, axis=0)

    cov_matrix = np.cov(X_meaned, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    X_pca = np.dot(X_meaned, sorted_eigenvectors)

    return X_pca, sorted_eigenvectors


