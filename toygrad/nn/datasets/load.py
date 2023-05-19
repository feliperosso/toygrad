"""
load.py

Loads the MNIST dataset
"""

# Load packages
import os, toygrad, gzip, pickle

# - load MNIST data -
def load_MNIST():
    """ Load the MNIST dataset, already included in the package.

        - train (50k): (x_train, y_train) 
        - validation (10k): (x_validation, y_validation)
        - test (10k): (x_test, y_test)

        x images: (num_elements, 784) np.array
        y labels: (num_elements, ) np.array
    """
    # Locate folder path
    toygrad_path = os.path.dirname(toygrad.__file__)
    folder_path = toygrad_path + '/nn/datasets/mnist.pkl.gz'
    # Load data
    with gzip.open(folder_path, 'rb') as file:
        data = pickle._Unpickler(file, encoding='latin1').load()
        train, validation, test = data
    return train, validation, test