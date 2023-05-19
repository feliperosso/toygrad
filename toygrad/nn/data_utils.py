""" data_utils.py """

# Load packages
import numpy as np

class MNISTDataLoader:

    def __init__(self, data, batch_size):
        # Store data
        self.values, self.target = data
        self.data_size = len(self.target)
        # Split into batches parameters
        self.batch_size = batch_size
        self.num_batches = self.data_size//self.batch_size
        self.extra_elements = self.data_size % self.num_batches
        if self.extra_elements != 0:
            print("Caution: Data size not divisible by batch size," \
                    f"last {self.extra_elements} elements are dropped.")
    
    def get_batches(self):
        # Shuffle
        permutations = np.random.permutation(self.data_size)
        x, y = self.values[permutations], self.target[permutations]
        # Split into batches
        if self.extra_elements == 0:
            x_split = x.reshape(self.num_batches, self.batch_size, -1)
            y_split = y.reshape(self.num_batches, self.batch_size)
        else:
            x_split = x[:-self.extra_elements].reshape(self.num_batches, self.batch_size, -1)
            y_split = y[:-self.extra_elements].reshape(self.num_batches, self.batch_size)
        return zip(x_split, y_split)