# Load packages
import numpy as np
import matplotlib.pyplot as plt

# Load modules
from toygrad.engine import Tensor
from toygrad.nn.layers import nnModule, Linear, Dropout
from toygrad.nn.data_utils import MNISTDataLoader


# - Fully Connected Network
class FCNetwork(nnModule):

    def __init__(self, layers_size, dropout_rate):
        super().__init__()
        # Store values
        self.layers_size = layers_size
        self.num_layers = len(layers_size)
        # Create linear layers
        self.linear_layers = []
        for dim_in, dim_out in zip(self.layers_size[:-1], self.layers_size[1:]):
            self.linear_layers += [Linear(dim_in, dim_out)]
        # Store parameters
        for layer in self.linear_layers:
            self.addParameter(layer.parameters[0])
            self.addParameter(layer.parameters[1])
        # Create dropout layer
        self.dropout = Dropout(dropout_rate)
        # Count number of learning parameters
        self.number_parameters = np.sum(p.item.size for p in self.parameters if p.requires_grad)
        print(f'The model has {self.number_parameters:,} learnable parameters.')
    
    def forward(self, input:np.array):
        """ input: (batch_size, dim_in)
            output (logits): (batch_size, dim_out) """
        # Applies dropout to all layers but the last one
        out = Tensor(input)
        for ind, lin_layer in enumerate(self.linear_layers):
            if ind != self.num_layers - 2:
                out = self.dropout(lin_layer(out))
            else:
                out = lin_layer(out)
        return out
    
    def accuracy(self, data, batch_size):
        # Create data_loader
        data_loader = MNISTDataLoader(data, batch_size)
        # Go over each of the batches
        accuracy, length = 0, 0
        for xb, yb in data_loader.get_batches():
            # Forward
            logits = self(xb)
            nn_prediction = np.argmax(logits.item, axis=-1)
            # Update
            accuracy += np.sum(nn_prediction == yb)
            length += len(yb)
        return accuracy*100/length
    
    def train_model(self, data, batch_size, optimizer, epoch_tol, save_model):
        """ data = train_data, validation_data, test_data
            epoch_tol : number of epochs without improving val_accuracy
            after the training is finalized
            save_model : name of the file where the model parameters are saved."""
        # Extract data and create data_loader
        train_data, validation_data, test_data = data
        train_loader = MNISTDataLoader(train_data, batch_size)
        # Initialize variables
        training_accuracy, validation_accuracy = [], []
        t_global_vec = []
        best_val_accuracy = 0
        t_global, counter_end = 0, 0
        # Loop over epochs
        while True:
            # Update t_global
            t_global += 1
            t_global_vec += [t_global]
            # Go over each of the batches
            for xb, yb in train_loader.get_batches():
                # Set grads to zero
                optimizer.zero_grad()
                # Forward
                logits = self(xb)
                y_prob = Tensor(np.eye(self.layers_size[-1])[yb]) # One-hot encoding
                # Compute Loss
                loss = logits.cross_entropy(y_prob)
                # Backprop and update parameters
                loss.backward()
                optimizer.step()

            # Compute accuracies
            training_accuracy += [self.accuracy(train_data, batch_size)]
            validation_accuracy += [self.accuracy(validation_data, batch_size)]

            # Save model if the val_accuracy has improved
            if validation_accuracy[-1] > best_val_accuracy:        
                # Update best_val_accuracy
                best_val_accuracy = validation_accuracy[-1]
                # Save model parameters
                np_parameters = [p.item for p in self.parameters]
                np.savez(save_model + '.npz', *np_parameters)
                # Update counter_end value
                counter_end = 0
                print(f"Validation Accuracy: {best_val_accuracy:.2f}")
            else:
                counter_end += 1
                if counter_end > epoch_tol:
                    break
                if counter_end == epoch_tol//3 or counter_end == 2*epoch_tol//3:
                    optimizer.lr *= 0.75

        # Load the best model and compute the test accuracy
        saved_parameters = np.load(save_model + '.npz')
        for ind, names in enumerate(saved_parameters):
            self.parameters[ind].item = saved_parameters[names]
        final_test_accuracy = self.accuracy(test_data, train_loader.batch_size)

        # Plot Accuracies
        plt.plot(t_global_vec, validation_accuracy, label="Validation")
        plt.plot(t_global_vec, training_accuracy, label="Training")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f'Final Test Accuracy: {final_test_accuracy:.2f}')
        plt.legend()
        plt.show()
        
        return t_global_vec, validation_accuracy, training_accuracy, final_test_accuracy