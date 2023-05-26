# Load packages
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Load modules
from toygrad.engine import Tensor
from toygrad.nn.layers import nnModule, Linear, Dropout
from toygrad.nn.data_utils import MNISTDataLoader

# - RL Policy for discrete/obs and discrete/actions environments -
class ddPolicy(nnModule):

    def __init__(self, env, hid_dim=128):
        super().__init__()
        # Store parameters
        self.env = env
        self.obs_space_dim = env.observation_space.shape[0]
        self.action_space_dim = env.action_space.n
        self.layers_size = [self.obs_space_dim, hid_dim, self.action_space_dim]
        self.num_layers = len(self.layers_size)        
        # Create linear layers
        self.linear_layers = []
        for dim_in, dim_out in zip(self.layers_size[:-1], self.layers_size[1:]):
            self.linear_layers += [Linear(dim_in, dim_out)]
        # Store parameters
        for layer in self.linear_layers:
            self.addParameter(layer.parameters[0])
            self.addParameter(layer.parameters[1])
    
    def forward(self, input:np.ndarray):
        """ input: (*, dim_in=self.obs_space_dim)
            output (logits): (*, dim_out=self.action_space_dim) """
        out = Tensor(input)
        for num, lin_layer in enumerate(self.linear_layers):
            if num != self.num_layers - 2:
                out = lin_layer(out).relu()
            else:
                out = lin_layer(out)
        return out
    
    def get_batch(self, batch_size: int, gamma: int=0.99):
        """ Uses the self to play batch_size number of episodes and returns:
            - batch_obs: list with the observed env for all iterations.
            - batch_actions: list with the actions taken for all iterations. 
            - batch_cum_rewards: list with the cumulative rewards for all iterations.
            - batch_ep_rewards: sum of rewards over each of the batch_size number of episodes"""
        # Create bookeeping lists
        batch_obs, batch_actions = [], []
        batch_cum_rewards, batch_ep_rewards = [], []
        # Play batch_size episodes
        for _ in range(batch_size):
            ep_obs, ep_actions, ep_rewards = [], [], []
            obs = self.env.reset()[0]
            ep_done = False
            # Play one episode
            while not ep_done:
                ep_obs.append(obs.tolist())
                # Sample action from self
                policy_prob = self(obs).softmax(-1)
                action = np.random.choice(list(range(self.action_space_dim)), p=policy_prob.item) # Sample
                # Take action on environment
                obs, reward, terminated, truncated, _ = self.env.step(action)
                # Store stuff
                ep_actions.append(action)
                ep_rewards.append(reward)
                # Update done variable
                ep_done = terminated or truncated
            
            # Compute cumulative rewards
            ep_length = len(ep_rewards)
            ep_cum_rewards, future_rewards = [0]*ep_length, 0
            for ind in range(ep_length)[::-1]:
                future_rewards = gamma*ep_cum_rewards[ind + 1] if ind + 1 < ep_length else 0
                ep_cum_rewards[ind] = ep_rewards[ind] + future_rewards
            # Store element in batch
            batch_obs += ep_obs
            batch_actions += ep_actions
            batch_cum_rewards += ep_cum_rewards
            batch_ep_rewards.append(sum(ep_rewards))
            
        return batch_obs, batch_actions, batch_cum_rewards, batch_ep_rewards
    
    def train_policy(self, epochs: int, batch_size: int, optimizer, smooth_window: int=10):
        """ Train the policy using vanilla policy gradient
            epoch_rewards: list with average episode reward for each epoch
            smooth_reward: smoothened version of epoch_rewards according to smooth_window input"""
        epoch_rewards = []
        prog_bar = tqdm(range(epochs), total=epochs)
        for t_epoch in prog_bar:
            # Play batch_size number of episodes
            out = self.get_batch(batch_size)
            batch_obs, batch_actions, batch_cum_rewards, batch_ep_rewards = out
            # - Update policy -
            optimizer.zero_grad()
            batch_len = len(batch_obs)
            # Feedforward
            logits = self(np.array(batch_obs))
            act_log_prob = logits.softmax().log()[range(batch_len), batch_actions]
            batch_return = act_log_prob.matmul(Tensor(np.array(batch_cum_rewards)))/Tensor(np.array(batch_len))
            # Backprop and update to maximize batch_return
            batch_return.backward()
            optimizer.step()
            # Store stuff
            batch_ave_reward = sum(batch_ep_rewards)/batch_size
            epoch_rewards.append(batch_ave_reward)
            prog_bar.set_description(f"Epoch {t_epoch + 1} average reward: {epoch_rewards[-1]:.2f}")

        # Test the final policy by playing batch_size episodes
        out = self.get_batch(batch_size)
        test_batch_rewards = np.array(out[-1])
        test_mean_reward = test_batch_rewards.mean()
        test_std_reward = test_batch_rewards.std()
        # Create smoothened version of rewards
        smooth_reward = []
        for k in range(epochs):
            temp = epoch_rewards[:k + 1] if k < smooth_window else epoch_rewards[k + 1 - smooth_window:k + 1]
            smooth_reward.append(sum(temp)/len(temp)) 
        # Plot rewards
        plt.plot(epoch_rewards)
        plt.plot(smooth_reward)
        plt.xlabel('Epochs')
        plt.ylabel('Average Episode Reward')
        plt.title(f'Test {batch_size} episodes. Mean reward: {test_mean_reward:.2f}. Mean std: {test_std_reward:.2f}')
        plt.show()
        
        return epoch_rewards, smooth_reward


# - RL Policy for Pong -
class PongPolicy(nnModule):

    def __init__(self, hid_dim):
        super().__init__()
        # Store values
        self.layers_size = [80*64, hid_dim, 2]
        self.num_layers = len(self.layers_size)        
        # Create linear layers
        self.linear_layers = []
        for dim_in, dim_out in zip(self.layers_size[:-1], self.layers_size[1:]):
            self.linear_layers += [Linear(dim_in, dim_out)]
        # Store parameters
        for layer in self.linear_layers:
            self.addParameter(layer.parameters[0])
            self.addParameter(layer.parameters[1])
        # Count number of learning parameters
        self.number_parameters = np.sum(p.item.size for p in self.parameters if p.requires_grad)
        print(f'The policy has {self.number_parameters:,} learnable parameters.')
    
    def forward(self, input:np.ndarray):
        """ input: (*, dim_in=80*64)
            output (logits): (*, dim_out=2) """
        out = Tensor(input)
        for num, lin_layer in enumerate(self.linear_layers):
            if num != self.num_layers - 2:
                out = lin_layer(out).relu()
            else:
                out = lin_layer(out)
        return out
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0
    
    def pre_process(self, cur_obs, prev_obs=None):
        """ Pre-process the observation given by env.
            cur_obs: (210, 160, 3) with current observation of env
            prev_obs: (80*64, ) with previous pre_processed observation of env

            policy_input: (80*64, ) with input to be fed into policy
            cur_obs: (80*64, ) pre_processed current obs to be used in next iteration
            """
        # - Pre_process current observation -
        # Extract one RGB channel and crop
        cur_obs = cur_obs[:, :, 0]
        cur_obs = cur_obs[35:195, 16:144]
        # Reduce resolution
        cur_obs = cur_obs[::2, ::2]
        # Set background to zero
        background_value1 = 144
        cur_obs[cur_obs == background_value1] = 0
        background_value2 = 109
        cur_obs[cur_obs == background_value2] = 0
        # Set ball and paddles to one
        cur_obs[cur_obs != 0] = 1
        # Change to float elements and flatten
        cur_obs = cur_obs.astype(float).flatten()
        # -

        # Output
        policy_input = np.zeros(80*64) if prev_obs is None else cur_obs - prev_obs
        return policy_input, cur_obs
    
    def play(self, tot_num_points:int):
        """ Play the game for tot_num_points """
        # Initialize environment
        env = gym.make("ALE/Pong-v5", render_mode="human")
        cur_obs = env.reset()[0]
        prev_obs = None

        points_played = 0
        while True:
            # Apply policy
            policy_input, prev_obs = self.pre_process(cur_obs, prev_obs)
            policy_prob = self(policy_input).softmax(axis=-1).item
            action = np.random.choice([2, 3], p=policy_prob)
            # Compute action
            cur_obs, reward, terminated, truncated, _ = env.step(action)
            
            if reward != 0:
                points_played += 1
                print(reward, terminated)
            if points_played == tot_num_points:
                break
            if terminated or truncated:
                cur_obs = env.reset()[0]
                
        env.close()
    

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
                out = self.dropout(lin_layer(out).relu())
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