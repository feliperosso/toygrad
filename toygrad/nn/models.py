""" models.py 

This modules defines the following two models:

1. MNISTNetwork: A simple fully connected network with Dropout
and ReLU activation functions which can be used to classify the MNIST dataset.

2. RLPolicyGradient: Defines a policy from a fully connected network which can 
be trained using two variants of Vanilla Policy Gradient or PPO. It also contains 
a value estimation network which is initialized with the policy and used during 
training.

"""

# Load packages
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm.auto import tqdm

# Load modules
from toygrad.engine import Tensor
from toygrad.nn.layers import nnModule, Linear, Dropout
from toygrad.nn.data_utils import MNISTDataLoader

# - Value Estimation Network -
class ValueEstimationNet(nnModule):

    def __init__(self, obs_space_dim: int, hid_dim=128):
        super().__init__()
        # Store values
        self.obs_space_dim = obs_space_dim
        self.layers_size = [self.obs_space_dim, hid_dim, 1]
        self.num_layers = len(self.layers_size)
        # Create linear layers
        self.linear_layers = []
        for dim_in, dim_out in zip(self.layers_size[:-1], self.layers_size[1:]):
            self.linear_layers += [Linear(dim_in, dim_out)]
        # Store parameters
        for layer in self.linear_layers:
            self.addParameter(layer.parameters[0])
            self.addParameter(layer.parameters[1])
    
    def forward(self, input: np.array):
        """ input: (batch_size, dim_in)
            output (logits): (batch_size, dim_out) """
        out = Tensor(input)
        for ind, lin_layer in enumerate(self.linear_layers):
            out = lin_layer(out).relu() if ind < self.num_layers - 2 else lin_layer(out)
        return out
    
    def train_value_net(self, batch_obs: list, batch_cum_rewards: list,
                             value_optimizer, value_epochs, clip_eps=0.2):
        """ Train the value network by minimizing a clipped version of the norm 
        squared (arXiv:1811.02553v3) and applying early stopping (arxiv:1506.02438) """
        # Compute value net prediction for network before update
        batch_len = len(batch_obs)
        in_value = self(np.array(batch_obs)).reshape(batch_len)
        in_value.requires_grad = False
        # Compute sigma^2 for early stopping
        in_diff = in_value - Tensor(np.array(batch_cum_rewards))
        sigma2 = in_diff.matmul(in_diff)/Tensor(batch_len)
        for _ in range(value_epochs):
            value_optimizer.zero_grad()
            # Feedforward
            value = self(np.array(batch_obs)).reshape(batch_len)
            diff = value - Tensor(np.array(batch_cum_rewards))
            # Compute loss
            raw_loss = diff.matmul(diff)/Tensor(batch_len)
            clip = value.maximum(in_value - Tensor(clip_eps)).minimum(in_value + Tensor(clip_eps))
            clip_diff = clip - Tensor(np.array(batch_cum_rewards))
            clip_loss = clip_diff.matmul(clip_diff)/Tensor(batch_len)
            batch_loss = raw_loss.minimum(clip_loss)
            # Backward and update
            batch_loss.backward()
            value_optimizer.step()
            # Early stopping?
            early_diff = ((value - in_value).matmul(value - in_value))/(Tensor(2*batch_len)*sigma2)
            if early_diff.item > clip_eps:
                break
    
    def cum_sum(self, x_t: list, discount: float):
        """ Performs the cumulative sum of x_t damped by discount 
            out = sum_k (discount^k*x_{t + k}): list"""
        length = len(x_t)
        out, future = [0]*length, 0
        for k in range(length)[::-1]:
            future = discount*out[k + 1] if k + 1 < length else 0
            out[k] = x_t[k] + future
        return out
    
    def est_advantage(self, batch_obs: list, batch_rewards: list, lmbda: float, gamma: float):
        """ Computes the estimation to the GAE advantage """
        # Construct delta_t
        r_t = np.array(batch_rewards)
        Vs_t = self(batch_obs)[:, 0].item
        Vs_tm1 = np.append(Vs_t[1:], np.array([0]))

        delta_t = r_t + gamma*Vs_tm1 - Vs_t
        # Compute GAE advantage
        gae_adv = np.array(self.cum_sum(delta_t, gamma*lmbda))        
        return gae_adv

# - Reinforced Learning Policy Gradient -
class RLPolicyGradient(nnModule):

    def __init__(self, env: gym.Env, obs_space_dim: int, act_space_dim, 
                       hid_dim: int=128, gamma: int=0.99):
        super().__init__()
        # Store env
        self.env = env
        self.obs_space_dim = obs_space_dim
        self.action_space_dim = act_space_dim
        self.gamma = gamma

        # - Initialize Policy network -
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
        
        # - Initialize Value Estimation Network -
        self.value_net = ValueEstimationNet(obs_space_dim)
    
    def forward(self, input: np.ndarray):
        """ input: (*, dim_in=self.obs_space_dim)
            output (logits): (*, dim_out=self.action_space_dim) """
        out = Tensor(input)
        for k, lin_layer in enumerate(self.linear_layers):
            out = lin_layer(out).relu() if k < self.num_layers - 2 else lin_layer(out)
        return out
    
    def get_batch(self, batch_size: int):
        """ Plays batch_size number of episodes and returns:
            - batch_obs: list with the observed env for all iterations.
            - batch_actions: list with the actions taken for all iterations. 
            - batch_rewards: list with the raw rewards for all iterations
            - batch_cum_rewards: list with the cumulative rewards for all iterations.
            """
        # Create bookeeping lists
        batch_obs, batch_actions = [], []
        batch_rewards, batch_cum_rewards = [], []
        # Play batch_size episodes
        for _ in range(batch_size):
            ep_obs, ep_actions, ep_rewards = [], [], []
            obs = self.env.reset()[0]
            ep_done = False
            # Play one episode
            while not ep_done:
                if isinstance(obs, int):
                    ep_obs.append([obs])
                else:
                    ep_obs.append(list(obs))
                # Sample action from self
                policy_prob = self(obs).softmax(-1)
                action = np.random.choice(list(range(self.action_space_dim)), p=policy_prob.item)
                # Take action on environment
                obs, reward, terminated, truncated, _ = self.env.step(action)
                # Store stuff
                ep_actions.append(action)
                ep_rewards.append(reward)
                # Update done variable
                ep_done = terminated or truncated
            
            # Compute cumulative rewards
            ep_cum_rewards = self.value_net.cum_sum(ep_rewards, self.gamma)
            # Store element in batch
            batch_obs += ep_obs
            batch_actions += ep_actions
            batch_rewards += ep_rewards
            batch_cum_rewards += ep_cum_rewards
            
        return batch_obs, batch_actions, batch_rewards, batch_cum_rewards
    
    def compute_return_weight(self, batch_obs, batch_rewards, batch_cum_rewards, value_est: bool,
                                    value_optimizer, value_epochs, lmbda, clip_eps):
        """ Computes the weight used in the return (which is then maximized by updating policy)
            - If value_est = True, it uses the advantage as the weight, computed using the
            value estimation network.
            - If value_est = False, it uses the batch cumulative reward as the weight."""
        if value_est:
            # - Train Value Estimation Net and estimate advantage -
            self.value_net.train_value_net(batch_obs, batch_cum_rewards, 
                                           value_optimizer, value_epochs, clip_eps)
            if lmbda == 1.0:
                gae_adv = np.array(batch_cum_rewards) - self.value_net(batch_obs)[:, 0].item
            else:
                gae_adv = self.value_net.est_advantage(batch_obs, batch_rewards, lmbda, self.gamma)
            weight = Tensor((gae_adv - gae_adv.mean())/gae_adv.std()) # Normalize
        else:
            # - Use cumulative rewards as weight -
            weight = Tensor(np.array(batch_cum_rewards))
        return weight
    
    def policy_update(self, batch_obs: list, batch_actions: list, weight: Tensor,
                            policy_optimizer, PPO=False, PPO_epochs=10, clip_eps=0.2, kl_eps=0.025):
        if not PPO:
            """ Updates policy parameters using the vanilla procedure, minimizing
            the return return = act_log_prob*weight """
            policy_optimizer.zero_grad()
            batch_len = len(batch_obs)
            # Feedforward
            logits = self(np.array(batch_obs))
            act_log_prob = logits.softmax().log()[range(batch_len), batch_actions]
            batch_return = act_log_prob.matmul(weight)/Tensor(batch_len)
            # Backprop and update to maximize batch_return
            batch_return.backward()
            policy_optimizer.step()
        if PPO:
            """ Updates policy parameters using the PPO procedure"""
            # Compute the initial policy prob of the taken actions
            batch_len = len(batch_obs)
            in_prob = self(np.array(batch_obs)).softmax()
            in_act_prob = in_prob[range(batch_len), batch_actions]
            in_act_prob.requires_grad, weight.requires_grad = False, False
            # Compute 
            for _ in range(PPO_epochs):
                policy_optimizer.zero_grad()
                # Feedforward
                prob = self(np.array(batch_obs)).softmax()
                act_prob = prob[range(batch_len), batch_actions]
                prob_ratio = act_prob/in_act_prob
                # Compute return
                raw_return = prob_ratio.matmul(weight)/Tensor(batch_len)
                clip = prob_ratio.maximum(Tensor(1 - clip_eps)).minimum(Tensor(1 + clip_eps))
                clipped_return = clip.matmul(weight)/Tensor(batch_len)
                batch_return = raw_return.minimum(clipped_return)
                # Backprop and update to maximize batch_return
                batch_return.backward()
                policy_optimizer.step()
                # Early stopping?
                ave_kl_entropy = (prob*((prob/in_prob).log())).sum()/Tensor(batch_len)
                if ave_kl_entropy.item > kl_eps:
                    break
                
    def train_policy(self, batch_size: int, policy_epochs: int, policy_optimizer, 
                           value_est: bool=False, value_optimizer=None, value_epochs: int=10, lmbda: float=1.0,
                           PPO: bool=False, PPO_epochs=10, clip_eps=0.2, kl_eps=0.025,
                           smooth_window: int=10):
        """ Train the policy using vanilla policy gradient
            epoch_rewards: list with average episode reward for each epoch
            smooth_reward: smoothened version of epoch_rewards according to smooth_window input
            value_est: If True the value estimation is used to compute the advantage and used as weight
                       If False the cumulative reward is used as the weight"""
        epoch_rewards = []
        prog_bar = tqdm(range(policy_epochs), total=policy_epochs)
        for t_epoch in prog_bar:
            # Play batch_size number of episodes, store rewards, update prog_bar
            batch_obs, batch_actions, batch_rewards, batch_cum_rewards = self.get_batch(batch_size)
            epoch_rewards.append(sum(batch_rewards)/batch_size)
            prog_bar.set_description(f"Epoch {t_epoch + 1} average reward: {epoch_rewards[-1]:.2f}.")
            # - Compute weights for batch return -
            weight = self.compute_return_weight(batch_obs, batch_rewards, batch_cum_rewards, 
                                                value_est, value_optimizer, value_epochs, lmbda, clip_eps)
            # - Update policy -
            self.policy_update(batch_obs, batch_actions, weight, policy_optimizer,
                               PPO, PPO_epochs, clip_eps, kl_eps)

        # Test the final policy by playing batch_size episodes
        out = self.get_batch(batch_size)
        test_mean_reward = np.array(out[-2]).sum()/batch_size
        # Create smoothened version of rewards
        smooth_reward = self.build_smooth_reward(policy_epochs, epoch_rewards)

        # Plot rewards
        plt.plot(epoch_rewards)
        plt.plot(smooth_reward)
        plt.axhline(y=200, color='black', linestyle='dashed', label='Win game')
        plt.axhline(y=test_mean_reward, color='green', linestyle='dashed', label='Test mean reward')
        plt.xlabel('Epochs')
        plt.ylabel('Average Episode Reward')
        plt.title(f'Mean test reward from playing {batch_size} episodes: {test_mean_reward:.2f}.')
        plt.legend()
        plt.show()
        return epoch_rewards, smooth_reward

    def build_smooth_reward(self, policy_epochs: int, epoch_rewards, smooth_window: int=10):
        """ It creates a smooth version of the epoch_rewards by averaging over smooth_window"""
        smooth_reward = []
        for k in range(policy_epochs):
            if k < smooth_window:
                temp = epoch_rewards[:k + 1]
                smooth_reward.append(sum(temp)/len(temp)) 
            elif k > policy_epochs - smooth_window:
                temp = epoch_rewards[k:]
                smooth_reward.append(sum(temp)/len(temp)) 
            else:
                temp = epoch_rewards[k + 1 - smooth_window//2:k + 1 + smooth_window//2]
                smooth_reward.append(sum(temp)/len(temp)) 
        return smooth_reward

    def play(self, num_episodes: int, env_render: gym.Env, 
                   save=False, filename=None, path='./'):
        """ env_render is the environment initialized with render_mode='rgb_array_list'
            If save=True, a gif is created and saved with the filename 
        """
        # Play batch_size episodes and get frames
        frames, ep_rewards = [], []
        for _ in range(num_episodes):
            obs = env_render.reset()[0]
            ep_done, temp_rewards = False, []
            # Play one episode
            while not ep_done:
                # Sample action from policy
                policy_prob = self(obs).softmax(-1)
                action = np.random.choice(list(range(self.action_space_dim)), p=policy_prob.item) # Sample
                # Take action on environment and store frames
                obs, reward, terminated, truncated, _ = env_render.step(action)
                # Update done variable
                temp_rewards.append(reward)
                ep_done = terminated or truncated
            frames += env_render.render()
            ep_rewards.append(sum(temp_rewards))
        
        # Generate a gif with the frames and save. The code below was obtained from:
        # https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
        if save:
            plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
            patch = plt.imshow(frames[0])
            plt.axis('off')

            def animate(i):
                patch.set_data(frames[i])
            anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
            anim.save(path + filename + '.gif', fps=60)
        
        return frames, ep_rewards

# - Fully Connected Network
class MNISTNetwork(nnModule):

    def __init__(self, layers_size: list, dropout_rate: float):
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
    
    def forward(self, input: np.array):
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
    
    def accuracy(self, data, batch_size: int):
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
    
    def train_model(self, data, batch_size: int, optimizer, epoch_tol: int, save_model: str):
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
                loss.backward(np.array(1))
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
                print(f"Validation Accuracy: {best_val_accuracy:.2f}", end='\r')
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