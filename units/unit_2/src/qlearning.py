import numpy as np

import pickle

from tqdm import trange

class Qlearning():

    def __init__(self, state_space, action_space):
        # initialized each values at 1, for optimistic exploration
        # this allow better exploration and faster convergence
        self.q_table = np.ones((state_space, action_space))

    def greedy_policy(self, state):
        # Exploitation: take the action with the highest state, action value
        action = np.argmax(self.q_table[state][:])

        return action

    def train(self, env, n_training_episodes, learning_rate, gamma, max_steps):
        for episode in trange(n_training_episodes):
            # Reduce epsilon (because we need less and less exploration)
            # Reset the environment
            state, info = env.reset()
            step = 0
            terminated = False
            truncated = False

            # repeat
            for step in range(max_steps):
                # Choose the action At using epsilon greedy policy
                action = self.greedy_policy(state)

                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                if terminated or truncated:
                    self.q_table[state][action] = self.q_table[state][action] + learning_rate * (reward - self.q_table[state][action])
                else:
                    self.q_table[state][action] = self.q_table[state][action] + learning_rate * (
                        reward + gamma * np.max(self.q_table[new_state]) - self.q_table[state][action]
                    )

                # If terminated or truncated finish the episode
                if terminated or truncated:
                    break

                # Our next state is the new state
                state = new_state

    def evaluate_agent(self, env, max_steps, n_eval_episodes, seed):
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
        :param env: The evaluation environment
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param Q: The Q-table
        :param seed: The evaluation seed array (for taxi-v3)
        """
        episode_rewards = []
        for episode in trange(n_eval_episodes):
            if seed:
                state, info = env.reset(seed=seed[episode])
            else:
                state, info = env.reset()
            step = 0
            truncated = False
            terminated = False
            total_rewards_ep = 0

            for step in range(max_steps):
                # Take the action (index) that have the maximum expected future reward given that state
                action = self.greedy_policy(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                total_rewards_ep += reward

                if terminated or truncated:
                    break
                state = new_state
            episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward
