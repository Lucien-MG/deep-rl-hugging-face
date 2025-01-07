from collections import deque

import numpy as np

import pickle

from tqdm import trange

class Qlearning():

    def __init__(self, state_space, action_space, epsilon_decay=1e-4, optimistic_value=1):
        """
        Qlearning: Q-learning algorithm.
        :param state_space: The size of the state space
        :param action_space: The size of the action space
        :param optimistic_value: Set default value in q_tabe (allow faster exploration)
        """
        self.epsilon_decay = 1e-4

        self.q_table = np.ones((state_space, action_space)) * optimistic_value

    def greedy_policy(self, state):
        # Exploitation: take the action with the highest state, action value
        action = np.argmax(self.q_table[state][:])

        return action

    def train(self, env, n_training_episodes, learning_rate, gamma, max_steps):
        cum_reward = deque([], maxlen=1000)
        epsilon = 0.99

        progress_bar = trange(n_training_episodes)
        for episode in progress_bar:
            # Reduce epsilon (because we need less and less exploration)
            epsilon = max(0.01, epsilon * (1 - self.epsilon_decay))

            # Reset the environment
            state, info = env.reset()
            cum_reward.append(0)

            # repeat
            for step in range(max_steps):
                # Choose the action At using epsilon greedy policy
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.greedy_policy(state)

                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)

                cum_reward[-1] += reward

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
            
            progress_bar.set_postfix({
                'epsilon': np.round(epsilon, 2),
                'reward': np.round(np.mean(cum_reward), 2)
            })

    def evaluate_agent(self, env, max_steps, n_eval_episodes, seed):
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
        :param env: The evaluation environment
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param Q: The Q-table
        :param seed: The evaluation seed array
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
