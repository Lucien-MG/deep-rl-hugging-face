import pickle
import gymnasium as gym
from tqdm import trange

from argparse import ArgumentParser

from qlearning import Qlearning
from seeds import eval_seed
from hugface import push_model

# Parse Args, username and token
parser = ArgumentParser()

parser.add_argument("-e", "--env", dest="env_id",
                    help="push to the huggingface user's repo")
parser.add_argument("-u", "--username", dest="username",
                    help="push to the huggingface user's repo")
parser.add_argument("-t", "--token", dest="token",
                    help="token to authenticate to hugging face repos")

if __name__ == '__main__':
    args = parser.parse_args()

    env = gym.make(args.env_id, render_mode="rgb_array")

    n_training_episodes = 150_000
    max_steps = 100
    learning_rate = 0.1
    gamma = 0.99
    n_eval_episodes = 100
    optimistic_value = 1

    agent = Qlearning(env.observation_space.n, env.action_space.n, optimistic_value=optimistic_value)

    print("training...")
    agent.train(env, n_training_episodes, learning_rate, gamma, max_steps)

    print("evaluating...")
    mean_reward, std_reward = agent.evaluate_agent(env, max_steps, n_eval_episodes, eval_seed)

    # Pickle the model
    model_path = "./models/q-learning-" + args.env_id  + ".pkl"
    with open(model_path, "wb") as f:
        pickle.dump(agent.q_table, f)
    print("model saved to", model_path)

    print("mean_reward:", mean_reward)
    print("std_reward:", std_reward)
    print("score:", mean_reward - std_reward)
