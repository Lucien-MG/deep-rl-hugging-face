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
    n_eval_episodes = 100

    # Pickle the model
    model_path = "./models/q-learning-" + args.env_id  + ".pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    push_model(args.username, args.token, "qlearning", agent.q_table, args.env_id, n_training_episodes, n_eval_episodes, eval_seed, learning_rate)
