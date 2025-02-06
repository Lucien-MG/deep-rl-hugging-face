import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import (
    notebook_login,
) 

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from argparse import ArgumentParser

# Parse Args, username and token
parser = ArgumentParser()

parser.add_argument("-e", "--env", dest="env_id",
                    help="push to the huggingface user's repo")

# Main
if __name__ == '__main__':
    args = parser.parse_args()

    env_id = args.env_id

    model_name = "./models/ppo-" + env_id + "/best_model"
    model = PPO.load(model_name)

    eval_env = Monitor(gym.make(env_id))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=200, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
