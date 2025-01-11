import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-e", "--env", dest="env_id",
                    help="push to the huggingface user's repo")

def eval_agent(env_id, model, n_eval_episodes):
    eval_env = Monitor(gym.make(env_id))
    return evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True)

if __name__ == '__main__':
    args = parser.parse_args()

    model_name = "./models/ppo-" + env_id + "/best_model"
    model = PPO.load(model_name)

    eval_agent(env_id=args.env_id)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
