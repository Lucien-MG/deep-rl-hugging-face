import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-e", "--env", dest="env_id",
                    help="push to the huggingface user's repo")

if __name__ == '__main__':
    args = parser.parse_args()

    env_name = args.env_id
    model_name = "./models/ppo-" + env_name + "/best_model"

    env = gym.make(env_name, render_mode="human")
    model = PPO.load(model_name)

    obs, info = env.reset()

    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        env.render()

    env.close()
