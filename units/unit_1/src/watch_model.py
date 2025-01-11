import numpy
import gymnasium as gym

from stable_baselines3 import PPO

from huggingface_sb3 import load_from_hub

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-e", "--env", dest="env_id",
                    help="Choose the environment to run")

parser.add_argument("-r", "--repo-id", dest="repo_id",
                    help="Choose from which repo pull the model")

parser.add_argument("-f", "--filename", dest="filename",
                    help="Choose the model filename (if no repo id specifided look for local file))")

if __name__ == '__main__':
    args = parser.parse_args()

    if args.repo_id != None:
        print("Loading model from hub...")
        model_filename = load_from_hub(args.repo_id, args.filename)
    else:
        print("Loading model from local...")
        model_filename = args.filename

    model = PPO.load(model_filename)

    print(model)

    env = gym.make(args.env_id, render_mode="human")
    obs, info = env.reset()

    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        env.render()

    env.close()
