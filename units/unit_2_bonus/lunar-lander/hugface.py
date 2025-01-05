import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from huggingface_sb3 import package_to_hub
from argparse import ArgumentParser

# Parse Args, username and token
parser = ArgumentParser()

parser.add_argument("-e", "--env", dest="env_id",
                    help="select the environment")
parser.add_argument("-u", "--username", dest="username",
                    help="push to the huggingface user's repo")
parser.add_argument("-t", "--token", dest="token",
                    help="token to authenticate to hugging face repos")

if __name__ == '__main__':
    args = parser.parse_args()

    # Vars
    env_id = args.env_id

    model_name = "./models/ppo-" + env_id + "/best_model"
    model_architecture = "PPO"
    model = PPO.load(model_name)

    repo_id = args.username + "/" + env_id
    commit_message = "Upload PPO " + env_id + " trained agent."

    print("Push to repo:", repo_id)

    # Create the evaluation env and set the render_mode="rgb_array"
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

    package_to_hub(model=model, # Our trained model
                   model_name=model_name, # The name of our trained model
                   model_architecture=model_architecture, # The model architecture we used: in our case PPO
                   env_id=env_id, # Name of the environment
                   eval_env=eval_env, # Evaluation Environment
                   repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name}
                   commit_message=commit_message,
                   token=args.token)
