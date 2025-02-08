import os

import gymnasium as gym
import panda_gym

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env


from argparse import ArgumentParser
# from huggingface_hub import notebook_login


parser = ArgumentParser()

parser.add_argument("-e", "--env", dest="env_id", help="select the environment")
parser.add_argument(
    "-u", "--username", dest="username", help="push to the huggingface user's repo"
)
parser.add_argument(
    "-t", "--token", dest="token", help="token to authenticate to hugging face repos"
)

args = parser.parse_args()





env_id = "PandaReachDense-v3"

# Create the env
env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape
a_size = env.action_space

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

env = make_vec_env(env_id, n_envs=16)

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = A2C(policy = "MultiInputPolicy",
            env = env,
            learning_rate=0.003,
            verbose=1,
            tensorboard_log="./logs/ppo_" + env_id + "/",)

model.learn(100_000)

# Save the model and  VecNormalize statistics when saving the agent
model.save("a2c-PandaReachDense-v3")
env.save("vec_normalize.pkl")

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Load the saved statistics
eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

# We need to override the render_mode
eval_env.render_mode = "rgb_array"

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

# Load the agent
model = A2C.load("a2c-PandaReachDense-v3")

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

package_to_hub(
    model=model,
    model_name=f"a2c-{env_id}",
    model_architecture="A2C",
    env_id=env_id,
    eval_env=eval_env,
    repo_id=args.username + f"/a2c-{env_id}",
    commit_message="Initial commit",
)