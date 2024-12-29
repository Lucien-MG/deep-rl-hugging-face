import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

env_name = "LunarLander-v3"

nb_steps = 1_000_000
eval_freq = 10_000

# 8 give faster results (~20 min on cpu) but less stable
# 16 takes longer (~3 hours on cpu) but have better generalisation and stability
n_training_envs = 16

model_save_path = "./models/ppo-" + env_name

env = make_vec_env(env_name, n_envs=n_training_envs)
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-3,
    n_steps=512,
    batch_size=32,
    n_epochs=3,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=0,
    tensorboard_log="./tensorboard/ppo_lunarlander/"
)

eval_env = Monitor(gym.make(env_name))
eval_callback = EvalCallback(eval_env, best_model_save_path=model_save_path,
                              log_path=model_save_path, eval_freq=max(eval_freq // n_training_envs, 1),
                              n_eval_episodes=200, deterministic=True, render=False)

model.learn(total_timesteps=nb_steps, callback=eval_callback)
