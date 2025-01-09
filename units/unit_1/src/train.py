import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-e", "--env", dest="env_id",
                    help="push to the huggingface user's repo")

# Main
if __name__ == '__main__':
    args = parser.parse_args()

    env_id = args.env_id
    model_save_path = "./models/ppo-" + env_id

    nb_steps = 1_600_000
    eval_freq = 10_000
    n_eval_episodes = 200
    n_training_envs = 32

    env = make_vec_env(env_id, n_envs=n_training_envs)
    eval_env = Monitor(gym.make(env_id))

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
        tensorboard_log="./logs/ppo_" + env_id + "/"
    )

    eval_callback = EvalCallback(eval_env, best_model_save_path=model_save_path,
                              log_path=model_save_path, eval_freq=max(eval_freq // n_training_envs, 1),
                              n_eval_episodes=n_eval_episodes, deterministic=True, render=False)

    model.learn(total_timesteps=nb_steps, callback=eval_callback)
