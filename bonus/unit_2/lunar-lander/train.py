import optuna
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from argparse import ArgumentParser

# Parse Args, username and token
parser = ArgumentParser()

parser.add_argument("-e", "--env", dest="env_id",
                    help="push to the huggingface user's repo")

def objective(trial):
    nb_steps = 100_000

    eval_freq = 20_000
    n_eval_episodes = 100

    n_training_envs = trial.suggest_int('n_training_envs', 2, 12),

    model_save_path = "./models/ppo-" + args.env_id

    env = make_vec_env(env_id, n_envs=n_training_envs)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2),
        n_steps=512,
        batch_size=trial.suggest_int('batch_size', 16, 128),
        n_epochs=3,
        gamma=trial.suggest_float('gamma', 0.99, 0.9999),
        gae_lambda=trial.suggest_float('gae_lambda', 0.95, 0.99),
        ent_coef=trial.suggest_float('ent_coef', 0.001, 0.02),
        verbose=0,
        tensorboard_log="./logs/ppo_" + env_id + "/"
    )

    eval_env = Monitor(gym.make(env_id))
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_save_path,
                              log_path=model_save_path, eval_freq=max(eval_freq // n_training_envs, 1),
                              n_eval_episodes=n_eval_episodes, deterministic=True, render=False)

    model.learn(total_timesteps=nb_steps, callback=eval_callback)

    return accuracy

# Main
if __name__ == '__main__':
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
