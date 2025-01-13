import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument(
    "-e", "--env", dest="env_id", help="push to the huggingface user's repo"
)
parser.add_argument(
    "-r", "--repo-id", dest="repo_id", help="Choose from which repo pull the model"
)
parser.add_argument(
    "-f",
    "--filename",
    dest="filename",
    help="Choose the model filename (if no repo id specifided look for local file))",
)


def eval_agent(env_id, model, n_eval_episodes):
    eval_env = Monitor(gym.make(env_id))
    return evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
    )


if __name__ == "__main__":
    args = parser.parse_args()

    if args.repo_id != None:
        from huggingface_sb3 import load_from_hub

        print("Loading model from hub...")
        model_filename = load_from_hub(args.repo_id, args.filename)
    else:
        print("Loading model from local...")
        model_filename = args.filename

    model = PPO.load(model_filename)

    print("Evaluating model...")
    mean_reward, std_reward = eval_agent(
        env_id=args.env_id, model=model, n_eval_episodes=400
    )

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
