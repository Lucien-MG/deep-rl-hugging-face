import os
import json
import zipfile
import gymnasium as gym

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
from argparse import ArgumentParser

from eval import eval_agent

def _generate_config(model, local_path: Path) -> None:
    """
    Generate a config.json file containing information
    about the agent and the environment
    :param model: name of the model zip file
    :param local_path: path of the local directory
    """
    unzipped_model_folder = model

    # Check if the user forgot to mention the extension of the file
    if model.endswith(".zip") is False:
        model += ".zip"

    # Step 1: Unzip the model
    with zipfile.ZipFile(local_path / model, "r") as zip_ref:
        zip_ref.extractall(local_path / unzipped_model_folder)

    # Step 2: Get data (JSON containing infos) and read it
    with open(Path.joinpath(local_path, unzipped_model_folder, "data")) as json_file:
        data = json.load(json_file)
        # Add system_info elements to our JSON
        data["system_info"] = stable_baselines3.get_system_info(print_info=False)[0]

    # Step 3: Write our config.json file
    with open(local_path / "config.json", "w") as outfile:
        json.dump(data, outfile)

def _save_model_card(
    local_path: Path, generated_model_card: str, metadata
):
    """Saves a model card for the repository.
    :param local_path: repository directory
    :param generated_model_card: model card generated by _generate_model_card()
    :param metadata: metadata
    """
    readme_path = local_path / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = generated_model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

def generate_metadata(
    model_name: str, env_id: str, mean_reward: float, std_reward: float
):
    """
    Define the tags for the model card
    :param model_name: name of the model
    :param env_id: name of the environment
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    """
    metadata = {}
    metadata["library_name"] = "stable-baselines3"
    metadata["tags"] = [
        env_id,
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "stable-baselines3",
    ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=model_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_id,
        dataset_id=env_id,
    )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    return metadata


def generate_model_card(
    model_name: str, env_id: str, mean_reward: float, std_reward: float
):
    """
    Generate the model card for the Hub
    :param model_name: name of the model
    :env_id: name of the environment
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    """
    # Step 1: Select the tags
    metadata = generate_metadata(model_name, env_id, mean_reward, std_reward)

    # Step 2: Generate the model card
    model_card = f"""
# **{model_name}** Agent playing **{env_id}**
This is a trained model of a **{model_name}** agent playing **{env_id}**
using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3).
"""

    model_card += """
## Usage (with Stable-baselines3)

```python
import gymnasium as gym

from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

model_filename = load_from_hub(your_repo_id, your_filename)

model = PPO.load(model_filename)

env = gym.make({env_id}, render_mode="human")
    obs, info = env.reset()

    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        env.render()

    env.close()
```
"""

    return model_card, metadata


def generate_replay(env, model, directory):
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder=directory,
        name_prefix="replay",
        episode_trigger=lambda x: x % 2 == 0,
    )
    obs, info = env.reset()

    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        env.render()

    env.close()
    os.rename(directory / 'replay-episode-0.mp4', directory / 'replay.mp4')


def package_to_hub(
    model,
    model_name: str,
    model_architecture: str,
    env_id: str,
    eval_env,
    repo_id: str,
    commit_message: str,
    n_eval_episodes=200,
    token=None,
):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the hub

    :param model: trained model
    :param model_name: name of the model zip file
    :param model_architecture: name of the architecture of your model
        (DQN, PPO, A2C, SAC...)
    :param env_id: name of the environment
    :param eval_env: environment used to evaluate the agent
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param commit_message: commit message
    :param n_eval_episodes: number of evaluation episodes (by default: 200)
    :param token: authentication token (See https://huggingface.co/settings/token)
        Caution: your token must remain secret. (See https://huggingface.co/docs/hub/security-tokens)
    """

    print(
        "This function will save, evaluate, generate a video of your agent, "
        + "create a model card and push everything to the hub."
    )

    repo_url = HfApi().create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        exist_ok=True,
    )

    # Step 0: create directory
    directory = Path("./repos/" + repo_id)
    directory.mkdir(parents=True, exist_ok=True)

    # Step 1: save model
    print("Saving model...")
    model.save(directory / model_name)

    # Step 2: Create a config file
    print("Generate config...")
    _generate_config(model_name, directory)

    # Step 3: Evaluate the agent
    print("Evaluating agent...")
    mean_reward, std_reward = eval_agent(env_id, model, n_eval_episodes)

    # Step 4: Generate a video
    print("Generating video...")
    generate_replay(eval_env, model, directory)

    # Step 5: Generate the model card
    generated_model_card, metadata = generate_model_card(
        model_architecture, env_id, mean_reward, std_reward
    )
    _save_model_card(directory, generated_model_card, metadata)

    print(f"Pushing repo {repo_id} to the Hugging Face Hub")

    repo_url = upload_folder(
       repo_id=repo_id,
       folder_path=directory,
       path_in_repo="",
       commit_message=commit_message,
       token=token,
    )

    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


# Parse Args, username and token
parser = ArgumentParser()

parser.add_argument("-e", "--env", dest="env_id", help="select the environment")
parser.add_argument(
    "-u", "--username", dest="username", help="push to the huggingface user's repo"
)
parser.add_argument(
    "-t", "--token", dest="token", help="token to authenticate to hugging face repos"
)

if __name__ == "__main__":
    args = parser.parse_args()

    env_id = args.env_id

    model_name = "./models/ppo-" + env_id + "/best_model"
    model_architecture = "PPO"
    model = PPO.load(model_name)

    repo_id = args.username + "/" + "ppo" + "-" + env_id
    commit_message = "Upload PPO " + env_id + " trained agent."

    print("Push to repo:", repo_id)

    # Create the evaluation env and set the render_mode="rgb_array"
    eval_env = gym.make(env_id, render_mode="rgb_array")

    package_to_hub(
        model=model,  # Our trained model
        model_name=model_name,  # The name of our trained model
        model_architecture=model_architecture,  # The model architecture we used: in our case PPO
        env_id=env_id,  # Name of the environment
        eval_env=eval_env,  # Evaluation Environment
        repo_id=repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name}
        commit_message=commit_message,
        token=args.token,
    )
