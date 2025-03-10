import os
import torch
import imageio
import numpy as np

import json
from pathlib import Path
import datetime

import gymnasium as gym
from parser import parse_arg
from ppo import Agent

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save

def package(
    model,
    hyperparameters,
    eval_envs,
    n_eval_episodes=10,
    video_fps=15,
    token=None,
):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the hub
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param model: trained model
    :param eval_env: environment used to evaluate the agent
    :param fps: number of fps for rendering the video
    :param commit_message: commit message
    :param logs: directory on local machine of tensorboard logs you'd like to upload
    """
    repos_dir = Path("repos/")
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Save the model
    torch.save(model.state_dict(), repos_dir / "model.pt")

    print("Evaluate agent.")
    # Step 3: Evaluate the model and build JSON
    mean_reward, std_reward = evaluate_agent(eval_envs, model, n_eval_episodes)

    # First get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
        "env_id": hyperparameters.env_id,
        "mean_reward": str(mean_reward),
        "std_reward": str(std_reward),
        "n_evaluation_episodes": n_eval_episodes,
        "eval_datetime": eval_form_datetime,
    }

    print("Write results.")
    # Write a JSON file
    with open(repos_dir / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 4: Generate a video
    video_path = repos_dir / "replay.mp4"
    # vidéo = record_video(eval_envs, model, video_path, video_fps)
    # imageio.mimsave(video_path, vidéo, fps=fps)

    # Step 5: Generate the model card
    generated_model_card, metadata = generate_model_card(
        "PPO", hyperparameters.env_id, mean_reward, std_reward, hyperparameters
    )
    save_model_card(repos_dir, generated_model_card, metadata)

    print("Repo created.")

    # Clone or create the repo
    repo_url = HfApi().create_repo(
        repo_id="/my-ppo-LunarLander-v2",
        token=token,
        private=False,
        exist_ok=True,
    )

    repo_url = upload_folder(
        repo_id="/my-ppo-LunarLander-v2",
        folder_path=repos_dir,
        path_in_repo="",
        commit_message="Push agent to the Hub",
        token=token,
    )

    return repo_url

def upload_to_hub(repo_id, repos_dir, token, commit_message="Push agent to the Hub",):
    # Clone or create the repo
    repo_url = HfApi().create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        exist_ok=True,
    )

    repo_url = upload_folder(
        repo_id=repo_id,
        folder_path=repos_dir,
        path_in_repo="",
        commit_message=commit_message,
        token=token,
    )
    msg.info(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")
    return repo_url

def generate_model_card(model_name, env_id, mean_reward, std_reward, hyperparameters):
    """
    Generate the model card for the Hub
    :param model_name: name of the model
    :env_id: name of the environment
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    :hyperparameters: training arguments
    """
    # Step 1: Select the tags
    metadata = generate_metadata(model_name, env_id, mean_reward, std_reward)

    # Transform the hyperparams namespace to string
    converted_dict = vars(hyperparameters)
    converted_str = str(converted_dict)
    converted_str = converted_str.split(", ")
    converted_str = "\n".join(converted_str)

    # Step 2: Generate the model card
    model_card = f"""
  # PPO Agent Playing {env_id}

  This is a trained model of a PPO agent playing {env_id}.

  # Hyperparameters
  """
    return model_card, metadata

def generate_metadata(model_name, env_id, mean_reward, std_reward):
    """
    Define the tags for the model card
    :param model_name: name of the model
    :param env_id: name of the environment
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    """
    metadata = {}
    metadata["tags"] = [
        env_id,
        "ppo",
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "custom-implementation",
        "deep-rl-course",
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


def save_model_card(local_path, generated_model_card, metadata):
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


def record_video(env, policy):
    print("Recording video at:", out_directory)
    images = []
    done, truncated = False, False
    state, info = env.reset()
    img = env.render()
    img = np.moveaxis(img, -1, 0)
    images.append(img)
    while not done and not truncated:
        state = torch.from_numpy(state).unsqueeze(0)
        # Take the action (index) that have the maximum expected future reward given that state
        action, _, _, _ = policy.get_action_and_value(state)
        state, reward, done, truncated, info = env.step(
            action.cpu().numpy()[0]
        )  # We directly put next_state = state for recording logic
        img = env.render()
        img = np.moveaxis(img, -1, 0)
        images.append(img)

    video = np.array([[np.array(img) for i, img in enumerate(images)]])
    return video

def evaluate_agent(eval_envs, policy, n_eval_episodes, seed=42):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param policy: The agent
    :param n_eval_episodes: Number of episode to evaluate the agent
    """
    episode_rewards = []
    next_obs, info = eval_envs.reset(seed=seed)

    while len(episode_rewards) < n_eval_episodes:
        with torch.no_grad():
            next_obs = torch.from_numpy(next_obs)
            action, logprob, _, value = policy.get_action_and_value(next_obs)
        next_obs, _, _, _, info = eval_envs.step(action.numpy())
        
        for item in info:
            if item == "episode":
                episode_rewards.append(info["episode"]["r"].max())
            if item == "final_info":
                element = None
                for e in info["final_info"]:
                    if e != None:
                        element = e
                        break
                episode_rewards.append(element["episode"]["r"].max())

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

if __name__ == "__main__":
    args = parse_arg()

    eval_envs = gym.make_vec(
            args.env_id,
            num_envs=args.num_envs,
            vectorization_mode="sync",
            wrappers=(
                gym.wrappers.RecordEpisodeStatistics,
                )
            )

    agent = Agent(eval_envs)

    agent.load_state_dict(torch.load("models/PPO_LunarLander-v3_train_3/model_weights.pth", weights_only=True))

    package(
        agent,
        args,
        eval_envs,
        token=args.token
    )
