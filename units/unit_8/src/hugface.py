import os
import torch
import imageio
import numpy as np

def record_video(env, policy, out_directory, fps=30):
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
    # imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

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

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

