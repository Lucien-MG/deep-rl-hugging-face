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
    images.append(img)
    while not done and not truncated:
        state = torch.from_numpy(state).unsqueeze(0)
        # Take the action (index) that have the maximum expected future reward given that state
        action, _, _, _ = policy.get_action_and_value(state)
        state, reward, done, truncated, info = env.step(
            action.cpu().numpy()[0]
        )  # We directly put next_state = state for recording logic
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

def evaluate_agent(env, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The agent
    """
    print("Evaluating agent.")

    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, info = env.reset()
        step = 0
        done, truncated = False, False
        total_rewards_ep = 0

        while done is False and truncated is False:
            state = torch.from_numpy(state).unsqueeze(0)
            action, _, _, _ = policy.get_action_and_value(state)
            new_state, reward, done, truncated, info = env.step(action.cpu().numpy()[0])
            total_rewards_ep += reward
            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

