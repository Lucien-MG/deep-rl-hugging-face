import torch
import random

import gymnasium as gym
import numpy as np

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from parser import parse_arg
from ppo import Agent
from hugface import *

def train_ppo(envs, eval_envs, policy, policy_save_path=None):
    score = float("-inf")

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    dones = torch.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs))

    global_step = 0
    next_obs = torch.Tensor(envs.reset()[0])
    next_done = torch.zeros(args.num_envs)
    num_updates = args.total_timesteps // args.batch_size

    print("Total timestaps:", args.total_timesteps)
    print("Number of rollouts:", num_updates)

    for update in range(1, num_updates + 1):
        print("Rollout n°" + str(update) + " / " + str(num_updates))

        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        training_mean_reward = []
        training_length = []

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
        
            next_obs, reward, done, truncated, info = envs.step(action.numpy())
            rewards[step] = torch.tensor(reward).view(-1)
            next_obs, next_done = torch.Tensor(next_obs), torch.Tensor(next_done)

            for item in info:
                if item == "episode":
                    training_mean_reward.append(info["episode"]["r"].max())
                    training_length.append(info["episode"]["l"].max())
                    break
        
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
        
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Evaluate:
        mean_reward, std_reward = evaluate_agent(eval_envs, policy, 10)
        tmp_score = mean_reward - std_reward

        print("Score:", tmp_score)

        if tmp_score > score:
            score = tmp_score
            policy_folder = Path(policy_save_path)
            policy_folder.mkdir(parents=True, exist_ok=True)
            torch.save(agent.state_dict(), policy_folder / 'model_weights.pth')

            print("New best score:", score, "saving model.")
            #video = record_video(eval_env, policy=agent, out_directory="videos/result.gif")
            #writer.add_video("video/mean_reward", video, global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("eval/score", tmp_score, global_step)
        writer.add_scalar("eval/mean_reward", mean_reward, global_step)

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/mean_episodic_return", np.mean(training_mean_reward), global_step)
        writer.add_scalar("charts/mean_episodic_length", np.mean(training_length), global_step)

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        print("")
    
    envs.close()


if __name__ == "__main__":
    args = parse_arg()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    exp_name = f"PPO_{args.env_id}_{args.exp_name}"
    writer = SummaryWriter(f"logs/{exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = gym.make_vec(
            args.env_id,
            num_envs=args.num_envs,
            vectorization_mode="sync",
            wrappers=(
                gym.wrappers.RecordEpisodeStatistics,
                )
            )
    
    eval_envs = gym.make_vec(
            args.env_id,
            num_envs=args.num_envs,
            vectorization_mode="sync",
            wrappers=(
                gym.wrappers.RecordEpisodeStatistics,
                )
            )
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs)

    train_ppo(envs, eval_envs, agent, policy_save_path=f"models/{exp_name}")

    # record_video(eval_env, policy=agent, out_directory="videos/result.gif")
