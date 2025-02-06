import pickle
import numpy as np
import gymnasium as gym

from argparse import ArgumentParser

def greedy_policy(q_table, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(q_table[state][:])

    return action

def watch(env_id, qtable):
    env = gym.make(env_id, render_mode="human")
    obs, info = env.reset()

    done = False

    while not done:
        action = greedy_policy(qtable, obs)
        obs, reward, done, truncated, info = env.step(action)
        print(reward)
        env.render()

    env.close()

# Parse Args, username and token
parser = ArgumentParser()

parser.add_argument("-e", "--env", dest="env_id",
                    help="push to the huggingface user's repo")
parser.add_argument("-m", "--model", dest="model",
                    help="load model")

if __name__ == '__main__':
    args = parser.parse_args()

    # Pickle the model
    model_path = "./models/q-learning-" + args.env_id  + ".pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    watch(args.env_id, model)
