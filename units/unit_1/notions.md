# Introduction to Deep Reinforcement Learning

Deep RL is a type of Machine Learning where an agent learns how to behave in an environment by performing actions and seeing the results.

## What is Reinforcement Learning?

The idea behind Reinforcement Learning is that an agent (an AI) will learn from the environment by interacting with it (through trial and error) and receiving rewards (negative or positive) as feedback for performing actions.

Learning from interactions with the environment comes from our natural experiences.

## Definition

Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.

## The Reinforcement Learning Framework

### The Goal of RL

The ultimate goal of RL is to find an optimal policy (π*) (the model) that maximizes the expected cumulative reward over time. This is often achieved through iterative learning processes, where the agent explores the environment, learns the consequences of its actions, and gradually refines its policy to achieve better outcomes.

### The RL Process

                   ____________________________________
                  |             action (At)            |
                  |                                    v
          +-----------------+                 +-----------------+
          |      Agent      |                 |   Environment   |
          +-----------------+                 +-----------------+
               ^      ^                            |       |
               |      |____________________________|       |
               |                reward (Rt)                |
               |___________________________________________|
                                state (St)

### The reward hypothesis: the central idea of Reinforcement Learning

Why is the goal of the agent to maximize the expected return?

Because RL is based on the reward hypothesis, which is that all goals can be described as the maximization of the expected return (expected cumulative reward).

That’s why in Reinforcement Learning, to have the best behavior, we aim to learn to take actions that maximize the expected cumulative reward.

### Markov Property

In papers, you’ll see that the RL process is called a Markov Decision Process (MDP).

We’ll talk again about the Markov Property in the following units. But if you need to remember something today about it, it’s this: the Markov Property implies that our agent needs only the current state to decide what action to take and not the history of all the states and actions they took before.

### Observations/States Space

Observations/States are the information our agent gets from the environment. In the case of a video game, it can be a frame (a screenshot). In the case of the trading agent, it can be the value of a certain stock, etc.

There is a differentiation to make between observation and state, however:

* State s: is a complete description of the state of the world (there is no hidden information). In a fully observed environment.

* Observation o: is a partial description of the state. In a partially observed environment.

### Action Space

The Action space is the set of all possible actions in an environment.

The actions can come from a discrete or continuous space:

* Discrete space: the number of possible actions is finite.

* Continuous space: the number of possible actions is infinite.

### Rewards and the discounting

The reward is fundamental in RL because it’s the only feedback for the agent. Thanks to it, our agent knows if the action taken was good or not.

