# Introduction to Deep Reinforcement Learning

Deep RL is a type of Machine Learning where an agent learns how to behave in an environment by performing actions and seeing the results.

## What is Reinforcement Learning?

The idea behind **Reinforcement Learning** is that an agent (an AI) will **learn from the environment by interacting with it** (through trial and error) and **receiving rewards** (negative or positive) as feedback for performing actions.

Learning from interactions with the environment comes from our natural experiences.

## Definition

**Reinforcement learning** is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.

## The Reinforcement Learning Framework

### The Goal of RL

The ultimate goal of RL is to find an optimal policy ($\pi_*$) (the model) that maximizes the expected cumulative reward over time. This is often achieved through iterative learning processes, where the agent explores the environment, learns the consequences of its actions, and gradually refines its policy to achieve better outcomes.

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

The cumulative reward at each time step t can be written as:  
$R(\tau) = r_{t+1} + r_{t+2} + r_{t+3} + r_{t+4} + ...$

Which is equivalent to:  
$R(\tau) = \sum_{k=0}^{\infty} r_{t+k+1}$

However, in reality, we can’t just add them like that. The rewards that come sooner (at the beginning of the game) are more likely to happen since they are more predictable than the long-term future reward.

We define a discount rate called gamma. It must be between 0 and 1. Most of the time between 0.95 and 0.99.

$R(\tau) = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + ...$

Which is equivalent to:  
$R(\tau) = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$

### Type of tasks

A task is an instance of a Reinforcement Learning problem. We can have two types of tasks:
- **episodic**: In this case, we have a starting point and an ending point (a terminal state). This creates an episode: a list of States, Actions, Rewards, and new States.
- **continuing**: These are tasks that continue forever (no terminal state). In this case, the agent must learn how to choose the best actions and simultaneously interact with the environment.

### The Exploration/Exploitation trade-off

Finally, before looking at the different methods to solve Reinforcement Learning problems, we must cover one more very important topic: the exploration/exploitation trade-off.

- Exploration is exploring the environment by trying random actions in order to find more information about the environment.
- Exploitation is exploiting known information to maximize the reward.

Remember, the goal of our RL agent is to maximize the expected cumulative reward. However, we can fall into a common trap.

### Two main approaches for solving RL problems

#### The Policy π: the agent’s brain

The Policy π is the brain of our Agent, it’s the function that tells us what action to take given the state we are in. So it defines the agent’s behavior at a given time.

This Policy is the function we want to learn, our goal is to find the optimal policy π*, the policy that maximizes expected return when the agent acts according to it. We find this π* through training.

There are two approaches to train our agent to find this optimal policy π*:

- Directly, by teaching the agent to learn which action to take, given the current state:  
**Policy-Based Methods**.
- Indirectly, teach the agent to learn which state is more valuable and then take the action that leads to the more valuable states:  
**Value-Based Methods**.

### Policy-Based Methods

In Policy-Based methods, we learn a policy function directly.

We have two types of policies:

- Deterministic: a policy at a given state will always return the same action.  
$a = \pi(s)$
- Stochastic: outputs a probability distribution over actions.  
$\pi(a|s) = P[A|s]$

### Value-based methods

In value-based methods, instead of learning a policy function, we learn a value function that maps a state to the expected value of being at that state.

The value of a state is the expected discounted return the agent can get if it starts in that state, and then acts according to our policy.

“Act according to our policy” just means that our policy is “going to the state with the highest value”.

$v_{\pi}(s) = E_{\pi}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + ... | S_t = s]$

### The “Deep” in Reinforcement Learning

Deep Reinforcement Learning introduces deep neural networks to solve Reinforcement Learning problems — hence the name “deep”.

For instance, in the next unit, we’ll learn about two value-based algorithms: Q-Learning (classic Reinforcement Learning) and then Deep Q-Learning.

You’ll see the difference is that, in the first approach, we use a traditional algorithm to create a Q table that helps us find what action to take for each state.

In the second approach, we will use a Neural Network (to approximate the Q value).

## Summary

That was a lot of information! Let’s summarize:

Reinforcement Learning is a computational approach of learning from actions. We build an agent that learns from the environment by interacting with it through trial and error and receiving rewards (negative or positive) as feedback.

The goal of any RL agent is to maximize its expected cumulative reward (also called expected return) because RL is based on the reward hypothesis, which is that all goals can be described as the maximization of the expected cumulative reward.

The RL process is a loop that outputs a sequence of state, action, reward and next state.

To calculate the expected cumulative reward (expected return), we discount the rewards: the rewards that come sooner (at the beginning of the game) are more probable to happen since they are more predictable than the long term future reward.

To solve an RL problem, you want to find an optimal policy. The policy is the “brain” of your agent, which will tell us what action to take given a state. The optimal policy is the one which gives you the actions that maximize the expected return.

There are two ways to find your optimal policy:
By training your policy directly: policy-based methods.
By training a value function that tells us the expected return the agent will get at each state and use this function to define our policy: value-based methods.

Finally, we speak about Deep RL because we introduce deep neural networks to estimate the action to take (policy-based) or to estimate the value of a state (value-based) hence the name “deep”.