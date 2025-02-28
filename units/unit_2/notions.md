# Introduction to Q-Learning

In this unit, we are going to dive deeper into one of the Reinforcement Learning methods:  
value-based methods and study our first RL algorithm: **Q-Learning**.

In this unit:
- Learn about value-based methods.
- Learn about the differences between Monte Carlo and Temporal Difference Learning.
- Study and implement our first RL algorithm: Q-Learning.

##  Two types of value-based methods

In value-based methods, we learn a value function that maps a state to the expected value of being at that state.

The value of a state is the expected discounted return the agent can get if it starts at that state and then acts according to our policy.
