# Deep Q-learning

## From Q-Learning to Deep Q-Learning

We learned that Q-Learning is an algorithm we use to train our Q-Function, an action-value function that determines the value of being at a particular state and taking a specific action at that state.

The Q comes from “the Quality” of that action at that state.

Internally, our Q-function is encoded by a Q-table, a table where each cell corresponds to a state-action pair value. Think of this Q-table as the memory or cheat sheet of our Q-function.

The problem is that Q-Learning is a tabular method. This becomes a problem if the states and actions spaces are not small enough to be represented efficiently by arrays and tables. In other words: it is not scalable.

The neural network will approximate, given a state, the different Q-values for each possible action at that state. And that’s exactly what Deep Q-Learning does.

## The Deep Q-Network (DQN)

When the Neural Network is initialized, the Q-value estimation is terrible. But during training, our Deep Q-Network agent will associate a situation with the appropriate action and learn to play the game well.

## The Deep Q-Learning Algorithm
