# Policy Gradient With Pytorch

Since the beginning of the course, we have only studied value-based methods,  
where we estimate a value function as an intermediate step towards finding an optimal policy.

The link between **Value** and **Policy**:  
$\pi_*(s) = argmax Q_*(s,a)$

In value-based methods, the policy (π) only exists because of the action value estimates since  
the policy is just a function (for instance, greedy-policy) that will select the action with the highest value given a state.

With policy-based methods, we want to optimize the policy directly without having an intermediate step of learning a value function.

So today, we’ll learn about policy-based methods and study a subset of these methods called policy gradient.  
Then we’ll implement our first policy gradient algorithm called **Monte Carlo Reinforce** from scratch using PyTorch.  
Then, we’ll test its robustness using the CartPole-v1 and PixelCopter environments.

## What are the policy-based methods?

The main goal of Reinforcement learning is to find the optimal policyπ∗π∗ that will maximize the expected cumulative reward. Because Reinforcement Learning is based on the reward hypothesis: all goals can be described as the maximization of the expected cumulative reward.

In the first unit, we saw two methods to find (or, most of the time, approximate) this optimal policyπ∗π∗.

    In value-based methods, we learn a value function.
        The idea is that an optimal value function leads to an optimal policyπ∗π∗.
        Our objective is to minimize the loss between the predicted and target value to approximate the true action-value function.
        We have a policy, but it’s implicit since it is generated directly from the value function. For instance, in Q-Learning, we used an (epsilon-)greedy policy.

    On the other hand, in policy-based methods, we directly learn to approximateπ∗π∗ without having to learn a value function.
        The idea is to parameterize the policy. For instance, using a neural networkπθπθ​, this policy will output a probability distribution over actions (stochastic policy).
- stochastic policy !!!!!!!!!!!!!!!!! TDOD  

Our objective then is to maximize the performance of the parameterized policy using gradient ascent.
To do that, we control the parameterθθ that will affect the distribution of actions over a state.

Consequently, thanks to policy-based methods, we can directly optimize our policyπθπθ​ to output a probability distribution over actionsπθ(a∣s)πθ​(a∣s) that leads to the best cumulative return. To do that, we define an objective functionJ(θ)J(θ), that is, the expected cumulative reward, and we want to find the valueθθ that maximizes this objective function.

# The difference between policy-based and policy-gradient methods

Policy-gradient methods, what we’re going to study in this unit, is a subclass of policy-based methods. In policy-based methods, the optimization is most of the time on-policy since for each update, we only use data (trajectories) collected by our most recent version ofπθπθ​.

The difference between these two methods lies on how we optimize the parameterθθ:

    In policy-based methods, we search directly for the optimal policy. We can optimize the parameterθθ indirectly by maximizing the local approximation of the objective function with techniques like hill climbing, simulated annealing, or evolution strategies.
    In policy-gradient methods, because it is a subclass of the policy-based methods, we search directly for the optimal policy. But we optimize the parameterθθ directly by performing the gradient ascent on the performance of the objective functionJ(θ)J(θ).

Before diving more into how policy-gradient methods work (the objective function, policy gradient theorem, gradient ascent, etc.), let’s study the advantages and disadvantages of policy-based methods.

## The advantages and disadvantages of policy-gradient methods