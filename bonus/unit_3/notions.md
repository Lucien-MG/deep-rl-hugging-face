# Advanced Topics in Reinforcement Learning

## Model Based Reinforcement Learning (MBRL)

Model-based reinforcement learning only differs from its model-free counterpart in learning a dynamics model,  
but that has substantial downstream effects on how the decisions are made.

The dynamics model usually models the environment transition dynamics, $s_t+1=f_θ(st,at)$,  
but things like inverse dynamics models (mapping from states to actions) or reward models (predicting rewards) can be used in this framework.

### Simple Definition

- There is an agent that repeatedly tries to solve a problem, accumulating state and action data.
- With that data, the agent creates a structured learning tool, a dynamics model, to reason about the world.
- With the dynamics model, the agent decides how to act by predicting the future.
- With those actions, the agent collects more data, improves said model, and hopefully improves future actions.

### Academic Definition

Model-based reinforcement learning (MBRL) follows the framework of an agent interacting in an environment,  
learning a model of said environment, and then **leveraging the model for control** (making decisions).

Specifically, the agent acts in a Markov Decision Process (MDP) governed by a transition functionst+1=f(st,at)st+1​=f(st​,at​) and  
returns a reward at each stepr(st,at)r(st​,at​). With a collected datasetD:=si,ai,si+1,riD:=si​,ai​,si+1​,ri​,  
the agent learns a model, st+1=fθ(st,at) to minimize the negative log-likelihood of the transitions.

We employ sample-based model-predictive control (MPC) using the learned dynamics model,  
which optimizes the expected reward over a finite, recursively predicted horizon,ττ,  
from a set of actions sampled from a uniform distributionU(a)U(a).