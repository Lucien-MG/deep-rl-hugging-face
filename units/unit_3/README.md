# Unit 3: Deep Q-Learning with Atari Games

Here, we are taking unit 1 and unit 3, using optuna to optimize the parameters

## Training

You can run the training of the ppo with this command:

```bash
python -m rl_zoo3.train --algo dqn  --env SpaceInvadersNoFrameskip-v4 -f logs/ --tensorboard-log logs/ -c dqn.yml
```

Edit the dqn.yml config. 

## Tensorboard:

To monitor trainning progress:

```bash
tensorboard --logdir ./logs
```
