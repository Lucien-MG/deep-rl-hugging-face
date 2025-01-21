# Unit 7: An Introduction to Unity ML-Agents

In this unit, we’ll train two agent, one agent that play SnowballTarget and
another one that will play Pyramids.

## Train:

To train the agent run:

SoccerTwos:
```bash
mlagents-learn ./SoccerTwos.yaml --env=./envs/SoccerTwos.x86_64 --run-id="SoccerTwos" --no-graphics
```

This training can take quite some time (4-8 hours) to resume the training use:
```bash
mlagents-learn ./SoccerTwos.yaml --env=./envs/SoccerTwos.x86_64 --run-id="SoccerTwos" --no-graphics --resume
```

## Push trained agent to hugging face:

To push your model, first login to hugging face:

```bash
huggingface-cli login
```

then push with the command:

SoccerTwos:
```bash
mlagents-push-to-hf --run-id="SoccerTwos" --local-dir="./results/SoccerTwos" --repo-id="username/poca-SoccerTwos" --commit-message="Your message."
```

## Dependencies

First execute the dependecies.sh script to download everything we need:
```bash
sh dependencies.sh
```

then install ml-agents:
```bash
pip install -e ml-agents/ml-agents-envs
pip install -e ml-agents/ml-agents
```
