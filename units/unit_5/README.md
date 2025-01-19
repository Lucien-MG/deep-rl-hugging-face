# Unit 5: An Introduction to Unity ML-Agents

In this unit, we’ll train two agent, one agent that play SnowballTarget and
another one that will play Pyramids.

## Train:

To train the agent run:

SnowballTarget:
```bash
mlagents-learn ./SnowballTarget.yaml --env=./envs/SnowballTarget/SnowballTarget.x86_64 --run-id="SnowballTarget" --no-graphics
```

Pyramids:
```bash
mlagents-learn ./Pyramids.yaml --env=./envs/Pyramids/Pyramids --run-id="Pyramids" --no-graphics
```

## Push trained agent to hugging face:

To push your model, first login to hugging face:

```bash
huggingface-cli login
```

then push with the command:

SnowballTarget:
```bash
mlagents-push-to-hf --run-id="SnowballTarget" --local-dir="./results/SnowballTarget" --repo-id="username/ppo-SnowballTarget" --commit-message="Your message."
```

Pyramids:
```bash
mlagents-push-to-hf --run-id="Pyramids" --local-dir="./results/Pyramids" --repo-id="username/ppo-Pyramids" --commit-message="Your message."
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
