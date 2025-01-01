# Bonus Unit 1: Introduction to Deep Reinforcement Learning with Huggy

## Train:

To train the agent run:

```bash
mlagents-learn ./config.yaml --env=./envs/Huggy/Huggy.x86_64 --run-id="Huggy" --no-graphics
```

## Push trained agent to hugging face:

To push your model, first login to hugging face:

```bash
huggingface-cli login
```

then push with the command:

```bash
mlagents-push-to-hf
    --run-id="Huggy"
    --local-dir="./results/Huggy"
    --repo-id="Username/ppo-Huggy"
    --commit-message="Your Message"
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
