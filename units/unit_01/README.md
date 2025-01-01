# Unit 1: Train your first Deep Reinforcement Learning Agent

## Training

You can run the training of the ppo with this command:

```bash
python src/train.py -e "LunarLander-v3"
```

Edit the src/train.py file to try different parameters. 

## Dependencies

```bash
pip install torch gymnasium[box2d] gymnasium[other] stable_baselines3 tensorboard huggingface-hub huggingface-sb3
```

## Project Structure

* **train.py:** Trains a PPO agent on the Lunar Lander environment.
* **test_lunar_lander.py:** Evaluates the trained agent and logs the performance.
* **watch_lunar_lander.py:** Visualizes the trained agent interacting with the environment.
* **hugface.py:** Pushes the trained model to the Hugging Face Hub.

## Tensorboard:

To monitor trainning progress:

```bash
tensorboard --logdir ./logs
```
