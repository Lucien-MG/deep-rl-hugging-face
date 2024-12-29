# Unit 1: Train your first Deep Reinforcement Learning Agent

**Dependencies:**

```bash
pip install torch gymnasium[box2d] gymnasium[other] stable_baselines3 tensorboard huggingface-hub huggingface-sb3
```

**Project Structure:**

* **train_lunar_lander.py:** Trains a PPO agent on the Lunar Lander environment.
* **test_lunar_lander.py:** Evaluates the trained agent and logs the performance.
* **watch_lunar_lander.py:** Visualizes the trained agent interacting with the environment.
* **push_to_hf.py:** Pushes the trained model to the Hugging Face Hub.
