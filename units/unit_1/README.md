# Unit 1: Train your first Deep Reinforcement Learning Agent

## Project Structure

Scripts:
* **hugface.py:** Pushes the trained model to the Hugging Face Hub.
* **test_model.py:** Evaluates the trained agent and logs the performance.
* **train.py:** Trains a PPO agent on the Lunar Lander environment.
* **watch_model.py:** Visualizes the trained agent interacting with the environment.

These folders will be created at training time,
* **logs:** which will contain tensorbord training logs
* **models:** which will contain trained models
* **repos:** which will contain the generated hugging face repo

## Training

All the training is done in local on my laptop, I tweaked hyperparameters to accelerate the training.
You can train the agent for 800_000 steps it will reach the required score. It takes about 25 minutes on a modern cpu.

## How to use it ?

You can run the training of the ppo with this command:

```bash
python src/train.py -e "LunarLander-v3"
```

Edit the src/train.py file to try different parameters. 

At any moment you can evaluate your agent with:
You can easily watch your agent perform thanks to this command:
```bash
python src/test_model.py -e "LunarLander-v3"
```

You can easily watch your agent in local perform (while training or not) thanks to this command:
```bash
python src/watch_model.py -e "LunarLander-v3" -f ./models/ppo-LunarLander-v3/best_model.zip
```

Or watch your agent pushed to hugging face perform with:
```bash
python src/watch_model.py -e "LunarLander-v3" -r {username}/ppo-LunarLander-v3 -f ./models/ppo-LunarLander-v3/best_model.zip
```

## Push your models

You can push your model with:
```bash
python src/hugface.py -e "LunarLander-v3" -u hugging_face_username -t your_token
```

## Dependencies

```bash
pip install torch gymnasium[box2d] gymnasium[other] stable_baselines3 tensorboard huggingface-hub huggingface-sb3
```

## Tensorboard:

To monitor trainning progress:

```bash
tensorboard --logdir ./logs
```
