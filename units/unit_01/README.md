# Unit 1: Train your first Deep Reinforcement Learning Agent

## Project Structure

* **hugface.py:** Pushes the trained model to the Hugging Face Hub.
* **test_model.py:** Evaluates the trained agent and logs the performance.
* **train.py:** Trains a PPO agent on the Lunar Lander environment.
* **watch_model.py:** Visualizes the trained agent interacting with the environment.

Two folders will be created at training time,
* **logs:** which will contain tensorbord training logs
* **models:** which will contain trained models

## Training

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

You can easily watch your agent perform thanks to this command:
```bash
python src/watch_model.py -e "LunarLander-v3"
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
