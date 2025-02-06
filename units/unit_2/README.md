
# Unit 2: Introduction to Q-Learning

## Project Structure

Scripts:
* **hugface.py:** Hugging face tools.
* **push.py:** Allow to push model to hugging face repository.
* **qlearning.py:** Q-learning implementation. 
* **seeds.py:** Seeds given by hugging face to train the model.
* **train.py:** Train the model.
* **watch_model.py:** Allow to watch model in live.

These folders will be created at training time,
* **models:** which will contain trained models
* **repos:** which will contain the generated hugging face repo

## How to use it ?

### Train the agent

You can run the training of the ppo with this command:

```bash
python src/train.py -e "Taxi-v3"
```
Edit the src/train.py file to try different parameters. 

### Watch it perform

You can easily watch your agent in local perform thanks to this command:
```bash
python src/watch.py -e "Taxi-v3" -m models/q-learning-Taxi-v3.pkl
```

## Push your models

You can push your model with:
```bash
python src/push_to_hf.py -e "Taxi-v3" -u hugging_face_username -t your_token
```

## Dependencies

```bash
pip install numpy gymnasium huggingface-hub
```
