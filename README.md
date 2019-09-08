# Learning to play Atari using Q-Learning
This project is aimed at helping a computer to learn how to play Atari games using Q-Learning Algorithm. The data used is screenshots of the Atari game which is obtained using the `gym` library. 
The trained model was successful in achieving the highest reward of 21 points in session. 

### Dependencies

* Python 3.0 or higher
* Gym [Atari]
* PyTorch

### Training the model

1. First, clone the repository and install all the dependencies
2. To start training, run the following command:
```
python run_dqn_pong.py
```
3. To evaluate the model, run:
``` 
python run_dqn_eval.py
```
