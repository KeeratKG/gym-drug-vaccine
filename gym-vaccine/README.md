# Vaccine Distribution Problem 
*The model described here has been used as a sub-model in the entire pipeline used in [VacSIM](https://arxiv.org/abs/2009.06602), a novel reinforcment learning-based
strategy for optimal distribution of vaccines against COVID as soon as they hit the market. The full code will be open sourced soon!*

This folder contains the code for `vd-v0`, the first version of an OpenAI gym environment developed for the Vaccine Distribution scenario. A detailed explanation of the technicalities
can be referred to from our paper on VacSIM.

## Installation 
1. Go to the *gym-vaccine* folder and run ```pip install -e```. This will install the gym environment. 
2. To use the environment, run the following commands:
```import gym
import gym_vaccine
env = gym.make('vd-v0')
```

You can now use the environment to train your model efficiently. 

Feel free to check out and the code and tweak the hyperparameters. Do open PRs/Issuess in case you find something interesting. Happy Hacking :) 

