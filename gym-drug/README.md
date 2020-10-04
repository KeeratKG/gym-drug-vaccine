## Drug Distribution Problem 

This folder contains the code for `dd-v0`, the first version of an OpenAI gym environment developed for the Drug Distribution scenario. This environment recommends a policy for fair distribution of a scarce drug in the event of a widespread medical crisis, given certain properties describing the state(social, economical, etc) of the recipients. 
## Installation 
1. Go to the *gym-drug* folder and run ```pip install -e```. This will install the gym environment. 
2. To use the environment, run the following commands:
```import gym
import gym_drug
env = gym.make('dd-v0')
```

You can now use the environment to train your model efficiently. 

Feel free to check out the code and tweak the hyperparameters. Do open PRs/Issuess in case you find something interesting. Happy Hacking :) 


