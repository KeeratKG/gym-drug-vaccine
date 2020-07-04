import numpy as np 
import gym 
from gym import spaces 
import matplotlib.pyplot as plt 
# Stable Baselines only supports tensorflow 1.x for now
#jupyter notebook 
# import sys 
# !{sys.executable} -m pip install tensorflow==1.15.0
#google colab 
# %tensorflow_version 1.x

# !pip install stable-baselines[mpi]==2.10.0

class StatesEnv(gym.Env):
    """
    Customised Environment that follows gym interface.
    Describes relevant properties of the state and action spaces. 
    """
    metadata = {'render.modes':['human']}
    
    
    s = 6 # Delhi, Guj, Raja, MP, Maha, TN

    def __init__(self, s, prop):
        #initialise state properties and their values, obsn space and action space???
        self.STATES = s #no of independent simulations to be run 
        self.PROPERTIES = prop 
        #observation will be the condition of state at a particular time 
        observation_space = spaces.Box(low= np.zeros((6, 5)), high = np.repeat([float('inf')]*5, 6, 0 ), shape = (s, prop), dtype = np.float32)
        #actions are vectors of the form [n1, n2, n3,...nk, r] for k states and r reserved amount of drug 
        action_space = spaces.Box(low = np.zeros((s+1, )), high = np.array([100]*(s+1)), shape = (s + 1, ), dtype = np.uint8) #insert condition that sum of actions=100!!!Declare array maybe?
        sum = 0
        for i in range(s + 1):
            sum += action_space[i]
        assert sum == 100 #returns error if total % is not 100 

    def reset(self):
        """
        Resets observation_space to a matrix initialising situation of states wrt the current figures; 
        action_space to start exploring from the point of equal distribution between all states.
        """
        action_space = [100/(s+1)]*(s+1)
        observation_space =  [[80188,	28329,	2558,	16787941,	0.03190003492],  
                              [30709,	6511,	1789,	60439692,	0.05825653717],
                              [16944,	3186,	391,	68548437,	0.02307601511],
                              [12965,	2444,	550,	72626809,	0.04242190513],
                              [159133,	67615,	7273,	112374333,	0.04570390805],
                              [78335,	33216,	1025,	72147030,	0.01308482798]]
                              # Confirmed   Active  Deaths   Population  P(dying)
                              # Delhi, Guj, Raja, MP, Maha, TN 
    def step(self, action):
        """
        Assumptions:
        1. Drug has 50% efficacy 
        2. Vaccine is passive, not antigen based- works to fight off existing infection.
        3. 1 person requires 1 vial (dose) only.
        """
        P = observation_space.copy()
        # Total number of vials available 
        self.TOTAL = k

        #no of units distrbuted to respective states 
        k_received = []
        for i in range(s):
            k_received[i] = k*action[i]/100
        
        #add column of units distributed per state to update observation space 
        P.append(k_received)

        #measuring the effect of drug on each state 
        # m is the no of ppl moving from active to recovered 

        for i in range(s):
            m[i] = 0.5*k_received[i]              #50% efficacy
            P[i, 1] -= m[i]
            prob_dying[i] = P[i, 2]/P[i, 0]

        #task is done when all states show a decrease in probability of dying 
        done = bool(P[i] < observation_space[i, 4] for i in range(s))
        #reward only when task done 
        reward = 10 if done else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return P, reward, done, info

    def render(self, mode='human'):
        x = EPISODES
        y = []
        for i in range(s):
            y[i] = prob_dying [i]
            y.append(y[i])
            plt.plot(x, y[i])
            plt.xlabel('Number of episodes')
            plt.ylabel('P(dying) of state')
            plt.title('Learning Process')
        
        plt.show()
        
        
    
    def close(self):
        pass 



