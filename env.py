import numpy as np 
import gym 
from gym import spaces 
import matplotlib.pyplot as plt 
import math
import random
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
    
    
    def __init__(self, s, episodes, total):
        """ 
        Observation:
        Type: Box(5)
                                                                Min         Max
        0	Confirmed Cases                                      0           Inf
        1	Active cases                                         0           Inf
        2	Recovery Rate = 100 - Death Rate=Recovered/Confirmed 0           Inf
        3	Population Density                                   0           Inf
        4	Projected Cases (to be introduced later by ABM)      0           Inf
        
    Actions:
    Type: Box (s+1)
    List of length (s+1)
    
    """
        self.states = s #no of independent simulations to be run 
        low = np.zeros(4)
        high = np.ones(4) * np.inf
        self.observation_space = spaces.Box(low, high, dtype = np.float32)
        #actions are vectors of the form [n1, n2, n3,...nk, r] for k states and r reserved amount of drug 
        self.action_space = spaces.Box(low = np.zeros((s+1, ), dtype = int), high = np.array([100]*(s+1)), shape = (s + 1, ), dtype = np.float)
        
        self.curr_step = 0
        self.done = False
        self.valueMap = np.zeros((self.states, 100))
        self.total = total #total number of vials available 
        self.episodes = episodes
        self.received = [0]*self.states
        self.rr = [0]*self.states
        self.states_cond = []
        self.action_list = []
        self.gamma = 0.90
        self.epsilon = 0.2
        self.set_episode_length(2)
    
    def get_discrete_int(self, n):
        discrete_int = int(n)
        return discrete_int

    def reset(self):
        """
        Resets observation_space to a matrix initialising situation of states wrt the current figures; 
        action_space tp start exploring from the point of equal distribution between all states.
        """
        self.curr_step = 0
        self.done = False
        self.total = 10000
        # Declare the Initial Conditions for the States
       
        self.states_cond =  np.array([(80188,	28329, 0., 11297 ),  
                              (30709,	6511,	0., 308),
                              (16944,	3186,	0., 201),
                              (12965,	2444,	0., 236),
                              (159133,	67615,	0., 365)])
                            #   (78335,	33216,	0., 555)])
                                # Confirmed   Active  Recovery Rate(due to effect of drug) Population Density
                                # Delhi, Guj, Raja, MP, Maha, TN 
        #store the actions in an array 
        self.action_list = np.array([100/(self.states+1)]*(self.states+1))
        #Declare the Value table 
        self.valueMap = np.zeros((self.states, 100))
        return (self.states_cond, self.action_list)
        

    def set_episode_length(self, minute_interval):
        """
        :param minute_interval: how often we will record information, make a recommendation
        """
        self.minute_interval = minute_interval
        global ns
        ns = int((24 * 60) / self.minute_interval) + 1
        # Final Time (hr)
        tf = 24  # simulate for 24 hours
        # Time Interval (min)
        self.t = np.linspace(0, tf, ns)
        self.episode_length = len(self.t)

    def step(self, action):
        """
        Assumptions:
        1. Drug has 50% efficacy 
        2. Vaccine is passive, not antigen based- works to fight off existing infection.
        3. 1 person requires 1 vial (dose) only.
        4. No of confirmed and active cases in one particular region is constant, until we integrate the ABM projections model. 
        So, for the time being, recovery rate will always increase when drug is supplied to a particular state.
        """

        if self.states_cond is None:
            raise Exception("You need to reset() the environment before calling step()!")
        
        # check if we're done
        if self.curr_step >= self.episode_length - 2:
            self.done = True
        
        # get updated time step
        ts = [self.t[self.curr_step], self.t[self.curr_step + 1]]

        # #exploration vs exploitation 
        # while self.done ==False:
        #     if random.uniform(0, 1) < self.epsilon:
        #         for i in range(self.states):
        #             action[i] = np.random.randint(0, 16.66)
        #     else:
        #         for i in range(self.states):
        #             action[i] = np.argmax(self.valueMap[i])+1                            

        #update action_list to store only the most recently used action values 
        self.action_list = action

        #no of units distrbuted to respective states 
        # received = []
        reserved = []
        for i in range(self.states):
            self.received[i] = self.total*action[i]/100
        reserved = reserved.append(self.total*action[self.states])
        
        
        #simulation
        recovered = [0]*self.states
        
        for i in range(self.states):
            recovered[i] = 0.5*int(self.received[i])  #50% efficacy
            self.rr[i] = recovered[i]/self.states_cond[i, 0] 
            print(self.rr)
            print(self.states_cond[:,2])       #recovery rate for the 'i'th state                         
            self.states_cond[:, 2] = self.rr                            #update values in states_cond matrix 

          
        #reward only when task done 
        reward = self.get_reward()

        #policy evaluation
        deltas = []
        for it in range(ns):
            copyValueMap = np.copy(self.valueMap)
            deltaState = [0]*self.states
            for state in range(self.states):
                value = np.zeros((self.states, ))
                value += reward[state]+(self.gamma*self.valueMap[state, self.get_discrete_int(self.action_list[state])])
                deltaState = np.append(deltaState, np.abs(copyValueMap[state, self.get_discrete_int(self.action_list[state])]-value[state]))
                copyValueMap[state, self.get_discrete_int(self.action_list[state])]= value[state]
            deltas.append(deltaState)
            valueMap = copyValueMap
            if it%250 == 0:
                print("Iteration {}".format(it+1))
                print(valueMap)  #print position also 
                print("")

        plt.figure(figsize=(20, 10))
        plt.plot(deltas)
       
        

        # increment episode
        self.curr_step += 1


        return self.states_cond, reward, self.done, {}
    
    def get_reward(self):
        for i in range(self.states):
            cpu = 1
            reward = [0]*self.states
            reward[i] = self.rr[i]*math.exp(-cpu*self.received[i])
        return reward 


   
    def close(self):
        pass 

env = StatesEnv(5, 500, 10000)

obs = env.reset()
episodes = 500
for step in range(episodes):
    print("Episode {}".format(step+1))
    obs, reward, done, info = env.step([16.66, 16.66, 16.66, 16.66, 16.66, 16.66])
    print("obs=", obs, "reward=", reward, "done=", done)
    if done: 
        print("Done:)")
        break

env.close()


