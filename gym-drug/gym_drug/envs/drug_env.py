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
        self.total = total #total number of vials available in 1 batch = batch size 
        self.episodes = episodes
        self.received = [0]*self.states
        self.rr = [0]*self.states
        self.states_cond = []
        self.action_list = []
        self.gamma = 0.80
        self.epsilon = 0.4  

        
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
       
        self.states_cond =  np.array([(80188,28329, 0., 11297 ),  
                              (30709,6511,0., 308),
                              (16944,3186,0., 201),
                              (12965,2444,0., 236),
                              (159133,67615,0., 365)])
                               # Confirmed   Active  Recovery Rate(due to effect of drug) Population Density
                               # Delhi, Guj, Raja, MP, Maha 
        #store the actions in an array 
        self.action_list = np.array([100/(self.states+1)]*(self.states+1))
        #Declare the Value table 
        self.valueMap = np.zeros((self.states, 100))
        return self.states_cond, self.action_list
        

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
        if self.curr_step >= self.episodes - 1:
            self.done = True
                    
        #start with equal distribution 
        if self.curr_step == 1:
            self.action_list = np.array([100/(self.states+1)]*(self.states+1))
        
        #exploration vs exploitation        
        if random.uniform(0, 1) < self.epsilon:
            for i in range(self.states):
                action[i] = np.random.randint(0, 100/(self.states+1))
            reserved = 100-sum(action)
            action[self.states] = reserved
        else:
            for i in range(self.states):
                action[i] = np.argmax(self.valueMap[i])+1 
            reserved = 100 - sum(action) 
            action[self.states] = reserved

        #update action_list to store only the most recently used action values 
        self.action_list = action
        print("Distribution set: ",self.action_list)
        

        #no of units distrbuted to respective states 
        # received = []        
        for i in range(self.states):
            self.received[i] = self.total*self.action_list[i]/100
        reserved_qty = self.total*self.action_list[self.states]/100
        print("reserved quantity: ", reserved_qty)
        
        
        #simulation
        recovered = [0]*self.states
        
        for i in range(self.states):
            recovered[i] = 0.5*self.get_discrete_int(self.received[i])  #50% efficacy
            self.rr[i] = recovered[i]/self.states_cond[i][0] 
        print("recovery rate due to drug: ",self.rr)
        self.states_cond = np.array(self.states_cond)
        print("recovery rate(before): ", self.states_cond[:,2])       #recovery rate for the 'i'th state                         
        self.states_cond[:, 2] = self.rr                            #update values in states_cond matrix 
        self.states_cond[:, 1] -= recovered 

          
        #reward only when task done 
        reward = self.get_reward()

        #update the value map
        copyValueMap = np.copy(self.valueMap)
        deltaState = [0]*self.states
        for state in range(self.states):
            value = np.zeros((self.states, ))
            value += reward+(self.gamma*self.valueMap[state, self.get_discrete_int(self.action_list[state])])
            deltaState = np.append(deltaState, np.abs(copyValueMap[state, self.get_discrete_int(self.action_list[state])]-value[state]))
            copyValueMap[state, self.get_discrete_int(self.action_list[state])]= value[state]
        valueMap = copyValueMap
        

        # increment episode
        self.curr_step += 1


        return self.states_cond, reward, self.done, {'action_list': self.action_list, 'episode': self.curr_step,
                                                    'change': deltaState}
    
    def get_reward(self):
        for i in range(self.states):
            cpu = 1
            reward = [0]*self.states
            reward[i] = self.rr[i]*math.exp(-cpu*self.received[i])
        reward = sum(reward)
        return reward 


   
    def close(self):
        pass 

locations = 5
episodes = 50
total_drugs_qty = 10000
#create an instance of the class 
env = StatesEnv(locations, episodes, total_drugs_qty)

action = [16.66, 16.66, 16.66, 16.66, 16.66, 16.66]
delta = []


obs = env.reset()
for ep in range(episodes):
    obs, reward, done, info = env.step(action)
    delta = np.append(delta, info)
    if ep%10 == 0:
        print("Episode {}".format(ep+1))        
        print("obs=", obs, "reward=", reward, "done=", done)        
    
#print(delta)
#     plt.figure(figsize=(20, 10))
#     plt.rcParams.update({'figure.max_open_warning': 0})
#     if done: 
#             print("Done:)")
#             break
# for l in range(locations):    
#     plt.subplot(locations, 1, l+1)
#     plt.plot([5,10,15,20,25,30,35,40,45,50], delta[l], 'b-')
# plt.show()


env.close()



