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
        0	Total Predicted Cases                                0           Inf
        1	Predicted Death Rate                                 0           100
        2	Predicted Recovery Rate                              0           100
        3	Population                                           0           Inf
        4	Susceptible Population                               0           Inf
        
    Actions:
    Type: Box (s+1)
    List of length (s+1)
    
    """
        self.states = s #no of independent simulations to be run 
        low = np.zeros((5,5))
        high = np.array([np.inf, 1, 1, np.inf, np.inf]*5).reshape((5,5))
        self.observation_space = spaces.Box(low, high, shape=(5, 5), dtype = np.float)
        #actions are vectors of the form [n1, n2, n3,...nk] for k states 
        self.action_space = spaces.Box(low = np.zeros((s, ), dtype = int), high = np.array([100]*(s)), shape = (s, ), dtype = np.float)
        
        self.curr_step = 0
        self.done = False
        self.valueMap = np.zeros((self.states, 100))
        self.total = total #total number of vials available in 1 batch = batch size 
        self.episodes = episodes
        self.received = [0]*self.states
        self.states_cond = []
        self.action_list = []
        self.epsilon = 0.4
        self.susc = [0]*self.states
        
            
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
       
        self.states_cond =  np.array([(251754,	1.029973704,	45.49679449,	30885978, 31205576), 
                              (494104,	1.029945113,	46.22893156,	16165867, 16787941),
                              (133576,	1.029376535,	45.96409535,	32819208, 32988134),
                              (4631183,	1.030103971,	43.0974548,	106380541, 112374333), 
                              (27048,	1.027802425,	43.34516415,	1943497, 1978502)])
                               # Total DR RR Susc Population 
                               # Assam, Delhi, Jh, Maha, Naga 
                               # For one date e.g.: Sept 1, 2020
        #store the actions in an array 
        self.action_list = np.array([100/(self.states)]*(self.states))

        return self.states_cond
        

    def step(self, action):
        """
        Actions taken based on the assumptions specified in the paper.
        
    """
         
        # check if we're done
        if self.curr_step >= self.episodes - 1:
            self.done = True
        print("Are we done?", self.done)
            
        if self.states_cond is None:
            raise Exception("You need to reset() the environment before calling step()!")
        else:
            print('Observation Space for this episode is: ', self.states_cond)
       
                    
        #start with equal distribution 
        if self.curr_step == 1:
          self.action_list = np.array([100/(self.states)]*(self.states))
        else:
          self.action_list = action
        
        #exploration vs exploitation        
        if random.uniform(0, 1) < self.epsilon:
            for i in range(self.states):
              action[i] = np.random.randint(0, 100/(self.states))
            self.action_list = action
                        
        else:
            self.action_list = action
            
        #update action_list to store only the most recently used action values 
        # self.action_list = action
        print("Distribution set: ",self.action_list)
        
        #no of units distrbuted to respective states              
        for i in range(self.states):
            self.received[i] = self.total*self.action_list[i]/100
              
        
        # change in condition of states 
        for i in range(self.states):
            self.susc[i] = self.states_cond[i, 3]-self.get_discrete_int(self.received[i])  #new count of susc people
        print("New Count of Susceptible people: ", self.susc)
        self.states_cond = np.array(self.states_cond)
        self.states_cond[:, 3] = self.susc                            #update values in states_cond matrix 
        
                  
        #reward 
        reward = self.get_reward()
        

        # increment episode
        self.curr_step += 1


        return self.states_cond, reward, self.done, {'action_list': self.action_list, 'episode_number': self.curr_step}
    
    def get_reward(self):
      reward = [0]*self.states              
      for i in range(self.states):          
        reward[i] = self.states_cond[i, 3]*math.exp(-self.action_list[i])
      print("Reward distribution: ", reward)
      reward = sum(reward)
      print("Reward: ", reward)
      return reward 

    
    #def render(self, mode='human', close= False)
      
   
    def close(self):
        pass 
