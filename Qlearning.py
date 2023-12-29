import numpy as np
import random
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
action_mapping = {0: "U", 1: "D", 2: "R", 3: "L"}
plt.style.use("Solarize_Light2")
#create the environment

class VanillaGridEnv:
    def __init__(self):
        self.height = 5
        self.width = 5
        self.grid = np.zeros((self.height,self.width))-1
        self.starting_point = (0,0)
        self.goal_point = (4,3)
        self.water_points = [(0,2),(1,2),(3,2),(3,3),(4,2)]
        self.agent_location = self.starting_point

        self.grid[self.goal_point[0], self.goal_point[1]] = 100 # TODO: can change reward
        for water_point in self.water_points:
            self.grid[water_point[0], water_point[1]] = -90 # TODO: can change reward

        self.actions = ['U', 'D', 'R', 'L']

    def step(self, action):
        prev_location = self.agent_location

        if action == 'U':
            if prev_location[1]==4:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0], self.agent_location[1]+1)
                reward = self.get_reward(self.agent_location)

        elif action == 'D':
            if prev_location[1]==0:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0], self.agent_location[1]-1)
                reward = self.get_reward(self.agent_location)

        elif action == 'R':
            if prev_location[0]==4:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0]+1, self.agent_location[1])
                reward = self.get_reward(self.agent_location)

        elif action == 'L':
            if prev_location[0]==0:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0]-1, self.agent_location[1])
                reward = self.get_reward(self.agent_location)
        return reward
    
    def get_reward(self, location):
        reward = self.grid[location[0], location[1]]
        return reward

    def is_done(self, location):
        return (location[0],location[1]) == (self.goal_point[0], self.goal_point[1]) 

action_mapping = {0: "U", 1: "D", 2: "R", 3: "L"}
reverse_action_mapping = {"U": 0, "D": 1, "R": 2, "L": 3}

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.height * env.width
        self.action_size = len(env.actions)
        # self.QTable = np.ndarray((5,5,4),dtype=np.float32)
        self.QTable = np.random.rand(5,5,4)*4  # TODO: can change low and high values
        #define hyperparameters 
        self.lr = 0.01
        #low gamma->immediate rewards are important high gamma->future rewards
        self.gamma = 0.99
        self.epsilon_start = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.01
    
    def getQTable(self):
        return self.QTable
    
    def action(self,state):
        rand = random.uniform(0,1)
        if rand > self.epsilon:

            action_index = action_mapping[np.argmax(self.QTable[state[0],state[1], :])]
            #print(f"action: {action_index}")
            #return np.argmax(self.QTable[state[0],state[1], :]) #select the action with the highest Q value for that state (returns the index -> coresponds to action)
            return action_index
        else: 
            randint = int(random.uniform(0,4))
            #print(f"action: {self.env.actions[randint]}")
            return self.env.actions[randint]
        
    def updateQvals(self, state, new_state, action, reward):
        #update Q table
        #𝑄𝑠,𝑎 =𝑄𝑠,𝑎 +𝑙∗(𝑟𝑠,𝑎 +𝛾∗max𝑄𝑠′,𝑎′ −𝑄𝑠,𝑎)    
        action_index = reverse_action_mapping.get(action)
        self.QTable[state[0],state[1],action_index] = self.QTable[state[0],state[1],action_index] + self.lr*(reward + self.gamma*np.max(self.QTable[new_state[0],new_state[1], :]) - self.QTable[state[0],state[1],action_index]) 


def print_ndarray(matrix):
    if matrix.shape == (5,5):
        print(np.flip(matrix.T, axis=0))
    else:
        for i in range(0,4):
            if i==0:
                print("U:")
            elif i==1:
                print("D:")
            elif i==2:
                print("R:")
            else:
                print("L:")
            print(np.flip(matrix[:,:,i].T, axis=0), '\n')

env = VanillaGridEnv()
agent = QLearningAgent(env) 

# U-U-U-U-R-R-R-R-D
# R-R-U-U-U-U-R-R-D
first_path = ['U','U','U','U','R','R','R','R','D']
second_path = ['R','R','U','U','U','U','R','R','D']

print_ndarray(agent.getQTable())
print_ndarray(env.grid)

print("First path: ")
first_path_reward = 0
env.agent_location = env.starting_point
for iter in range(len(first_path)):
    state = env.agent_location
    action = first_path[iter]
    reward = env.step(action)
    new_state = env.agent_location
    agent.updateQvals(state,new_state,action,reward)
    first_path_reward+=reward
print(f"     first path total reward: {first_path_reward}")
print("Q-table after first path: ")
print_ndarray(agent.getQTable())

print("Second path: ")

env.agent_location = env.starting_point
second_path_reward = 0
for iter in range(len(second_path)):
    state = env.agent_location
    action = second_path[iter]
    reward = env.step(action)
    new_state = env.agent_location
    agent.updateQvals(state,new_state,action,reward)
    second_path_reward+=reward
    
print(f"     second path total reward: {second_path_reward}")
print("Q-table after second path: ")
print_ndarray(agent.getQTable())



