import numpy as np
import random
import matplotlib.pyplot as plt
import time

np.set_printoptions(precision=2)
plt.style.use("Solarize_Light2")

#create the environment
class JetPackGridEnv:
    def __init__(self):
        self.grid = np.zeros((5,5))-1
        self.starting_point = (0,0)
        self.goal_point = (4,3)
        self.water_points = [(0,2),(1,2),(3,2),(3,3),(4,2)]
        self.jetpack_point = (4,0)
        self.height = 5
        self.width = 5
        self.agent_location = self.starting_point
        self.have_jetpack = False
        self.first_time_jp = True

        self.grid[self.goal_point[0], self.goal_point[1]] = 100 # TODO: can change reward
        for water_point in self.water_points:
            self.grid[water_point[0], water_point[1]] = -90 # TODO: can change reward
        self.grid[self.jetpack_point[0], self.jetpack_point[1]] = -1 # TODO: can change reward
        self.actions = ['U', 'D', 'R', 'L']

    def step(self, action):
        prev_location = self.agent_location
        # if(env.agent_location == env.jetpack_point):
 
        if action == 'U':
            if prev_location[1]==4:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0], self.agent_location[1]+1)
                reward = self.get_reward(self.agent_location)
        elif action == 'UU':
            if prev_location[1]==4 or prev_location[1]==3:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0], self.agent_location[1]+2)
                reward = self.get_reward(self.agent_location)
        elif action == 'D':
            if prev_location[1]==0:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0], self.agent_location[1]-1)
                reward = self.get_reward(self.agent_location)
        elif action == 'DD':
            if prev_location[1]==0 or prev_location[1]==1:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0], self.agent_location[1]-2)
                reward = self.get_reward(self.agent_location)
        elif action == 'R':
            if prev_location[0]==4:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0]+1, self.agent_location[1])
                reward = self.get_reward(self.agent_location)
        elif action == 'RR':
            if prev_location[0]==4 or prev_location[0]==3:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0]+2, self.agent_location[1])
                reward = self.get_reward(self.agent_location)
        elif action == 'L':
            if prev_location[0]==0:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0]-1, self.agent_location[1])
                reward = self.get_reward(self.agent_location)
        elif action == 'LL':
            if prev_location[0]==0 or prev_location[0]==1:
                reward = self.get_reward(prev_location) - 1
            else:
                self.agent_location = (self.agent_location[0]-2, self.agent_location[1])
                reward = self.get_reward(self.agent_location)

    
        #if agent is in jetpack location, it takes the jetpack, and isnt effected by water points anymore
        if self.agent_location == self.jetpack_point and self.first_time_jp:
            self.have_jetpack = True
            self.first_time_jp = False
            self.actions = ['U', 'D', 'R', 'L', 'UU', 'DD', 'RR', 'LL']
            for water_point in self.water_points:
                self.grid[water_point[0], water_point[1]] = -1 
            #print("*************JETPACK OBTAINED*************")
            reward += 50
        return reward
    
    def get_reward(self, location):
        #print(location[0], location[1])
        reward = self.grid[location[0], location[1]]
        return reward

    def is_done(self, location):
        return (location[0],location[1]) == (self.goal_point[0], self.goal_point[1]) 

action_mapping = {0: "U", 1: "D", 2: "R", 3: "L", 4: "UU", 5: "DD", 6: "RR", 7: "LL"}
reverse_action_mapping = {"U": 0, "D": 1, "R": 2, "L": 3, "UU": 4, "DD": 5, "RR": 6, "LL": 7}

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.height * env.width
        self.action_size = len(env.actions)
        # self.QTable = np.ndarray((5,5,4),dtype=np.float32)
        self.QTable = np.random.rand(5,5,4)*4  # TODO: can change low and high values #state[0], state[1], actions
        self.QTable_jetpack = np.random.rand(5,5,8)*4  # TODO: can change low and high values #state[0], state[1], actions (u, uu, d, dd, r, rr, l, ll)
        self.QTable_jetpack = np.copy(self.QTable[:, :, :4])
        self.QTable_jetpack = np.concatenate((self.QTable_jetpack, self.QTable[:, :, :4]), axis=2)
        #define hyperparameters 
        self.lr = 0.01
        #low gamma->immediate rewards are important high gamma->future rewards
        self.gamma = 0.99
        self.epsilon_start = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.01
    def getQTable(self):
        if(self.env.have_jetpack == False):
            return self.QTable
        else:
            return self.QTable_jetpack
    
    def action(self,state):
        rand = random.uniform(0,1)
        if(self.env.have_jetpack == False):
            if rand > self.epsilon:
                action_index = action_mapping[np.argmax(self.QTable[state[0],state[1], :])]
                #print(f"action: {action_index}")
                #return np.argmax(self.QTable[state[0],state[1], :]) #select the action with the highest Q value for that state (returns the index -> coresponds to action)
                return action_index
            else: 
                randint = int(random.uniform(0,4))
                #print(f"action: {self.env.actions[randint]}")
                return self.env.actions[randint]
        else:
            if rand > self.epsilon:
                action_index = action_mapping[np.argmax(self.QTable_jetpack[state[0],state[1], :])]
                #print(f"action: {action_index}")
                #return np.argmax(self.QTable[state[0],state[1], :]) #select the action with the highest Q value for that state (returns the index -> coresponds to action)
                return action_index
            else: 
                randint = int(random.uniform(0,8))
                #print(f"action: {self.env.actions[randint]}")
                return self.env.actions[randint]
        
    def updateQvals(self, state, new_state, action, reward):
        #update Q table
        #𝑄𝑠,𝑎 =𝑄𝑠,𝑎 +𝑙∗(𝑟𝑠,𝑎 +𝛾∗max𝑄𝑠′,𝑎′ −𝑄𝑠,𝑎)
        action_index = reverse_action_mapping.get(action)
        #print(state, action, reward, new_state)

        if(self.env.have_jetpack==False):
            self.QTable[state[0],state[1],action_index] = self.QTable[state[0],state[1],action_index] + self.lr*(reward + self.gamma*np.max(self.QTable[new_state[0],new_state[1], :]) - self.QTable[state[0],state[1],action_index]) 
            self.QTable_jetpack[state[0],state[1],action_index] = self.QTable_jetpack[state[0],state[1],action_index] + self.lr*(reward + self.gamma*np.max(self.QTable[new_state[0],new_state[1], :]) - self.QTable_jetpack[state[0],state[1],action_index]) 
            self.QTable_jetpack[state[0],state[1],action_index+4] = self.QTable_jetpack[state[0],state[1],action_index+4] + self.lr*(reward + self.gamma*np.max(self.QTable[new_state[0],new_state[1], :]) - self.QTable_jetpack[state[0],state[1],action_index+4]) 
        else:
            self.QTable_jetpack[state[0],state[1],action_index] = self.QTable_jetpack[state[0],state[1],action_index] + self.lr*(reward + self.gamma*np.max(self.QTable[new_state[0],new_state[1], :]) - self.QTable_jetpack[state[0],state[1],action_index]) 
        
        #update epsilon
        #epsilon = self.epsilon_min + (self.epsilon_start-self.epsilon_min) * np.exp(-1*episode*self.epsilon_decay)


def train(env, episodes, agent):
    total_reward_per_episode = []
    for episode in range(0, episodes):
        #env.reset()
        done = False
        score = 0
        step = 0
        env.agent_location = env.starting_point
        #loop until done or max steps
        while not done:
            #action selection exploration explotiation
            state = env.agent_location
            action = agent.action(state)
            
            #take step in environment
            reward = env.step(action)
            #print(f"selected action: {action}, reward: {reward}")
            #update reward and state and step
            new_state = env.agent_location
            score += reward
            #update Q table
            agent.updateQvals(state, new_state, action, reward)
            #state = new_state
            #print(f"agent location: {env.agent_location}")
            if(env.is_done(env.agent_location)):
                done = True
            if(episode%100==0 or episode==episodes-1):
                print(f"selected action: {action}, reward: {reward}")
                print(f"agent location: {env.agent_location}")
                if env.agent_location == env.jetpack_point:
                    print("********JETPACK OBTAINED*********")
        total_reward_per_episode.append(score)

        #update epsilon
        agent.epsilon = agent.epsilon_min + (agent.epsilon_start-agent.epsilon_min) * np.exp(-1*episode*agent.epsilon_decay)
        
        if(episode%100==0 or episode==episodes-1):
            print('Episode:{} Score:{}'.format(episode, score))
            print("------------------------------------------------------------------------------------")  
        if(episode==episodes-1):
            print_ndarray(agent.getQTable())
        # reset environmet:
        env.have_jetpack = False
        env.first_time_jp = True
        env.actions = ['U', 'D', 'R', 'L']
        for water_point in env.water_points:
            env.grid[water_point[0], water_point[1]] = -90 
    return total_reward_per_episode

def print_ndarray(matrix):
    if matrix.shape == (5,5):
        print(np.flip(matrix.T, axis=0))
    elif matrix.shape == (5,5,4):
        for i in range(0,4):
            action_str = action_mapping[i]
            print(f"{action_str}:")
            print(np.flip(matrix[:,:,i].T, axis=0), '\n')
    elif matrix.shape == (5,5,8):
        for i in range(0,8):
            action_str = action_mapping[i]
            print(f"{action_str}:")
            print(np.flip(matrix[:,:,i].T, axis=0), '\n')

env = JetPackGridEnv()
agent = QLearningAgent(env) # TODO: initialize table with random values

print_ndarray(agent.getQTable())
print_ndarray(env.grid)

start_time = time.time()
total_reward_per_episode = train(env=env, episodes=1000, agent=agent)
end_time = time.time()
print(f"Learning time: {end_time - start_time} seconds")


plt.plot(total_reward_per_episode)
plt.title('Rewards per Episode (jetpack env)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.savefig(f'Q-learningGraphwithjetpack.png')
plt.show()









