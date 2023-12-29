import numpy as np
import random
import matplotlib.pyplot as plt
import time
import Qlearning


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

        total_reward_per_episode.append(score)
        #update epsilon
        agent.epsilon = agent.epsilon_min + (agent.epsilon_start-agent.epsilon_min) * np.exp(-1*episode*agent.epsilon_decay)
        if(episode%100==0 or episode==episodes-1):
            print('Episode:{} Score:{}'.format(episode, score))
            print("------------------------------------------------------------------------------------")  
        #print('Episode:{} Score:{}'.format(episode, score))
        #print(f"total reward: {total_reward_per_episode}")
        #print("------------------------------------------------------------------------------------")
    return total_reward_per_episode

env = Qlearning.VanillaGridEnv()
agent = Qlearning.QLearningAgent(env) 

Qlearning.print_ndarray(env.grid)

start_time = time.time()
total_reward_per_episode = train(env=env, episodes=1000, agent=agent)
end_time = time.time()
Qlearning.print_ndarray(agent.getQTable())
print(f"Learning time: {end_time - start_time} seconds")



plt.plot(total_reward_per_episode)
plt.title('Rewards per Episode (vanilla env)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.savefig(f'Q-learningGraph.png')
plt.show()