from reinforcement_learning.environment import BlockudokuEnv
from reinforcement_learning.agent import DQNAgent
import numpy as np

def train(episodes, max_steps, batch_size=32):
    env = BlockudokuEnv()
    state_size = 9*9 + 3*3*5  # 보드 크기 + 3개의 최대 블록 크기 (3x5)
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.train(batch_size)
            
            if done:
                break
        
        if episode % 10 == 0:
            agent.update_target_model()
        
        print(f"Episode: {episode+1}/{episodes}, Score: {env.game.score}, Total Reward: {total_reward}")
    
    print("Training finished.")
