import numpy as np
from env import Gobang
from copy import deepcopy
import settings
from mempool import SampleManager
from modelpool import ModelManager
from model import DQNModel
import time
import torch

class Agent:
    def __init__(self, mode = 'test') -> None:
        self.latest = False
        self.model = DQNModel()
        self.mode = mode
    
    def load(self, param):
        self.model.load_state_dict(param)
    
    def act(self, state, turn):
        if turn:
            state = 1 - state - 3*(state==-1)
        state_block = state.reshape((-1)) != -1
        
        if self.mode == 'common':
            for i in range(settings.BOARD_SIZE**2):
                if not state_block[i]:
                    return i
                
        if (self.mode=='train' and np.random.rand() < 0.2) or self.mode == 'random':
            return np.random.randint(settings.BOARD_SIZE**2)
        
        state = torch.tensor(state).unsqueeze(0)
        self.model.eval()
        value = self.model.forward(state)[0].detach()[0, :].cpu().numpy()
        value[state_block] = -1e9
        action = np.argmax(value)
        return int(action)

def gameTrain(sample_manager, model_manager):
    if len(model_manager.buffer) == 0:
        return
    
    env = Gobang(settings.BOARD_SIZE, settings.GAME_TARGET)
    agents = [Agent('train'), Agent('test')]
    agents[0].load(model_manager.getLatestModel())
    agents[0].latest = True
    #agents[1].load(model_manager.getRandomModel())
    #agents[1].latest = True
    np.random.shuffle(agents)
    
    last_state = None
    last_action = None
    reward_count = np.array([0, 0])
    (state, turn) = env.reset()
    done = False
    last_action = None
    while not done:
        action = agents[turn].act(deepcopy(state), turn)
        (next_state, next_turn), reward, done, info = env.step(action)
        
        reward_count = reward_count + reward
        if not (last_state is None) and not (agents[next_turn].mode in ['random', 'common']) and agents[next_turn].latest:
            sample_manager.sendSample(deepcopy(last_state), last_action, reward_count[next_turn], deepcopy(next_state), done, next_turn)
            reward_count[next_turn] = 0
        # *********************** compress ***********************
        if not (agents[next_turn].mode in ['random', 'common']) and agents[next_turn].latest:
            sample_manager.sendSample_compress(last_action, action, reward_count[next_turn], done, next_turn)
        # ********************************************************
        last_action = action
        last_state = state
        state = next_state
        turn = next_turn
    
    next_turn ^= 1
    sample_manager.sendSample(deepcopy(last_state), last_action, reward_count[next_turn], deepcopy(state), done, next_turn)
    # *********************** compress ***********************
    sample_manager.sendSample_compress(last_action, None, reward_count[next_turn], done, next_turn)
    # ********************************************************
        

def gameTest(model, render=False):
    env = Gobang(settings.BOARD_SIZE, settings.GAME_TARGET)
    agents = [Agent('test'), Agent('test')]
    ours = np.random.randint(2)
    agents[ours].load(model.state_dict())
    if render:
        agents[ours^1].load(model.state_dict())
    
    (state, turn) = env.reset()
    total_reward = np.array([0., 0.])
    done = False
    while not done:
        action = agents[turn].act(state, turn)
        (next_state, next_turn), reward, done, info = env.step(action)
        if render:
            print(state)
            print(action)
            print(next_state)
            print('\n\n\n')
        total_reward += reward
        
        state = next_state
        turn = next_turn
        
    return total_reward[ours]        

if __name__ == "__main__":
    sample_manager = SampleManager('sender')
    model_manager = ModelManager('receiver')
    model_manager.start()
    
    while True:
        time.sleep(0.1)
        print('actor', len(model_manager.buffer))
        if len(model_manager.buffer):
            for i in range(10):
                gameTrain(sample_manager, model_manager)
            