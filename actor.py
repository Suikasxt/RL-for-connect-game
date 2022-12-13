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
        if (self.mode=='train' and np.random.rand() < 0.2) or self.mode == 'random':
            return np.random.randint(settings.BOARD_SIZE**2)
        if turn:
            state = 1 - state - 3*(state==-1)
        state_space = state.reshape((-1)) != -1
        
        state = torch.tensor(state).unsqueeze(0)
        self.model.eval()
        value = self.model.forward(state).detach()[0, :].cpu().numpy()
        value[state_space] = -1e9
        action = np.argmax(value)
        return int(action)

def gameTrain(sample_manager, model_manager):
    if len(model_manager.buffer) == 0:
        return
    
    env = Gobang(settings.BOARD_SIZE, settings.GAME_TARGET)
    agents = [Agent('train'), Agent('train')]
    agents[0].load(model_manager.getLatestModel())
    agents[1].load(model_manager.getLatestModel())
    last_state = None
    last_action = None
    reward_count = np.array([0, 0])
    (state, turn) = env.reset()
    done = False
    while not done:
        action = agents[turn].act(deepcopy(state), turn)
        (next_state, next_turn), reward, done, info = env.step(action)
        
        reward_count = reward_count + reward
        if not (last_state is None):
            sample_manager.sendSample(deepcopy(last_state), last_action, reward_count[next_turn], deepcopy(next_state), done, next_turn)
            reward_count[next_turn] = 0
        last_state = state
        last_action = action
        state = next_state
        turn = next_turn
    
    next_turn ^= 1
    sample_manager.sendSample(deepcopy(last_state), last_action, reward_count[next_turn], deepcopy(state), done, next_turn)
        

def gameTest(model):
    env = Gobang(settings.BOARD_SIZE, settings.GAME_TARGET)
    agents = [Agent('test'), Agent('test')]
    ours = np.random.randint(2)
    agents[ours].load(model.state_dict())
    
    (state, turn) = env.reset()
    total_reward = np.array([0, 0])
    done = False
    while not done:
        action = agents[turn].act(state, turn)
        (next_state, next_turn), reward, done, info = env.step(action)
        total_reward += reward
        
        state = next_state
        turn = next_turn
        
    return total_reward[ours] - total_reward[ours^1]
        

if __name__ == "__main__":
    sample_manager = SampleManager('sender')
    model_manager = ModelManager('receiver')
    model_manager.start()
    
    while True:
        time.sleep(0.1)
        print('actor', len(model_manager.buffer))
        if len(model_manager.buffer):
            gameTrain(sample_manager, model_manager)
            