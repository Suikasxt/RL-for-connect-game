import torch
import time
import argparse
import settings
import numpy as np
from model import DQNModel
from mempool import SampleManager
from modelpool import ModelManager
from actor import gameTest

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=float, default=0)
args = parser.parse_args()
if torch.cuda.is_available() and args.gpu != -1:
    device = torch.device('cuda:%d' % args.gpu)
else:
    device = torch.device('cpu')

    

def train(model, sample_manager):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    batch_size = settings.BATCH_SIZE
    loss_list = []
    for i in range(100):
        optimizer.zero_grad()
        state, action, reward, next_state, done = sample_manager.getSample(batch_size)
        
        state = torch.tensor(state, device=device)
        action = torch.tensor(action, device=device)
        reward = torch.tensor(reward, device=device)
        next_state = torch.tensor(next_state, device=device)
        done = torch.tensor(done, device=device, dtype=torch.bool)
        
        
        target_value = torch.max(model(next_state).detach(), dim=-1)[0]
        value = model(state)[(torch.arange(batch_size), action)]
        target_value = target_value * (~done) * settings.GAMMA + reward
        
        #print(state[0])
        #print(action[0])
        #print(next_state[0])
        #print(value[0])
        #print(target_value[0])
        #print(reward[0])
        loss = torch.sum((target_value - value)**2)
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
    print(np.mean(loss_list))
        
        
        

if __name__ == "__main__":
    model = DQNModel()
    model.to(device)
    #print(model.state_dict())
    #exit(0)
    
    sample_manager = SampleManager('receiver')
    model_manager = ModelManager('sender')
    sample_manager.start()
    
    while True:
        torch.manual_seed(time.time())
        for i in range(3):
            print('learner', len(sample_manager.buffer))
            if len(sample_manager.buffer):
                train(model, sample_manager)
            time.sleep(0.1)
        model_manager.sendModel(model.state_dict())
        model_manager.saveModel(model.state_dict())
        
        reward = 0
        for i in range(100):
            reward += gameTest(model)
        print('Test reward', reward)