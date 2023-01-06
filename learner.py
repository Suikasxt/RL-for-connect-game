import torch
import time
import argparse
import settings
import numpy as np
from model import DQNModel
from mempool import SampleManager
from modelpool import ModelManager
from actor import gameTest
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
import matplotlib.pyplot as plt

class MyDQNPolicy(DQNPolicy):
    def _target_q(self, batch):
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            return target_q[np.arange(len(result.act)), result.act]
        else:  # Nature DQN, over estimate
            return target_q.max(dim=1)[0]

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=float, default=0)
args = parser.parse_args()
if torch.cuda.is_available() and args.gpu != -1:
    device = torch.device('cuda:%d' % args.gpu)
else:
    device = torch.device('cpu')

    

def train(policy, sample_manager):
    batch_size = settings.BATCH_SIZE
    loss_list = []
    for i in range(100):
        state, action, reward, next_state, done = sample_manager.getSample(batch_size)
        batch = Batch(obs=torch.tensor(state, device=device),
                      act=torch.tensor(action, device=device),
                      rew=torch.tensor(reward, device=device),
                      obs_next=torch.tensor(next_state, device=device),
                      done=torch.tensor(done, device=device),
                      info=[{}]*batch_size)
        
        batch.returns=(policy._target_q(batch) * (~batch.done) * settings.GAMMA + batch.rew).detach()
        loss = policy.learn(batch)['loss']
        loss_list.append(loss)
    return np.mean(loss_list)
        
        
        

if __name__ == "__main__":
    model = DQNModel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    policy = MyDQNPolicy(model, optimizer, settings.GAMMA, target_update_freq=0)
    #print(model.state_dict())
    #exit(0)
    
    sample_manager = SampleManager('receiver')
    model_manager = ModelManager('sender')
    sample_manager.start()
    
    plt.ion()
    reward_list = []
    loss_list = []
    epoch = 0
    sample_list = []
    while epoch < 80:
        torch.manual_seed(time.time())
        print('learner', len(sample_manager.buffer))
        sample_list.append(len(sample_manager.buffer))
        for i in range(1):
            loss = None
            if len(sample_manager.buffer):
                loss = train(policy, sample_manager)
        
        if epoch%5==0:
            model_manager.sendModel(policy.model.state_dict())
            gameTest(policy.model, True)
            model_manager.saveModel(policy.model.state_dict())
        epoch += 1
        
        reward = 0
        for i in range(100):
            reward += gameTest(policy.model)
        #gameTest(policy.model, True)
        #print('Test reward', reward)
        reward_list.append(reward)
        if loss is not None:
            loss_list.append(loss)
        
        plt.subplot(1, 2, 1)
        plt.plot(reward_list)
        plt.title('rew')
        plt.subplot(1, 2, 2)
        plt.plot(loss_list)
        plt.title('loss')
        plt.pause(0.1)
        
    print(reward_list)
    print(reward_list)
    print(loss_list)
    print(reward_list)
        