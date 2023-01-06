import numpy as np
import json
import settings
import socket
import threading
from copy import deepcopy
import sys
import gzip, zlib, bz2

class SampleManager(threading.Thread):
    def __init__(self, mode='sender'):
        super().__init__()
        self.ip = settings.MEM_POOL_IP
        self.port = settings.MEM_POOL_PORT
        self.address = (self.ip, self.port)
        self.buffer_size = settings.MEM_POOL_SIZE
        self.mode = mode
        
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # *********************** compress ***********************
        # 记录数据量，最后一位为样本总数
        self.size = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)
        if mode == 'receiver':
            self.socket.bind(self.address)
            # *********************** compress ***********************
            self.state = np.zeros((settings.BOARD_SIZE, settings.BOARD_SIZE), dtype=np.int32) - 1


    # *********************** compress ***********************
    def sampleEnode_compress(self, last_action, action, reward, done):
        sample = {
            'last_action': int(last_action) if last_action is not None else None,
            'action': int(action) if action is not None else None,
            'reward': float(reward),
            'done': bool(done)
        }
        return json.dumps(sample).encode('utf-8')

    def sampleDecode_compress(self, data):
        sample = json.loads(data.decode('utf-8'))
        state = self.state
        next_state = deepcopy(state)
        action_list = [sample['last_action'], sample['action']]
        for i in range(2):
            action = action_list[i]
            if type(action) == int and 0 <= action < settings.BOARD_SIZE ** 2:
                action = [action // settings.BOARD_SIZE, action % settings.BOARD_SIZE]
                if next_state[action[0]][action[1]] == -1:
                    next_state[action[0]][action[1]] = i

        return state, sample['last_action'], sample['reward'], next_state, sample['done']

    def sendSample_compress(self, last_action, action, reward, done, turn):
        data = self.sampleEnode_compress(last_action, action, reward, done)
        # print('Incremental Compressed Size = {}'.format(len(data)))
        self.size[4] += len(data)
        self.size[5] += len(gzip.compress(data))
        self.size[6] += len(zlib.compress(data))
        self.size[7] += len(bz2.compress(data))
        if done and self.size[8] % 1000 == 0:
            print('Incremental Compressed Size = {}'.format(self.size[4]))
            print('Gzip Incremental Compressed Size = {}'.format(self.size[5]))
            print('Zlib Incremental Compressed Size = {}'.format(self.size[6]))
            print('Bz2 Incremental Compressed Size = {}'.format(self.size[7]))
            self.size[4: 8] *= 0
        # print('Send sample.')
        '''print(state)
        print(action, reward, done)
        print(next_state)
        print('\n')
        import time
        time.sleep(1)'''
        # self.socket.sendto(data, self.address)
    # ********************************************************


    def sampleEnode(self, state, action, reward, next_state, done):
        if type(state) == np.ndarray:
            state = state.tolist()

        if type(next_state) == np.ndarray:
            next_state = next_state.tolist()

        sample = {
            'state': state,
            'action': int(action),
            'reward': float(reward),
            'next_state': next_state,
            'done': bool(done)
        }
        return json.dumps(sample).encode('utf-8')

    def sampleDecode(self, data):
        sample = json.loads(data.decode('utf-8'))
        return sample['state'], sample['action'], sample['reward'], sample['next_state'], sample['done']


    def sendSample(self, state, action, reward, next_state, done, turn):
        if turn:
            state = 1 - state - 3*(state==-1)
            next_state = 1 - next_state - 3*(next_state==-1)
        data = self.sampleEnode(state, action, reward, next_state, done)

        # print('Size = {}'.format(len(data)))
        # print('Gzip Compressed Size = {}'.format(len(gzip.compress(data))))
        # print('Zlib Compressed Size = {}'.format(len(zlib.compress(data))))
        # print('Bz2 Compressed Size = {}'.format(len(bz2.compress(data))))
        self.size[0] += len(data)
        self.size[1] += len(gzip.compress(data))
        self.size[2] += len(zlib.compress(data))
        self.size[3] += len(bz2.compress(data))
        if done:
            if self.size[8] % 1000 == 0:
                print('Original Size = {}'.format(self.size[0]))
                print('Gzip Compressed Size = {}'.format(self.size[1]))
                print('Zlib Compressed Size = {}'.format(self.size[2]))
                print('Bz2 Compressed Size = {}'.format(self.size[3]))
                self.size[:4] *= 0
            self.size[8] += 1

        # print('Send sample.')
        '''print(state)
        print(action, reward, done)
        print(next_state)
        print('\n')
        import time
        time.sleep(1)'''
        self.socket.sendto(data, self.address)


    
    def recvSample(self):
        data, address = self.socket.recvfrom(4096)
        return self.sampleDecode(data)
    
    def getSample(self, size):
        self.buffer_lock.acquire()
        samples = np.random.choice(self.buffer, size)
        self.buffer_lock.release()

        # *********************** compress ***********************
        none_action = []
        for i in range(len(samples)):
            if samples[i]['action'] is None:
                none_action.append(i)
        samples = np.delete(samples, none_action)
        # ********************************************************

        state = [sample['state'] for sample in samples]
        action = [sample['action'] for sample in samples]
        reward = [sample['reward'] for sample in samples]
        next_state = [sample['next_state'] for sample in samples]
        done = [sample['done'] for sample in samples]
        return state, action, reward, next_state, done
    
    def run(self):
        while True:
            state, action, reward, next_state, done = self.recvSample()
                
            self.buffer_lock.acquire()
            self.buffer.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            if len(self.buffer) >= self.buffer_size:
                self.buffer = self.buffer[-self.buffer_size:]
            self.buffer_lock.release()
            
        