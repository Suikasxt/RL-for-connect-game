import numpy as np
import json
import settings
import socket
import threading

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
        if mode == 'receiver':
            self.socket.bind(self.address)
    
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
        #print('Send sample.')
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
            
        