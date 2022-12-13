import numpy as np
import json
import torch
import settings
import socket
import threading
import time
from collections import OrderedDict


class ModelManager(threading.Thread):
    def __init__(self, mode='sender'):
        super().__init__()
        self.ip = settings.MODEL_POOL_IP
        self.port = settings.MODEL_POOL_PORT
        self.address = (self.ip, self.port)
        self.buffer_size = settings.MODEL_POOL_SIZE
        self.mode = mode
        
        self.block_size = 1024
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.socket = socket.socket()
        if mode == 'sender':
            self.socket.bind(self.address)
            self.socket.listen(5)
            self.client, address = self.socket.accept()
        else:
            while True:
                try:
                    self.socket.connect(self.address)
                except socket.error:
                    continue
                break
            
    
    def modelEncode(self, model):
        data = {}
        for param in model:
            data[param] = model[param].cpu().numpy().tolist()
        return json.dumps(data).encode('utf-8')
    
    def modelDecode(self, data):
        data = json.loads(data.decode('utf-8'))
        model = OrderedDict()
        for key in data:
            model[key] = torch.tensor(data[key], dtype=torch.float32)
        return model
        
    
    def sendModel(self, model):
        data = self.modelEncode(model)
        length = len(data)
        print('Send model.')
        
        self.client.send(str(length).encode('utf-8'))
        for i in range((length-1)//self.block_size+1):
            self.client.send(data[i*self.block_size : min((i+1)*self.block_size, length)])
    
    def recvModel(self):
        length = int(self.socket.recv(self.block_size).decode('utf-8'))
        data = b''
        while len(data) < length:
            data = data + self.socket.recv(self.block_size)
        return self.modelDecode(data)
    
    def getLatestModel(self):
        self.buffer_lock.acquire()
        model = self.buffer[-1]
        self.buffer_lock.release()
        
        return model
    
    def getRandomModel(self):
        self.buffer_lock.acquire()
        model = np.random.choice(self.buffer)
        self.buffer_lock.release()
        
        return model
    
    def run(self):
        while True:
            model = self.recvModel()
            
            self.buffer_lock.acquire()
            while len(self.buffer) >= self.buffer_size:
                self.buffer = self.buffer[::2]
            self.buffer.append(model)
            self.buffer_lock.release()
    
    def saveModel(self, model):
        torch.save(model,'./pth/model.pth')