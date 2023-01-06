import numpy as np
import json
import torch
import settings
import socket
import threading
import time
from collections import OrderedDict
import pdb
import time
from model import DQNModel


class ModelManager(threading.Thread):
    def __init__(self, mode='sender'):
        super().__init__()
        self.ip = settings.MODEL_POOL_IP
        self.port = settings.MODEL_POOL_PORT
        self.address = (self.ip, self.port)
        self.buffer_size = settings.MODEL_POOL_SIZE
        self.mode = mode
        self.old_data = {}
        self.block_size = 1024
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.socket = socket.socket()
        self.model_summary =  {k:v.shape for k, v in DQNModel().state_dict().items()}
        self.model_params = list(self.model_summary)
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
        # 按参数名 一维列表索引 值构成三元组的形式压缩数据
        data = {}
        eps = .2
        for param in model:
            data[param] = model[param].cpu().numpy().flatten()

        if len(self.old_data) == 0:
            for param in data:
                self.old_data[param] = np.zeros_like(data[param])
        packet = {}
        for id, k in enumerate(data.keys()):
            delta = data[k] - self.old_data[k]
            vardelta = delta / data[k]                      #相对误差超过1%更新
            idx = np.where(np.abs(vardelta) > eps)[0]
            packet[id] = [(int(k),float(v)) for k,v in zip(idx, data[k][idx])]

        self.old_data = data
        json_packet = json.dumps(packet).encode('utf-8')
        for key in data:
            data[key] = data[key].tolist()
        json_data = json.dumps(data).encode('utf-8')
        compress_rate = len(json_packet) / len(json_data)
        print("compress_rate: ", compress_rate)
        return json.dumps(packet).encode('utf-8')
    
    def modelDecode(self, packet):
        packet = json.loads(packet.decode('utf-8'))
        
        data = {}
        if len(self.old_data) == 0:
            for key in packet:
                k = int(key)
                data[k] = [v for idx, v in packet[key]]
        else:
            for key in packet:
                k = int(key)
                data[k] = self.old_data[k]
                for idx, v in packet[key]:
                    data[k][idx] = v 
        self.old_data = data
        model = OrderedDict()
        for key in data:
            param = self.model_params[key]
            data_shaped = np.array(data[key]).reshape(self.model_summary[param])
            model[param] = torch.tensor(data_shaped, dtype=torch.float32)
        return model
        
    
    def sendModel(self, model):
        data = self.modelEncode(model)
        length = len(data)
        print('Send model.')
        
        self.client.send(str(length).encode('utf-8'))
        time.sleep(1)
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