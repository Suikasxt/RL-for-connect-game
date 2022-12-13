from copy import deepcopy
import numpy as np

DIRECTION_8 = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])
class Gobang:
    def __init__(self, size = 9, target = 4):
        self.size = size
        self.target = target
        self.shape = (size, size)
        self.reset()
    
    def reset(self):
        self.state = np.zeros((self.size, self.size), dtype=np.int32) - 1
        self.turn = 0
        return (deepcopy(self.state), self.turn)
    
    def vaildCoordinate(self, coord):
        if any(coord < 0) or any(coord >= self.shape):
            return False
        return True
    
    def judgeFinish(self):
        state = self.state
        assert(np.min(state) >= -1)
        assert(np.max(state) <= 2)
        
        win = [False, False]
        space_count = 0
        
        for dir in DIRECTION_8[:4]:
            count = np.zeros(self.shape, dtype=np.uint8)
            for x in range(self.size):
                for y in range(self.size):
                    if state[x][y] == -1:
                        space_count += 1
                        continue
                    last_pos = np.array([x, y]) - dir
                    if self.vaildCoordinate(last_pos) and state[x][y] == state[last_pos[0]][last_pos[1]]:
                        count[x][y] = count[last_pos[0]][last_pos[1]] + 1
                    else:
                        count[x][y] = 1
                    if count[x][y] == self.target:
                        win[state[x][y]] = True
        
        if all(win):
            return True, -1
        for p in [0, 1]:
            if win[p]:
                return True, p
        if space_count == 0:
            return True, -1
        return False, None
    
    def step(self, position):
        reward = [-0, -0]
        if type(position) == int:
            position = [position//self.size, position%self.size]
        position = np.array(position)
        
        if self.vaildCoordinate(position) and self.state[position[0]][position[1]] == -1:
            self.state[position[0]][position[1]] = self.turn
        else:
            reward[self.turn] -= 10
        self.turn ^= 1
        
        done, winner = self.judgeFinish()
        if done and winner != -1:
            reward[winner] += 30
            reward[winner^1] -= 30
        return (deepcopy(self.state), self.turn), reward, done, {}
        
        
        
        
    