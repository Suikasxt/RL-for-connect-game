import torch
import settings


class DQNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.board_size = settings.BOARD_SIZE
        layer_size = self.board_size**2
        self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
                    #torch.nn.Sigmoid(),
                    #torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    
                    torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
                    #torch.nn.Sigmoid(),
                    #torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                )
        
        self.fc = torch.nn.Sequential(
                    torch.nn.Linear(4*layer_size, 2*layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2*layer_size, layer_size),
                )
    
    def forward(self, obs, state=None, info={}):
        obs_0 = (obs == 0).unsqueeze(1)
        obs_1 = (obs == 1).unsqueeze(1)
        obs = torch.concat((obs_0, obs_1), dim=1).float()
        feature = self.conv(obs)
        feature = feature.view(feature.shape[0], -1)
        value = self.fc(feature)
        return value, state
    