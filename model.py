import torch
import settings


class DQNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.board_size = settings.BOARD_SIZE
        layer_size = self.board_size**2
        self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
                    torch.nn.Sigmoid(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    
                    torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1),
                    torch.nn.Sigmoid(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                )
        
        self.fc = torch.nn.Sequential(
                    torch.nn.Linear(layer_size, layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(layer_size, layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(layer_size, layer_size),
                )
    
    def forward(self, state):
        state_0 = (state == 0).unsqueeze(1)
        state_1 = (state == 1).unsqueeze(1)
        state = torch.concat((state_0, state_1), dim=1).float()
        feature = self.conv(state)
        feature = feature.view(-1, self.board_size**2)
        value = self.fc(feature)
        return value
    