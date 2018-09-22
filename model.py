import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.action_size = action_size
        
        self.conv1 = nn.Conv2d(in_channels=state_size, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=7*7*64, out_features=fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        self.fc3 = nn.Linear(in_features=fc2_units, out_features=action_size)

        self.relu = nn.ReLU()

        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)

        # batch_size = x.size(0)
        x = self.relu(self.conv1(state))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        #val = self.relu(self.fc1_val(x))

        #adv = self.fc2_adv(adv)
        #val = self.fc2_val(val).expand(x.size(0), self.action_size)

        return x


    
    
class Dueling_DQN(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(Dueling_DQN, self).__init__()
        self.action_size = action_size
        
        self.conv1 = nn.Conv2d(in_channels=state_size, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=action_size)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)
        
        # self.seed = torch.manual_seed(seed)
        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        # self.fc3_adv = nn.Linear(in_features=fc2_units, out_features=action_size)
        # self.fc3_val = nn.Linear(in_features=fc2_units, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, state):
        #batch_size = x.size(0)
        x = self.relu(self.conv1(state))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.action_size)
        
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # adv = self.fc3_adv(x)
        # val = self.fc3_val(x).expand(x.size(0), self.action_size)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x