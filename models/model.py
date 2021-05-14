import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

random_seed = 777

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# arrhythmia detection model
class ATMNet(nn.Module):
    def __init__(self):
        super(ATMNet, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding_mode="zeros",
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding_mode="zeros",
            padding=1,
        )
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding_mode="zeros",
            padding=1,
        )
        self.conv4 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding_mode="zeros",
            padding=1,
        )
        self.conv5 = nn.Conv1d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            padding_mode="zeros",
            padding=1,
        )
        self.max_pool_1d = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(512 * 5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.max_pool_1d(F.relu(self.conv1(x)))
        x = self.max_pool_1d(F.relu(self.conv2(x)))
        x = self.max_pool_1d(F.relu(self.conv3(x)))
        x = self.max_pool_1d(F.relu(self.conv4(x)))
        x = self.max_pool_1d(F.relu(self.conv5(x)))
        x = x.view(-1, 512 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x





