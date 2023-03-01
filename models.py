import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class Network(torch.nn.Module):
    def __init__(self, context):
        super(Network, self).__init__()
        # TODO: Please try different architectures
        self.in_size = int(((context - 1)*2 + 3 ) *13)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU(0.1)
        layers = [nn.Linear(self.in_size, 2*self.in_size),
            nn.BatchNorm1d(2*self.in_size),
            self.relu,
            self.dropout,
            nn.Linear(2*self.in_size, 3*self.in_size),
            nn.BatchNorm1d(3*self.in_size),
            self.relu,
            self.dropout,
            nn.Linear(3*self.in_size, 4*self.in_size),
            nn.BatchNorm1d(4*self.in_size),
            self.relu,
            self.dropout,
            nn.Linear(4*self.in_size, 3*self.in_size),
            nn.BatchNorm1d(3*self.in_size),
            self.relu,
            self.dropout,
            nn.Linear(3*self.in_size, 2*self.in_size),
            nn.BatchNorm1d(2*self.in_size),
            self.relu,
            self.dropout,
            nn.Linear(2*self.in_size, self.in_size),
            nn.BatchNorm1d(self.in_size),
            self.relu,
            self.dropout,
            nn.Linear(self.in_size, 64),
            nn.BatchNorm1d(64),
            self.relu,
            self.dropout,
            nn.Linear(64, 40)
        ]
        self.laysers = nn.Sequential(*layers)

    def forward(self, A0):
        x = self.laysers(A0)
        return x