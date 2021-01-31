import torch.nn as nn
import torch.nn.functional as F

class Model_luzon(nn.Module):
    def __init__(self, number_of_features):
        super().__init__()
        self.linear1 = nn.Linear(number_of_features, 30)
        self.linear2 = nn.Linear(30, 10)
        self.linear3 = nn.Linear(10, 1)

    def forward(self, x):
        # Now it only takes a call to the layer to make predictions
        x = F.dropout(self.linear1(x), 0.1)
        x = F.leaky_relu(self.linear2(x))
        x = F.dropout(self.linear3(x), 0.1)
        return F.leaky_relu(x)


class Model(nn.Module):
    def __init__(self, number_of_features):
        super().__init__()
        self.linear1 = nn.Linear(number_of_features, 60)
        self.linear2 = nn.Linear(60, 30)
        self.linear3 = nn.Linear(30, 10)
        self.linear4 = nn.Linear(10, 1)

    def forward(self, x):
        # Now it only takes a call to the layer to make predictions
        x = F.dropout(self.linear1(x), 0.1)
        x = F.leaky_relu(self.linear2(x))
        x = F.dropout(self.linear3(x), 0.1)
        return F.leaky_relu(self.linear4(x))