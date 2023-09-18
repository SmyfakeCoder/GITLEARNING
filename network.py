from torch import nn

class mnistnet(nn.Module):
    def __init__(self):
        super(mnistnet,self).__init__()
        self.layer1 = nn.Linear(784,300)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(300,10)
    def forward(self,X):
        # print(X)
        X = X.reshape(-1,28*28)
        X = self.layer1(X)
        X = self.relu(X)
        y = self.layer2(X)
        return y
