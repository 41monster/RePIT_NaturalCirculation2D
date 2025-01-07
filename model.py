import torch
import torch.nn as nn

def make_mlp(input_dim=15, hidden_dim=512, num_hidden_layers=11, output_dim=1):
    # In Keras code: 
    # - input_shape=(15,) 
    # - 11 layers of Dense(512, relu)
    # - 1 layer of Dense(1)
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

# Define the networks from the snippet
class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.model = make_mlp(input_dim=15)
    def forward(self, x):
        return self.model(x)

class NetU(nn.Module):
    def __init__(self):
        super(NetU, self).__init__()
        self.model = make_mlp(input_dim=15)
    def forward(self, x):
        return self.model(x)

class NetV(nn.Module):
    def __init__(self):
        super(NetV, self).__init__()
        self.model = make_mlp(input_dim=15)
    def forward(self, x):
        return self.model(x)

# The PyTorch equivalent of FVMN model that holds multiple subnetworks
class FVMNetwork(nn.Module):
    def __init__(self):
        super(FVMNetwork, self).__init__()
        self.net_a = NetA()
        self.net_u = NetU()
        self.net_v = NetV()

    def forward(self, x):
        # x should be shape [batch, 15] for these subnets
        pred_a = self.net_a(x)
        pred_u = self.net_u(x)
        pred_v = self.net_v(x)
        return {
            'T': pred_a, 
            'U_x': pred_u, 
            'U_y': pred_v
        }
    

if __name__ == "__main__":
    # Create a dummy input tensor
    x = torch.randn(100, 15)
    # Create the model
    model = FVMNetwork()
    # Forward pass
    output = model(x)
    print(output["T"].shape)