import torch
from config import TrainingConfig

class FVMNetwork(torch.nn.Module):
    def __init__(self, training_config:TrainingConfig, vars_list:list=None,
                 hidden_layers:int=11, hidden_size:int=512, activation:torch.nn.ReLU=None):
        '''
        Args: 
        training_config: configuration classes are set to give minimum arguments during initialization.
        vars_list: list containing the variables to be predicted. If None, it will be taken from the training_config.
                    example: ["U_x", "U_y", "T"]
        hidden_layers: number of hidden layers in the network
        hidden_size: number of neurons in each hidden layer
        activation: activation function to be used in the hidden
        '''
        super().__init__()
        self.num_dims = training_config.data_dim
        self.vars = training_config.get_variables() if vars_list is None else vars_list
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.activation = activation if activation is not None else training_config.activation
        
        # Create networks dynamically based on vars_list
        self.networks = torch.nn.ModuleDict(
            {
                f"net_{var}": self._build_network() for var in self.vars
            }
        )

    def forward(self,x:torch.Tensor):
        '''
        returns output for each variable as a tuple: 
        (ux_hat, uy_hat, t_hat)
        '''
        outputs = [net(x) for _, net in self.networks.items()]
        return torch.cat(outputs, dim=1)

    def _build_network(self):
        """
        Build a single sub-network architecture.
        """
        input_shape = 15
        output_shape = 1
        layers = [torch.nn.Linear(input_shape, self.hidden_size), self.activation()]
        for _ in range(self.hidden_layers - 1):
            layers.extend([torch.nn.Linear(self.hidden_size, self.hidden_size), self.activation()])
        layers.append(torch.nn.Linear(self.hidden_size, output_shape))
        return torch.nn.Sequential(*layers)

if __name__ == "__main__":
    model = FVMNetwork(TrainingConfig())
    
    dummy_input = torch.randn(10,15)
    output = model(dummy_input)
    print(output.shape)