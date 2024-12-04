from dataclasses import dataclass
from pathlib import Path
import torch.cuda as cuda
import torch

@dataclass
class BaseConfig:
    # Data parameters
    data_path: Path = Path("./Assets/natural_convection")
    time_step:int = 0.01
    
    data_dim:int = 2 # It is to denote either data is 1D or 2D or 3D.
    grid_x: int = 200 # Number of grid points in x-direction
    grid_y: int = 200 # Number of grid points in y-direction
    grid_z: int = 1 # Number of grid points in z-direction

class TrainingConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.batch_size: int = 10000
        self.epochs: int = 5
        self.learning_rate: float = 0.001
        self.device: str = "cuda" if cuda.is_available() else "cpu"
        self.optimizer = torch.optim.Adam
        self.loss = torch.nn.MSELoss()
        self.activation = torch.nn.ReLU
        self.data_vars = {"vectors":["U"],"scalars":["T"]}
        self.model_path = Path("./model.pth")
        self.training_start_time = 5.0
        self.training_end_time = 5.02
        self.prediction_start_time = 5.02
        self.prediction_end_time = 5.05

    def get_variables(self, vars_dict:dict=None):
        """
        Because we have a dictionary that separates variables into scalars and vectors,
        we need to combine them into a single list. We need to further differentiate vectors 
        based on the number of dimensions (1D or 2D or 3D).
        """
        self.data_vars = self.data_vars if vars_dict is None else vars_dict
        vars_list = []
        for key, value in self.data_vars.items():
            if key == "vectors":
                for var in value:
                    if self.data_dim == 1:
                        vars_list.append(f"{var}_x")
                    elif self.data_dim == 2:
                        vars_list.extend([f"{var}_x", f"{var}_y"])
                    elif self.data_dim == 3:
                        vars_list.extend([f"{var}_x", f"{var}_y", f"{var}_z"])
            elif key == "scalars":
                vars_list.extend(value)
            else: 
                raise ValueError(f"Invalid key {key} in vars_dict. Must be either 'vectors' or 'scalars'.")
        return vars_list
    
    def extend_variables(self, vars_dict:dict=None):
        '''
        There are cases where we don't exactly need dimension separated variables, just the variables themselves just like 
        how they are represented in OpenFOAM. Hence, this method is to return the variables as they are, as a list.
        '''
        vars_dict = self.data_vars if vars_dict is None else vars_dict
        return [var for key, value in vars_dict.items() for var in value]
    
if __name__ == "__main__":
    training_config = TrainingConfig()
    print(training_config.get_variables())
    print(training_config.extend_variables())