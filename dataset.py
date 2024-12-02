from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple
from torch import Tensor,mean, std
from config import TrainingConfig

class FVMNDataset(Dataset):
    def __init__(self, training_config:TrainingConfig, data_path:Path=None, start_time:float=None, 
                 end_time:float=None, time_step:float=None, vars_list:list=None):
        '''
        Keep in mind:
        1. To prepare for the training data, we must have numpy files from start_time to end_time mentioned here.
        Example: 
        start_time = 0, end_time=5, then we must have numpy files from 0 to 5.
        2. We don't use the fifth time step data as a network input. 
        but, we use the difference between the fifth and fourth time step as a network label.
        Example: 
        we give input from start_time(0) to end_time(5) - time_step(1) then labels will be:
        1 - 0 : 1st label
        2 - 1 : 2nd label
        3 - 2 : 3rd label
        4 - 3 : 4th label
        5 - 4 : 5th label

        input_shape from 0 to 4: [(grid_x-2) * (grid_y-2) * 5, 15]
        label_shape from 1 to 5: [(grid_x-2) * (grid_y-2) * 5, 3]

        Note:
        vars_list: list containing the variables to be predicted. If None, it will be taken from the training_config.
                    here, we should be careful that we are not extending the dimensions of variables like U_x, U_y, U_z (we did while creating the network).
            example: ["U", "T"]
        '''
        super().__init__()
        self.training_config = training_config
        self.start_time = self.training_config.training_start_time if not start_time else start_time
        self.end_time = self.training_config.training_end_time if not end_time else end_time
        self.vars: list = self.training_config.extend_variables() if not vars_list else vars_list
        self.time_step = self.training_config.time_step if not time_step else time_step   

        self.grid_x = self.training_config.grid_x
        self.grid_y = self.training_config.grid_y
        ############## ------------ Data Integrity Check --------------############

        # First thing first, we must ensure that we have the data from start_time to end_time
        # in the data_path directory.
        self.data_path = self.training_config.data_path if not data_path else Path(data_path)
        assert self.data_path.exists(), f"Data path: {self.data_path} doesn't exist."
        assert self._is_present(), f"Data is missing in the directory: {self.data_path}.\n\
                                    You must have data from {start_time} to {end_time} for variables: {self.vars}" 
        assert self.start_time != 0, "Start time can't be zero. We don't have functionality to implement initial condition in the dataset."     
        ############ ----------------------------------------------############
        # Preprocess inputs and labels:
        self.inputs, self.labels = self._prepare_inputs_and_labels()

    def _is_present(self) -> bool:
        for var in self.vars:
            for time in np.arange(self.start_time, self.end_time+self.time_step, self.time_step):
                if not (self.data_path / f"{var}_{time}.npy").exists():
                    return False
        return True
    
    @staticmethod
    def parse_numpy(training_config:TrainingConfig, data_path:Path) -> np.ndarray:

        grid_x = training_config.grid_x
        grid_y = training_config.grid_y
        data:np.ndarray = np.load(data_path)
        if len(data.shape) == 2: # (40000, 3): when get from OpenFOAM. VECTOR data
            if training_config.data_dim == 1: # 1D
                return data[:,0].reshape(grid_y, grid_x, order="C")
            elif training_config.data_dim == 2: #2D
                assert data.shape[0] == grid_x * grid_y, "check data shape and grid size mentioned in config."
                # Why order="F"? Check: https://github.com/JBNU-NINE/repit_container/blob/main/repit_wiki/Data-Loader-for-FVMN.md
                x_data = data[:,0].reshape(grid_y, grid_x, order="C")
                y_data = data[:,1].reshape(grid_y, grid_x, order="C")
                return np.stack([x_data, y_data], axis=-1)
            elif training_config.data_dim == 3: #3D
                x_data = data[:,0].reshape(grid_y, grid_x, order="C")
                y_data = data[:,1].reshape(grid_y, grid_x, order="C")
                z_data = data[:,2].reshape(grid_y, grid_x, order="C")
                return np.stack([x_data, y_data, z_data], axis=-1)
            else: # Beyond 3D
                raise NotImplementedError("This framework doesn't support beyond 3D.")
        elif len(data.shape) > 2:
            raise NotImplementedError("Till now, we have not come across this use case.")
        else: # SCALAR data
            return data.reshape(grid_y, grid_x, order="C")
        
    def _add_zero_padding(self, data:np.ndarray) -> np.ndarray:
        return np.pad(data, 1, mode="constant", constant_values=0)

    @staticmethod
    def add_feature(input_matrix:np.ndarray) -> np.ndarray:
        window_shape = (3, 3)
        sliding_window = np.lib.stride_tricks.sliding_window_view(input_matrix, window_shape)
        x,y = window_shape[0] // 2, window_shape[1] // 2 
        correlated_features = np.stack([
            sliding_window[:,:,x,y],
            sliding_window[:,:,x-1,y],
            sliding_window[:,:,x+1,y],
            sliding_window[:,:,x,y-1],
            sliding_window[:,:,x,y+1]
        ], axis=-1)
        return correlated_features.reshape(-1, 5)
    
    def _prepare_input(self, time) -> np.ndarray:
        '''
        Regarding the order of the variables in input data, two things matter: 
        1. The list of variables in the config file: "data_vars"
        2. The dimension of the data: 1D, 2D, 3D defined in the config file as "data_dim"
        Example: 
        1. If data_vars = ["U", "T"] and data_dim = 2, then the order of the variables in the input data will be: 
            U_x, U_y, T
        2. If data_vars = ["T", "U"] and data_dim = 3, then the order of the variables in the input data will be:
            T, U_x, U_y, U_z

        Functionality:
        1. Load the numpy files from the data_path directory. [U_0.npy, T_0.npy]
        2. Parse the numpy files.
           a. If the data is VECTOR, split the data into x, y, z components. From this function: we get [200,200,2] shape.
           b. If the data is SCALAR, keep the data as it is. From this function: we get [200,200] shape.
        3. Add zero padding to the data. From this function: we get [202,202] shape.
        4. Add correlated features to the data. From this function: we get [40000,5] shape.
        5. We exclude the boundary cells from the data. From this function: we get [39204,5] shape i.e. (200-2) * (200-2) = 39204
        5. Concatenate the data. From this function: we get [39204,15] shape. if we have 3 variables in the data_vars. 
        '''
        data_path = self.data_path
        full_data_path = [data_path / f"{var}_{time}.npy" for var in self.vars]
        numpy_data = [self.parse_numpy(self.training_config, data_path) for data_path in full_data_path]
        temp = list()
        for data in numpy_data:
            if len(data.shape) > 2:
                for i in range(2):
                    temp.append(data[:,:,i])
            else:
                temp.append(data)
        data = [FVMNDataset.add_feature(data) for data in temp] # In original code: zero padded done but while extracting features, these zero padded regions are not touched. 
        # grid_number_excluding_bc = (self.grid_x-2) * (self.grid_y - 2) # If you do zero padding, uncomment this line.
        return np.concatenate(data, axis=1)

    def _calculate_difference(self, time) -> np.ndarray:
        '''
        If we have data from 0s to 5s and input data should be from 0s to 4s. 
        This function calculates the labels(target data) as difference between two consecutive time steps.
        Example: 
        1 - 0: 1st label
        2 - 1: 2nd label
        3 - 2: 3rd label
        4 - 3: 4th label
        5 - 4: 5th label
        '''
        data_t = self._prepare_input(time)
        data_t_next = self._prepare_input(time + self.time_step)
        return data_t_next[:,::5] - data_t[:,::5]
    
    @staticmethod
    def normalize(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean)/std
        return normalized_data
    
    @staticmethod
    def denormalize(data:Tensor)->Tensor:
        mean_ = mean(data, axis=0)
        std_ = std(data, axis=0)
        return (data * std_) + mean_
    
    def _prepare_inputs_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        inputs, labels = [], []
        for time in np.round(np.arange(self.start_time, self.end_time, self.time_step),2):
            inputs.append(self._prepare_input(time))
            labels.append(self._calculate_difference(time))
        inputs = np.concatenate(inputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        normalized_inputs = self.normalize(inputs)
        normalized_labels = self.normalize(labels)

        #TODO: hardcoded because this is what done in the original paper. Try to find a better way.
        final_input = np.concatenate((normalized_inputs,inputs[:, 0:1], inputs[:, 5:6], inputs[:, 10:11]), axis=1)
        # looking into the original code, it doesn't seem necessary to concatenate the original data with the normalized data.

        return Tensor(normalized_inputs), Tensor(normalized_labels)
        
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
    


if __name__ == "__main__":
    training_config = TrainingConfig()
    data_path = Path("./Assets/natural_convection")
    start_time = 5.0
    end_time = 5.02
    time_step = 0.01
    data = FVMNDataset(training_config,data_path, start_time, end_time, time_step)
    inputs , labels = data._prepare_inputs_and_labels()
    print(inputs.shape, labels.shape, len(data))

    