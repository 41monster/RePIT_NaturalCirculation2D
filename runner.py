from typing import Tuple, List
from pathlib import Path
from copy import deepcopy
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ResidualNaturalConvection import residual_mass, residual_momentum, residual_heat
from dataset import FVMNDataset
from config import TrainingConfig
from model import FVMNetwork
from plot_utils import visualize_output

def get_dataloader(training_config:TrainingConfig, data_path=None, start_time=None, end_time=None,
                   time_step=None, batch_size=None)->Tuple[DataLoader, DataLoader, Dataset]:
    
    data_path = data_path if data_path else training_config.assets_path
    start_time = start_time if start_time else training_config.training_start_time
    end_time = end_time if end_time else training_config.training_end_time
    time_step = time_step if time_step else training_config.write_interval
    batch_size = batch_size if batch_size else training_config.batch_size
    # Create dataset instance
    dataset = FVMNDataset(training_config=training_config, data_path=data_path, start_time=start_time, end_time=end_time, time_step=time_step)

    # Indices for splitting
    data_size = len(dataset)
    indices = list(range(data_size))
    #TODO: hardcoded get the dataset from the last time step
    # grid_calculation = (training_config.grid_x-2) * (training_config.grid_y-2) # Because after feature extraction, we will have the total grid points from each boundary. 
    # prediction_indices = indices[-grid_calculation:]
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    # Subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    # prediction_dataset = Subset(dataset, prediction_indices)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Because we need the last time step's true values to give as input to the model, we also return the dataset for the last time step

    return train_loader, val_loader

class Trainer:
    def __init__(self, training_config:TrainingConfig, model:torch.nn.Module, optimizer:torch.optim.Adam, 
                 loss_fn:torch.nn.MSELoss, model_path:Path=None):
        self.training_config = training_config
        self.device = training_config.device
        self.model = model.to(self.device)
        self.model = self.model if model_path is None else self.load_model(model_path)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.losses = {"train": [], "val": []}
        self.best_val_accuracy = 1

        # Residuals
        self.residual_mass = list()
        self.residual_momentum = list()
        self.residual_heat = list()

        self.ux_matrix = torch.zeros((training_config.grid_y, training_config.grid_x))
        self.uy_matrix = torch.zeros((training_config.grid_y, training_config.grid_x))
        self.t_matrix = torch.zeros((training_config.grid_y, training_config.grid_x))
        self.t_matrix_prev = torch.zeros((training_config.grid_y, training_config.grid_x))
        self.ux_matrix_prev = torch.zeros((training_config.grid_y, training_config.grid_x))

        self.variables = self.training_config.get_variables()
        self.ux_index = self.variables.index("U_x")
        self.uy_index = self.variables.index("U_y")
        self.t_index = self.variables.index("T")

    def train(self, train_loader:DataLoader, val_loader:DataLoader, epochs) -> bool:
        self.model.train()  # Set the model to training mode
        for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
            epoch_loss = list()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                predictions = self.model(x)
                loss = self.loss_fn(predictions, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.item())
            
            epoch_loss = np.mean(epoch_loss)
            
            self.losses["train"].append(epoch_loss)
            self.training_config.logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
            
            # Validation loss
            val_loss = self.validate(val_loader)
            if val_loss < self.best_val_accuracy:
                self.best_val_accuracy = val_loss
                self.save_model("best_model.pth")
            self.losses["val"].append(val_loss)

        # Save the losses
        with open(self.training_config.root_dir / "losses.json", "w") as f:
            json.dump(self.losses, f)
        
        return True

    def validate(self, val_loader):
        self.model.eval()  # Set the model to evaluation mode
        val_loss = list()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.model(x)
                val_loss.append(self.loss_fn(predictions, y).item())

        val_loss = np.mean(val_loss)
        self.training_config.logger.info(f"Validation Loss: {val_loss:.4f}")
        return val_loss
    
    def predict(self, prediction_start_time=None, prediction_end_time=None, data_path:Path=None):
        '''
        prediction_input: To start the prediction we must have the input data. If not provided, it will use the last time step's data.
        prediction_start_time: from which time step to start the prediction. Default is the last time step.
        prediction_end_time: float: end time for prediction
        '''
        start_time = prediction_start_time if prediction_start_time else self.training_config.prediction_start_time
        end_time = prediction_end_time if prediction_end_time else self.training_config.prediction_end_time
        time_step = self.training_config.write_interval
        time_range = np.round(np.arange(start_time, end_time, time_step),training_config.round_to)
        data_path = self.training_config.assets_path if data_path is None else Path(data_path)

        self.model.eval()
        prediction_input = None
        with torch.no_grad():
            for time in time_range:
                first_prediction = True if time == start_time else False
                prediction_input = self.prepare_input_for_prediction(time, data_path, first_prediction, prediction_input)
                normalized_input,mean, std  = FVMNDataset.normalize(prediction_input)
                predicted_output:torch.Tensor = self.model(normalized_input.to(self.device))
                denormed_output = FVMNDataset.denormalize(predicted_output.cpu(), mean, std)
                prediction_input = prediction_input[:, ::5] + denormed_output
    
    def save_model(self, model_name:str):
        torch.save(self.model.state_dict(), model_name)
        self.training_config.logger.info(f"Model saved as {model_name}")
        return model_name

    def load_model(self, path) -> True:
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.training_config.logger.info(f"Model loaded from {path}")
        return self.model
    
    def get_ground_truth_data(self, time_step:int|float, data_path:Path=None, first_prediction:bool=False) -> List[np.ndarray]:
        '''
        Because in FVMN, we are only predicting the interior points, we need to add the boundary data to the model output.
        Also, we need to calculate the residue. Hence, we need the true data for the time step.

        Args
        ---- 
        data_path: Path: 
            If we predict for time step 5.03 then we need the original data for the 
            time step 5.03 to get the boundary data.This is the path to that data.
        time_step: float: 
            The time step for which we are predicting. e.g., 5.03
        first_prediction: bool: # TODO: This is not used. Remove it.
            If this is the first prediction, we return the whole data. 
            Else, we set the all the other except boundary to zero.

        Returns
        -------
        Each numpy array is the data for each variable separated dimension wise:
        e.g., [U_x, U_y, T] for each variable.
        
        ground_truth_data: List[np.ndarray]:
            Along with boundary values, we send the true values also.

        Functionality
        -------------
        1. Get the boundary data for the time step from ground truth data.
        2. Parse the numpy data for the variables.
        3. Separate the dimensions of the data if present. 
        5. Because to calculate the residue, we need true values also. 
           Hence, this method returns true values also. 
        '''
        data_path = data_path if data_path else self.training_config.assets_path
        variables = self.training_config.extend_variables()
        full_data_path = [data_path / f"{var}_{time_step}.npy" for var in variables]
        numpy_data = [FVMNDataset.parse_numpy(self.training_config, data_path) for data_path in full_data_path]
        temp = list()
        for data in numpy_data:
            if len(data.shape) > 2:
                for i in range(2):
                    temp.append(data[:,:, i])
            else:
                temp.append(data)
        
        self.training_config.logger.info(f"Loaded ground truth data for time step {time_step}")
        return temp
        
    def prepare_input_for_prediction(self, time_step:int|float, data_path:Path, first_prediction:bool,data:torch.Tensor=None) -> torch.Tensor:
        '''
        Prepare the input for the model for prediction. This will include the boundary data as well.

        Args
        ----
        time_step: int|float:
            If we are predicting for t then time_step = t-dt.
        data: torch.Tensor: 
            The output from the model after denormalizing and adding with the input [batch_size, num_features]
        data_path: Path: 
            if we predict for time step 5.03 then we need the original data for the time step 5.03 to get the boundary data.
        first_prediction: bool: 
            If this is the first prediction, we return the whole data. Else, we set the all the other except boundary to zero.

        Functionality
        -------------
        1. Get the boundary data for the time step from ground truth data.
        2. If this is not the first prediction, do the zero padding on the boundary of predicted data [198,198] -> [200,200].
        3. If not first prediction, internal nodes in the true data are set to zero.
        4. Because, we are using the same training_config.get_variables() to get the variables.
           We leverage this to get the index of U_x, U_y, T.
        5. If it is not the first prediction, we are setting U_x and T values in that iteration as previous values 
           and as the process progresses, we update the previous values with the predicted values.
        6. We save the predicted values here. In the prediction loop, we get the output for time(running_time) + dt.
           So, it makes sense that we can update the running time, and while preparing input for the next prediction, 
           we can add boundary values to the prev. predicted values and that would represent the predicted values for 
           currently running_time in prediction loop.

        Reasoning
        ---------
        But why did we assign the present/previous ux_matrix, uy_matrix, t_matrix in this function? 
        Because, we would have input and output data both in the predict method. Wouldn't it make sense to assign the values there?

        Sadly NO.
        Because, the input for the network is feature extracted. Example shape: [40000,15]
        And the output from the network is boundary excluded data. Example shape: [39204,3]
        Hence, we must do the post-processing before calculating the residue. So, for me, 
        it made a lot of sense to assign the values here. If you have a better idea, please let me know.
        '''
        ground_truth = self.get_ground_truth_data(time_step, data_path, first_prediction)
        temp = deepcopy(ground_truth)

        if not first_prediction:

            # Modelling predicted data: adding zero padding to the predicted data.
            data = data.numpy()
            predicted_data_grid_x = self.training_config.grid_x - 2
            predicted_data_grid_y = self.training_config.grid_y - 2
            assert data.shape[0] == predicted_data_grid_x * predicted_data_grid_y, f"Shape of the data is {data.shape} but should be {(predicted_data_grid_x * predicted_data_grid_y, data.shape[-1])}"
            data = [data[:, i].reshape(self.training_config.grid_y-2, self.training_config.grid_x-2) for i in range(data.shape[-1])]

            # Copying just the boundary values from the ground truth data:
            for i in range(len(temp)): temp[i][1:-1, 1:-1] = 0 # Setting the internal nodes to zero.
            data = [np.pad(data[i], 1, mode="constant",constant_values=0) for i in range(len(data))]

            # Adding the zero padded predicted data to the zeroed internal nodes in the ground truth data.
            temp = [np.add(t,d) for t,d in zip(temp, data)]

            # Now, we completely have the predicted data in the temp variable.
            self.ux_matrix = temp[self.ux_index]
            self.uy_matrix = temp[self.uy_index]
            self.t_matrix = temp[self.t_index]

            ux_matrix_true = ground_truth[self.ux_index]
            uy_matrix_true = ground_truth[self.uy_index]
            # for i, var in enumerate(temp):

            #     # TODO: just to test the framework, we are hardcoding it for now. To concatenate vectors into a single matrix. 
            #     # While saving it to the numpy file, we have to concatenate the vectors into a single matrix, because it will be harder to 
            #     # change to foam format then. 
            #     np.save(data_path / f"{self.variables[i]}_{time_step}_predicted.npy", var)
            #     self.training_config.logger.info(f"Saved {self.variables[i]}_{time_step}_predicted.npy")

            ##################### Saving the predicted values #####################
            u_vector = np.concatenate([self.ux_matrix.reshape(-1,1),
                                       self.uy_matrix.reshape(-1,1)], axis=1)
            t_scalar = self.t_matrix.reshape(-1,1)
            self.training_config.logger.info(f"Saving the predicted values for time step {time_step}")
            np.save(data_path / f"U_{time_step}_predicted.npy", u_vector)
            np.save(data_path / f"T_{time_step}_predicted.npy", t_scalar)
            self.training_config.logger.info(f"Saved variables at {data_path}")
            ##################### Saved the predicted values #####################

             # Calculate the residue
            predicted_residual_mass = residual_mass(ux_matrix=self.ux_matrix, uy_matrix=self.uy_matrix)
            true_residual_mass = residual_mass(ux_matrix=ux_matrix_true, uy_matrix=uy_matrix_true)
            self.relative_residual_mass = predicted_residual_mass / true_residual_mass

            self.residual_mass.append(predicted_residual_mass)
            self.residual_momentum.append(residual_momentum(ux_matrix=self.ux_matrix, ux_matrix_prev=self.ux_matrix_prev, 
                                                            uy_matrix=self.uy_matrix, t_matrix=self.t_matrix))
            self.residual_heat.append(residual_heat(ux_matrix=self.ux_matrix, uy_matrix=self.uy_matrix,
                                                    t_matrix=self.t_matrix, t_matrix_prev=self.t_matrix_prev))
            
            # Update the previous values
            self.ux_matrix_prev = self.ux_matrix
            self.t_matrix_prev = self.t_matrix
        else: 
            self.ux_matrix_prev = temp[self.ux_index]
            self.t_matrix_prev = temp[self.t_index]
        
        temp_ = [FVMNDataset.add_feature(data) for data in temp]
        data = np.concatenate(temp_, axis=1)

        return torch.Tensor(data)

if __name__ == "__main__":
    training_config = TrainingConfig()
    # Get dataloaders
    train_loader, val_loader = get_dataloader(training_config)

    # Create model instance
    model = FVMNetwork(training_config)
    
    # Create trainer instance
    trainer = Trainer(training_config=training_config, model=model, 
                      optimizer=training_config.optimizer(model.parameters(), lr=training_config.learning_rate),
                      loss_fn=training_config.loss)
    
    # Train the model
    trainer.train(train_loader, val_loader, training_config.epochs)

    # Save the model
    trainer.save_model("last_model.pth")

    # Predict
    print("Predicting...")
    trainer.predict()
