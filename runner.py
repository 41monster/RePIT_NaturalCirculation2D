from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from dataset import FVMNDataset
from model import FVMNetwork
from config import TrainingConfig
from typing import Tuple, List
import torch
import numpy as np
from pathlib import Path

def get_dataloader(training_config:TrainingConfig, data_path=None, start_time=None, end_time=None,
                   time_step=None, batch_size=None)->Tuple[DataLoader, DataLoader, Dataset]:
    
    data_path = data_path if data_path else training_config.data_path
    start_time = start_time if start_time else training_config.training_start_time
    end_time = end_time if end_time else training_config.training_end_time
    time_step = time_step if time_step else training_config.time_step
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

    def train(self, train_loader:DataLoader, val_loader:DataLoader, epochs) -> bool:
        self.model.train()  # Set the model to training mode
        for epoch in range(epochs):
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
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
            
            # Validation loss
            val_loss = self.validate(val_loader)
            if val_loss < self.best_val_accuracy:
                self.best_val_accuracy = val_loss
                self.save_model("best_model.pth")
            self.losses["val"].append(val_loss)
        
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
        print(f"Validation Loss: {val_loss:.4f}")
        return val_loss
    
    def predict(self, prediction_start_time=None, prediction_end_time=None, data_path:Path=None):
        '''
        prediction_input: To start the prediction we must have the input data. If not provided, it will use the last time step's data.
        prediction_start_time: from which time step to start the prediction. Default is the last time step.
        prediction_end_time: float: end time for prediction
        '''
        start_time = prediction_start_time if prediction_start_time else self.training_config.prediction_start_time
        end_time = prediction_end_time if prediction_end_time else self.training_config.prediction_end_time
        time_step = self.training_config.time_step
        time_range = np.round(np.arange(start_time, end_time, time_step),2)
        data_path = self.training_config.data_path if data_path is None else Path(data_path)

        self.model.eval()
        prediction_input = None
        with torch.no_grad():
            for time in time_range:
                first_prediction = True if time == start_time else False
                prediction_input = self.prepare_input_for_prediction(time, data_path, first_prediction, prediction_input)
                predicted_output = self.model(prediction_input)
                denormed_output = FVMNDataset.denormalize(predicted_output.cpu())
                prediction_input = prediction_input[:, ::5] + denormed_output
    
    def save_model(self, model_name:str):
        torch.save(self.model.state_dict(), model_name)
        print(f"Model saved as {model_name}")
        return model_name

    def load_model(self, path) -> True:
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print(f"Model loaded from {path}")
        return self.model
    
    def get_boundary_data(self, time_step, data_path:Path=None, first_prediction:bool=False) -> List[np.ndarray]:
        '''
        Because in FVMN, we are only predicting the interior points, we need to add the boundary data to the model output.
        So, that we can again extract the features as we did for the training data and normalize it and again give it as input to the model.

        Args: 
        model_output_data: torch.Tensor: The output from the model after denormalizing and adding with the input.
        data_path: Path: if we predict for time step 5.03 then we need the original data for the time step 5.03 to get the boundary data.
                        this is the path to that data.
        time_step: float: the time step for which we are predicting. e.g., 5.03
        first_prediction: bool: If this is the first prediction, we return the whole data. Else, we set the all the other except boundary to zero.

        Returns:
        List[np.ndarray]: List of numpy arrays. Each numpy array is the data for each variable separated dimension wise:
        e.g., [U_x, U_y, T] for each variable.
        '''
        data_path = data_path if data_path else self.training_config.data_path
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
        
        if not first_prediction:
            # Set all the values to zero except the boundary values
            for i in range(len(temp)):
                temp[i][1:-1, 1:-1] = 0
        return temp
        
    def prepare_input_for_prediction(self, time_step, data_path:Path, first_prediction:bool,data:torch.Tensor=None) -> torch.Tensor:
        '''
        Prepare the input for the model for prediction. This will include the boundary data as well.
        Args:
        time_step: The time step for which we are predicting.
        data: torch.Tensor: The output from the model after denormalizing and adding with the input [batch_size, num_features]
        data_path: Path: if we predict for time step 5.03 then we need the original data for the time step 5.03 to get the boundary data.
        first_prediction: bool: If this is the first prediction, we return the whole data. Else, we set the all the other except boundary to zero.
        '''
        # reshape the data to the 2D grid:
        temp = self.get_boundary_data(time_step, data_path, first_prediction)
        variables = self.training_config.get_variables()
        if not first_prediction:
            data = data.numpy()
            data = [data[:, i].reshape(self.training_config.grid_y-2, self.training_config.grid_x-2) for i in range(data.shape[-1])]
            data = [np.pad(data[i], 1, mode="constant",constant_values=0) for i in range(len(data))]
            temp = [np.add(t,d) for t,d in zip(temp, data)]
            for i, var in enumerate(temp):
                np.save(data_path / f"{variables[i]}_{time_step}_predicted.npy", var)
                print(f"Saved {variables[i]}_{time_step}_predicted.npy")
        temp_ = [FVMNDataset.add_feature(data) for data in temp]
        data = np.concatenate(temp_, axis=1)
        return torch.Tensor(FVMNDataset.normalize(data))

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
    trainer.predict(prediction_start_time=5.02, prediction_end_time=5.05, data_path=training_config.data_path)