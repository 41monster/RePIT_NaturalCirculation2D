from dataclasses import dataclass
from pathlib import Path
import torch.cuda as cuda
import torch
import logging
from datetime import datetime
@dataclass
class BaseConfig:
	# Data parameters
	root_dir: Path = Path(__file__).resolve().parents[0]
	assets_path: Path = Path.joinpath(root_dir,"Assets/natural_convection")
	write_interval:int = 0.01
	logs_dir: Path = Path.joinpath(root_dir, "Logs")
	
	data_dim:int = 2 # It is to denote either data is 1D or 2D or 3D.
	grid_x: int = 200 # Number of grid points in x-direction
	grid_y: int = 200 # Number of grid points in y-direction
	grid_z: int = 1 # Number of grid points in z-direction
	grid_step: float = 0.05 # Grid step size
	round_to: int = len(str(write_interval).split(".")[-1]) # Number of decimal places to round to
	data_vars = {"vectors":["U"],"scalars":["T"]}

	# Logging Level
	logger_level = logging.DEBUG

	def get_variables(self, vars_dict:dict=None):
		"""
		Because we have a dictionary that separates variables into scalars and vectors,
		we need to combine them into a single list. We need to further differentiate vectors 
		based on the number of dimensions (1D or 2D or 3D).

		Args
		----
		vars_dict: dict: 
			The dictionary containing the variables separated into vectors and scalars.

		Returns
		-------
		list: 
			The list of variables: ["U_x", "U_y", "T"]
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
		There are cases where we don't exactly need dimension separated variables, 
		just the variables themselves just like how they are represented in OpenFOAM. 
		Hence, this method is to return the variables as they are, as a list.

		Args
		----
		vars_dict: dict:
			The dictionary containing the variables separated into vectors and scalars.
		
		Returns
		-------
		list: 
			The list of variables: ["U", "T"]
		'''
		vars_dict = self.data_vars if vars_dict is None else vars_dict
		return [var for _, value in vars_dict.items() for var in value]
	
	def setup_logger(self, name:str,log_file: Path) -> logging.Logger:
		"""
		Sets up and returns a logger instance.

		Args
		----
		name: str: 
			The name of the logger.
		log_file: Path: 
			The file path where logs will be saved.

		Returns
		-------
		logging.Logger: 
			Configured logger.
		"""
		level = self.logger_level
		logger = logging.getLogger(name)

		today_date = datetime.now().strftime("%Y-%m-%d")
		today_date_dir = Path(self.logs_dir, today_date)
		today_date_dir.mkdir(parents=True, exist_ok=True)
		log_file = Path(today_date_dir, log_file)
		# Prevent adding multiple handlers if the logger is already configured
		if not logger.handlers:
			logger.setLevel(level)
			formatter = logging.Formatter('%(asctime)s:%(pathname)s:%(levelname)s:%(message)s', datefmt='%H:%M:%S')

			file_handler = logging.FileHandler(log_file)
			file_handler.setFormatter(formatter)

			logger.addHandler(file_handler)

		return logger
	
	def __post_init__(self):
		log_file: Path = Path("BaseConfig.log")
		self.logger = self.setup_logger("BaseLogger",log_file)
class TrainingConfig(BaseConfig):
	def __init__(self):
		super().__init__()
		self.log_file: Path = Path("Training.log")
		self.logger = self.setup_logger("TrainingLogger", self.log_file)
		self.batch_size: int = 10000
		self.epochs: int = 5000
		self.learning_rate: float = 0.001
		self.device: str = "cuda" if cuda.is_available() else "cpu"
		self.optimizer = torch.optim.Adam
		self.loss = torch.nn.MSELoss()
		self.activation = torch.nn.ReLU
		self.model_path = Path("./model.pth")

		self.training_start_time = 10.0
		self.training_end_time = 10.03
		self.prediction_start_time = 10.03
		self.prediction_end_time = 10.05
	
if __name__ == "__main__":
	training_config = TrainingConfig()
	print(training_config.get_variables())
	print(training_config.extend_variables())