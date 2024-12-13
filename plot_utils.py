from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from config import BaseConfig
import warnings
import imageio
from typing import Dict

def visualize_output(base_config:BaseConfig,
					timestamp:int|float,
					np_data_dir:Path = None, 
					data_vars:list=None, 
					save_path:Path=None,
					mode:str="image",
					is_ground_truth:bool=True):
	'''
	This function is used to visualize the output of the simulation.

	Args
	----
	base_config: 
		Base configuration object, so that we can give less arguments to the function.
	timestamp: int or float
		The time at which the output is to be visualized.
	np_data_dir: Path
		The path to the directory where the output numpy files are stored. 
		1. files should be in the format: U_{timestamp}.npy, T_{timestamp}.npy, etc.
		2. It must be a pathlib.Path object.
		3. The last directory of the path should be case name for the framework to work properly. e.g. 
			/home/openfoam/repitframework/repitframework/Assets/natural_convection
	data_vars: list
		The list of variables to be visualized. Default is ["U", "T"]
	save_path: Path
		The path to the directory where the output images are to be saved.
	mode: str
		1. "rgb_array": It will return the RGB array of the output.
		2. "image": It will save the image of the output in the specified directory. 
	is_ground_truth: bool
		1. If True, it will load the ground truth data.
		2. If False, it will load the predicted data.

	Returns
	-------
	1. If mode is "rgb_array": it will return the list (timestamps) of RGB arrays -> list[rgb_arrays]
	2. If mode is "image": it will  just save the images in the specified directory -> list[image_files]

	Functionality
	-------------
	1. numpy files are loaded in C-order i.e row-major order because OpenFOAM stores the data in row-major order.
	2. np.flipuid is used to flip the rows from top to bottom because numpy writes first data in first row which is different from OpenFOAM.
	'''
	np_data_dir = Path(np_data_dir) if np_data_dir else base_config.assets_path
	data_vars = data_vars if data_vars else base_config.extend_variables()
	save_path = base_config.root_dir / "plots" / np_data_dir.name if not save_path else save_path
	save_path.mkdir(exist_ok=True, parents=True)
	data_dim = base_config.data_dim
	ny = base_config.grid_y
	nx = base_config.grid_x

	data_dict:Dict[str, np.ndarray] = {} 
	'''
	Remember the format of the data_dict is:
	{"T": np.ndarray, "U_x": np.ndarray, "U_y": np.ndarray}
	'''
	for var in data_vars:
		numpy_file_name = f"{var}_{timestamp}.npy" if is_ground_truth else f"{var}_{timestamp}_predicted.npy"
		data_dict[var] = np.load(np_data_dir / numpy_file_name)
		shape_of_variable = data_dict[var].shape
		if len(shape_of_variable) == 2:
			if shape_of_variable[-1] != data_dim: warnings.warn(f"Data dimension mismatch. Expected {data_dim} but got {data_dict[var].shape[-1]}. PLEASE CHECK!")
			for i in range(shape_of_variable[-1]): # Be sure to change the data_dim parameter if you have dimension other than 2: even if it is 3D if data_dim is 2, it will visualize for x and y only.
				axis_list = ["x","y","z"]
				data_dict[f"{var}_{axis_list[i]}"] = np.flipud(data_dict[var][:,i].reshape(ny,nx,order="C"))
			del data_dict[var]
		elif len(shape_of_variable) > 2:	
			raise ValueError(f"Data dimension mismatch. Expected 1 or 2 but got {data_dict[var].shape}")
		else:
			data_dict[var] = np.flipud(data_dict[var].reshape(ny,nx,order="C"))
	
	num_subplots = len(data_dict)
	fig, ax = plt.subplots(1, num_subplots, figsize=(5*num_subplots, 5))
	for i,(key,value) in enumerate(data_dict.items()):
		value = ax[i].imshow(value, cmap="coolwarm")
		fig.colorbar(value, ax=ax[i])
		ax[i].set_title(key)
	fig.tight_layout()
	fig.suptitle("At time={}s".format(timestamp))
	if mode == "image":
		plt.savefig(save_path / f"output_{timestamp}.png")
		plt.close()
		return True
	elif mode == "rgb_array": # Convert plot to image
		fig.canvas.draw()
		rgb_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
		plt.close()
		return rgb_array
	else:
		raise ValueError("Invalid mode. Must be either 'image' or 'rgb_array'.")

def make_animation(base_config:BaseConfig, 
					timestamps:list[int|float],
					is_ground_truth:bool,
					np_data_dir:Path=None, 
					data_vars:list=None, 
					save_name:Path=None,
					set_fps:int=1)->bool:
	'''
	This function is used to make an animation of the output of the simulation.
	'''
	np_data_dir = Path(np_data_dir) if np_data_dir else base_config.assets_path
	save_dir = base_config.root_dir / "plots" / np_data_dir.name
	save_dir.mkdir(exist_ok=True, parents=True)
	save_path = save_dir / f"{save_name}.gif" if save_name else save_dir / "output.gif"

	images_list = []
	for timestamp in timestamps:
		images_list.append(visualize_output(base_config=base_config,
											timestamp=timestamp,
											np_data_dir=np_data_dir,
											data_vars=data_vars,
											mode="rgb_array",
											is_ground_truth=is_ground_truth))
	imageio.mimsave(save_path, images_list, fps=set_fps)
	return True

if __name__ == "__main__":

	base_config = BaseConfig()
	# time_list = [10.0,10.01,10.02,10.03]
	# make_animation(base_config=base_config,
	# 				timestamps=time_list,
	# 				is_ground_truth=True)
	visualize_output(base_config=base_config,
					timestamp=10.03,
					is_ground_truth=True)
	make_animation(base_config=base_config,
					timestamps=[10.0,10.01,10.02,10.03],
					is_ground_truth=True,
					save_name="natural_convection")
