from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from config import BaseConfig
import warnings
import imageio

pwd = Path(__file__).parent.resolve() # get the current working directory
base_config = BaseConfig()
data_dim = base_config.data_dim
nx = base_config.grid_x
ny = base_config.grid_y

def visualize_output(np_data_dir:Path, 
					timestamp:int|float, 
					data_vars:list=["U", "T"], 
					save_path:Path=None,
					mode:str="rgb_array"):
	'''
	This function is used to visualize the output of the simulation.

	Args:
	np_data_dir: Path
		The path to the directory where the output numpy files are stored. 
		1. files should be in the format: U_{timestamp}.npy, T_{timestamp}.npy, etc.
		2. It must be a pathlib.Path object.
		3. The last directory of the path should be case name for the framework to work properly. e.g. 
			/home/openfoam/repitframework/repitframework/Assets/natural_convection
	timestamp: int or float
		The time at which the output is to be visualized.
	data_vars: list
		The list of variables to be visualized. Default is ["U", "T"]
	save_path: Path
		The path to the directory where the output images are to be saved.
	mode: str
		1. "rgb_array": It will return the RGB array of the output.
		2. "image": It will save the image of the output in the specified directory. 

	Returns:
	1. If mode is "rgb_array": it will return the list (timestamps) of RGB arrays -> list[rgb_arrays]
	2. If mode is "image": it will  just save the images in the specified directory -> list[image_files]

	Functionality: 
	1. numpy files are loaded in C-order i.e row-major order because OpenFOAM stores the data in row-major order.
	2. np.flipuid is used to flip the rows from top to bottom because numpy writes first data in first row which is different from OpenFOAM.
	'''
	# Ensure the np_data_path is a pathlib.Path object
	np_data_dir = Path(np_data_dir) if not isinstance(np_data_dir, Path) else np_data_dir
	save_path = pwd / "plots" / np_data_dir.name if save_path is None else save_path
	save_path.mkdir(exist_ok=True) if not save_path.exists() else None

	data_dict:np.ndarray = {} 
	'''
	Remember the format of the data_dict is:
	{"T": np.ndarray, "U_x": np.ndarray, "U_y": np.ndarray}
	'''
	for var in data_vars:
		data_dict[var] = np.load(np_data_dir / f"{var}_{timestamp}.npy")
		if len(data_dict[var].shape) == 2:
			if data_dict[var].shape[-1] != data_dim: warnings.warn(f"Data dimension mismatch. Expected {data_dim} but got {data_dict[var].shape[-1]}. PLEASE CHECK!")
			for i in range(data_dim):
				axis_list = ["x","y","z"]
				data_dict[f"{var}_{axis_list[i]}"] = np.flipud(data_dict[var][:,i].reshape(ny,nx,order="C"))
			del data_dict[var]
		elif len(data_dict[var].shape) > 2:	
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

def make_animation(np_data_dir:Path, 
					timestamps:list[int|float], 
					data_vars:list=["U", "T"], 
					save_name:Path=None,
					set_fps:int=1)->bool:
	'''
	This function is used to make an animation of the output of the simulation.
	'''
	# Ensure the np_data_path is a pathlib.Path object
	np_data_dir = Path(np_data_dir) if not isinstance(np_data_dir, Path) else np_data_dir
	save_dir = pwd / "plots" / np_data_dir.name 
	save_dir.mkdir(exist_ok=True, parents=True) if not save_dir.exists() else None
	save_path = Path.joinpath(save_dir, "output.gif") if not save_name else Path.joinpath(save_dir,save_name)

	images_list = []
	for timestamp in timestamps:
		images_list.append(visualize_output(np_data_dir, timestamp, data_vars, mode="rgb_array"))
	imageio.mimsave(save_path, images_list, fps=set_fps)
	return True

if __name__ == "__main__":
	make_animation(np_data_dir=pwd / "Assets/natural_convection", timestamps=[i for i in np.arange(5.0,5.02+0.01,0.01)], 
				data_vars=["U", "T"], set_fps=1, save_name="output_predicted.gif")
