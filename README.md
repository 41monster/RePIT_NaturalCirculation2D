# Finite Volume Method Network
This network architecture is coined by J. Jeon et al. You can refer to the [link](https://onlinelibrary.wiley.com/doi/full/10.1002/er.7879) to read the full paper. 

# Description
**Assets**: This is to store the training and predicted data results. For now it just contains dummy input for the training of the architecture. <br>
**dataset**: This script is responsible for creating the dataset to enable the training process. It is where feature extraction is implemented. <br>
**model**: This script contains the basic architecture of the FVMN.<br>
**plot_utils**: If you want to visualize or animate your plots, you can utilize this script. <br>
**runner**: This is the main script, which makes the dataloader, trains, predicts and inserts boundary values to the predicted output.

# Before running the code
The codes in this repo are tailored to train the FVMN with Natural Circulation 2D data. For this case, the domain is a square with 200 grid points in the x and y directions. The grid step is 0.05 and the time step is 0.01. We are just predicting the values of velocities(ux,uy) and temperature from the neural network. And, all these things are set in the **config.py** file. If you have different setup, it is recommended to change the parameters in this file to match up with your use case. 

For example: 
If you have a three dimensional data; change **data_dim** to 3. 
If you want to predict just pressure and temperature; change **data_vars** as **{"scalars":["p","T"]}**

## Dataset requirement
Here, we are training for timestamps 10 - 10.03 seconds for U and T fields. So, in the assets directory we must have data named as U_10.0.npy and T_10.0.npy and so on for all the timestamps until when we want predict. 

We require the whole dataset even if we are predicting after 10.03 timestep because, we are just predicting the internal node from the neural network and attaching the true boundary values from the ground truth. 
For example: 
If we have [200,200] grid of data, we predict internal node [198,198] grid values from the neural network and attach true boundary data to this grid and make it again a [200,200] grid which is input to the next time step then. 

# Run the code:
I hope you have anaconda installed in your machine, if not please visit this [link](https://docs.anaconda.com/miniconda/install/) and download it in your machine.
After you have installed it, you can install the dependencies using: 
```
conda env create -f environment.yml
conda activate fvmn_env
python runner.py (be sure to be in the RePIT_NaturalCirculation2D directory)
```