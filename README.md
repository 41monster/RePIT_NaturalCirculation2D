# Finite Volume Method Network
This network architecture is coined by J. Jeon et al. You can refer to the [link](https://onlinelibrary.wiley.com/doi/full/10.1002/er.7879) to read the full paper. 

# Description
**Assets**: This is to store the training and predicted data results. For now it just contains dummy input for the training of the architecture. <br>
**dataset**: This script is responsible for creating the dataset to enable the training process. It is where feature extraction is implemented. <br>
**model**: This script contains the basic architecture of the FVMN.
**plot_utils**: If you want to visualize or animate your plots, you can utilize this script. <br>
**runner**: This is the main script, which makes the dataloader, trains, predicts and inserts boundary values to the predicted output.

# Run the code:
I hope you have anaconda installed in your machine, if not please visit this [link](https://docs.anaconda.com/miniconda/install/) and download it in your machine.
After you have installed it, you can install the dependencies using: 
```
conda env create -f environment.yml
conda activate fvmn_env
python runner.py (be sure to be in the FVMN directory)
```