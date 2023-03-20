# Neu_sde
Code to paper: Discovering mesoscopic descriptions of collective animal movement with neural stochastic modelling


![Pipeline](fig/pipeline.png)

# How to run:
1. Install the dependencies mentioned in the requirements(requirements.txt) file

2. To train :
    - Enter path of the training data in train.py
    - Enter path of where to save model weights in train.py
    - Change hyperparameters for training in train.py (as per need).
    - Run python train.py

3. To visualize the field plots from trained Neural model:
    - Enter path of model weights in plot_field.py
    - Change parameters for visualization in plot_field.py (as per need)
    - Run python plot_field.py

4. Extra utilites:
    - Data augmentation: Use augment.py to augment the data as per discussed in the paper (You will need to train on this new data)
    - Sample path: Use sample_path.py to sample a path from learnt neural model.
    - analysis(directory): This folder contains notebooks for:
        - Goodness-of-fit analysis (Wasserstein metric and relative timescale discrepancy)
        - Generation of drift and diffusion plots for theoretically derived mesoscale SDEs (Appendix A)
        - Analysis of autocorrelation of mx and my components of the polarization.
        - This requires the following packages to be installed: sdeint, pydaddy (can use pip to install)
 



# Data:
analysis(directory)

# ACKNOWLEGDEMENT
We used: [Dietrich et al.](https://arxiv.org/abs/2106.09004) as a reference for our work
