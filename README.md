# Neu_sde
Code to paper: Discovering mesoscopic descriptions of collective animal movement with neural stochastic modelling


![Pipeline](fig/pipeline.png)

# How to run:
1. Install the dependencies mentioned in the requirements(requirements.txt) file

2. To train :
    - Enter path of the training data in train.py
    - Enter path of where to save model weights in train.py
    - Change hyperparameters for training in train.py (as per need)
    - Run python train.py

3. To visualize the field plots from learnt Neural model:
                                *a) Enter path of model weights plot_field.py
                                *b) change parameters for visualization in plot_field.py (as per need)
                                *c) Run python plot_field.py

4. Extra utilites:
        *a) Data augmentation: Use augment.py to augment the data as per discussed in the paper (You will need to train on this new data)
        *b) Sample path: Use sample_path.py to sample a path from learnt neural model



# Data:
Can be made available on request

# ACKNOWLEGDEMENT
I used: [Dietrich et al.](https://gitlab.com/felix.dietrich/sde-identification/-/tree/master/) as a reference for my work
