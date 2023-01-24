import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils.modelbuilder import ModelBuilder
from utils.approxSDE import SDEApproximationNetwork
from utils.train_utils import SDEIdentification
import pandas as pd

def main():
    random_seed = 2
    n_layers = 5
    n_dim_per_layer = 150
    n_dimensions = 2
    LEARNING_RATE = 1e-3
    ACTIVATIONS = tf.nn.elu
    VALIDATION_SPLIT = .1
    #BATCH_SIZE = 512
    BATCH_SIZE = 512
    #N_EPOCHS = 300 --> for expermimental data?
    N_EPOCHS = 150
    #N_EPOCHS = 50


    # only diagonal, but we are in 1D so it does not matter anyway
    #diffusivity_type = "diagonal"
    diffusivity_type = "spd"
    #diffusivity_type = "spd_general"

    tf.random.set_seed(random_seed)

    tf.keras.backend.set_floatx('float64')
    NUMBER_TYPE = tf.float64  # or tf.float32

    STD_MIN_VALUE = 1e-13  # the minimal number that the diffusivity models can have

    #data = pd.read_csv("/home/ece/utkarsh/fish_traj_extract/PyDaddy/pydaddy/data/model_data/vector/augmented_pairwise.csv", sep = " ", header = None)
    #data = pd.read_csv("/home/ece/utkarsh/fish_traj_extract/on_pydaddy_data/sim_vec_ternary/arsh_new_data/n30_ter.csv", sep ="\t", header=None)
    #data = data.values[:,:2]
    #x_data = data.values[:-1]
    #y_data = data.values[1 :]

    dt = 0.04
    #dt = 0.12

    data = np.load("/home/ece/utkarsh/fish_traj_extract/on_pydaddy_data/sim_vec_ternary/arsh_new_data/augmented/augmented.npy")
    x_data = data[:-1]
    y_data = data[1 :]

    step_sizes = np.zeros((x_data.shape[0],)) + dt


    encoder_euler = ModelBuilder.define_gaussian_process(
                                            n_input_dimensions=n_dimensions,
                                            n_output_dimensions=n_dimensions,
                                            n_layers=n_layers,
                                            n_dim_per_layer=n_dim_per_layer,
                                            name="Euler",
                                            diffusivity_type=diffusivity_type,
                                            activation=ACTIVATIONS)

    model_euler = SDEApproximationNetwork(sde_model=encoder_euler, method="euler", diffusivity_type = 'spd')
    model_euler.compile(optimizer=tf.keras.optimizers.Adamax())

    sde_i_euler = SDEIdentification(model=model_euler)

    hist_euler = sde_i_euler.train_model(x_data, y_data, step_size=step_sizes,
                                        validation_split=VALIDATION_SPLIT, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)

    print(f"Training loss final, Euler: {hist_euler.history['loss']}")
    print(f"Validation loss final, Euler: {hist_euler.history['val_loss']}")

    encoder_euler.save("/home/ece/utkarsh/fish_traj_extract/on_pydaddy_data/sim_vec_ternary/arsh_new_data/augmented")

if __name__ == "__main__":
    main()
