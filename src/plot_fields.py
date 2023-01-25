""" Plot field plots """

import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import numpy as np
import keras.backend as K
from tensorflow import keras
plt.rcParams.update(
    {
        'font.family': 'sans-serif',
        'font.sans-serif': 'Nimbus Sans',
        'font.size': 32,
    }
)
plt.rc('axes', unicode_minus=False)

from utils.visualization import plot_diffusion_field, plot_drift_field
from utils.modelbuilder import ModelBuilder
from utils.postprocess import get_cov_matrix_per_arg
import tensorflow as tf
import numpy as np

def main(read_model_path = ""):
    n_dimensions = 2 #4
    n_layers = 5
    n_dim_per_layer = 150
    #n_dim_per_layer = 100
    diffusivity_type = "spd"
    ACTIVATIONS = tf.nn.elu
    loaded_model = ModelBuilder.define_gaussian_process(
                                            n_input_dimensions=n_dimensions,
                                            #n_input_dimensions=3,
                                            n_output_dimensions=n_dimensions,
                                            n_layers=n_layers,
                                            n_dim_per_layer=n_dim_per_layer,
                                            name="Euler",
                                            diffusivity_type=diffusivity_type,
                                            activation=ACTIVATIONS)

    loaded_model.load_weights(read_model_path)

    def fx(x, y):
        val = np.empty_like(x)
        for i in range(len(val)):
            val[i] = get_cov_matrix_per_arg(loaded_model, x[i], y[i], n_dim = 2, step_size = 0.04)[0].numpy()[0][0]
            
        return val


    def fy(x, y):
        #print(0.194 * x - 0.484 * x ** 3 - 0.484 * x * y ** 2)
        #print(x)
        val = np.empty_like(y)
        for i in range(len(x)):
            val[i] = get_cov_matrix_per_arg(loaded_model, x[i], y[i], n_dim = 2, step_size = 0.04)[0].numpy()[0][1]
        return val

    def gxx(x, y):

        return get_cov_matrix_per_arg(loaded_model, x, y, n_dim = 2, step_size = 0.04)[1].numpy()[0][0][0]


    def gyy(x, y):

        return get_cov_matrix_per_arg(loaded_model, x, y, n_dim = 2, step_size = 0.04)[1].numpy()[0][1][1]

    def gxy(x, y):
        return get_cov_matrix_per_arg(loaded_model, x, y, n_dim = 2, step_size = 0.04)[1].numpy()[0][0][1]


    fig, ax = plt.subplots(figsize=(8, 8))
    plot_drift_field(ax, fx, fy)
    plt.tight_layout()
    plt.savefig("drift.pdf")

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_diffusion_field(ax, gxx, gyy, gxy, scale=400)
    #plot_drift_field(ax, fx, fy)
    plt.tight_layout()
    #plt.savefig("diffusion.pdf", facecolor='w') 
    plt.savefig("diffusion.pdf") 

if __name__ == '__main__':
    main(read_model_path = "path_to_model_dir")


