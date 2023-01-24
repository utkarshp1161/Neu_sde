
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import numpy as np
import keras.backend as K
from tensorflow import keras
import tensorflow as tf





def get_cov_matrix_per_arg(model, x, y, n_dim = 2, step_size = 0.04):
    """
    
    Args:
        model --->  neural network
        x -> x component
        y -> y component
        ndim -> 2 for 2 dim data
        step_size -> delta_t in euler mauriyama

    
    """
    mean_network, std_network = model(np.array([[x,y]]))
    #mean_network = keras.backend.eval(mean_network)# shape out n 
    std_network = keras.backend.eval(std_network) # shape out n*n
    diffusivity_spd_ = std_network
    step_sizes = np.zeros((1,)) + step_size
    spd_step_size = tf.reshape(0.04, (-1, 1)) ##imp line
    spd_step_size = tf.math.sqrt(spd_step_size) # NO square root because we use cholesky below?
    full_shape = n_dim * n_dim
    step_size_matrix = tf.broadcast_to(spd_step_size, [K.shape(step_sizes)[0], full_shape])
    step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

    # multiply with the step size
    covariance_matrix = tf.multiply(step_size_matrix , diffusivity_spd_)
    # square the matrix so that the cholesky decomposition does not change the eienvalues
    covariance_matrix = tf.linalg.matmul(covariance_matrix, tf.linalg.matrix_transpose(covariance_matrix))
    # perform cholesky to get the lower trianular matrix needed for MultivariateNormalTriL
    covariance_matrix = tf.linalg.cholesky(covariance_matrix)
    return [mean_network, covariance_matrix]

def get_cov_matrix_per_arg_r(model, x, y, r, n_dim = 2, step_size = 0.04):
    """
    
    Args:
        model --->  neural network
        x -> x component
        y -> y component
        ndim -> 2 for 2 dim data
        step_size -> delta_t in euler mauriyama

    
    """
    mean_network, std_network = model(np.array([[x,y,r]]).astype(np.float32))
    #mean_network = keras.backend.eval(mean_network)# shape out n 
    std_network = keras.backend.eval(std_network) # shape out n*n
    diffusivity_spd_ = std_network
    step_sizes = np.zeros((1,)) + step_size
    spd_step_size = tf.reshape(0.04, (-1, 1)) ##imp line
    spd_step_size = tf.math.sqrt(spd_step_size) # NO square root because we use cholesky below?
    full_shape = n_dim * n_dim
    step_size_matrix = tf.broadcast_to(spd_step_size, [K.shape(step_sizes)[0], full_shape])
    step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

    # multiply with the step size
    covariance_matrix = tf.multiply(step_size_matrix , diffusivity_spd_)
    # square the matrix so that the cholesky decomposition does not change the eienvalues
    covariance_matrix = tf.linalg.matmul(covariance_matrix, tf.linalg.matrix_transpose(covariance_matrix))
    # perform cholesky to get the lower trianular matrix needed for MultivariateNormalTriL
    covariance_matrix = tf.linalg.cholesky(covariance_matrix)
    return [mean_network, covariance_matrix]



def get_cov_matrix(model, x_data, n_dim = 2, step_size = 0.04):
    """
    Will work for r as argument as well

    Args:
        model --->  neural network
        x_data -> 2d data (eg) 
        n_dim -> 2 for 2 dim data
        step_size -> delta_t in euler mauriyama

    
    """
    step_sizes = np.zeros((x_data.shape[0],)) + step_size
    mean_network, std_network = model(x_data.astype(np.float32)) # neural net based
    mean_network = keras.backend.eval(mean_network)# shape out n 
    std_network = keras.backend.eval(std_network) # shape out n*n
    diffusivity_spd_ = std_network
    spd_step_size = tf.reshape(step_sizes, (-1, 1)) ##imp line
    spd_step_size = tf.math.sqrt(spd_step_size) # NO square root because we use cholesky below?
    n_dim = 2
    full_shape = n_dim * n_dim
    step_size_matrix = tf.broadcast_to(spd_step_size, [K.shape(step_sizes)[0], full_shape])
    step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

    # multiply with the step size
    covariance_matrix = tf.multiply(step_size_matrix , diffusivity_spd_)
    # square the matrix so that the cholesky decomposition does not change the eienvalues
    covariance_matrix = tf.linalg.matmul(covariance_matrix, tf.linalg.matrix_transpose(covariance_matrix))
    # perform cholesky to get the lower trianular matrix needed for MultivariateNormalTriL
    covariance_matrix = tf.linalg.cholesky(covariance_matrix)
    return [mean_network, covariance_matrix]

if __name__ == "__main__":
    print ("Executed when invoked directly")
else:
    print ("Executed when imported")

