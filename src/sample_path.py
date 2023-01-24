from utils.modelbuilder import ModelBuilder
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import tqdm


def main(sample_pts = 100000):
    data = pd.read_csv("/home/ece/utkarsh/fish_traj_extract/PyDaddy/pydaddy/data/model_data/vector/ternary.csv", sep =" ", header=None)
    data = data.values
    #data = np.load("/home/ece/utkarsh/fish_traj_extract/nature_paper_data/data/raw_data/30/30_extracted.npy") # experimental data
    n_dimensions = 2 #4
    n_layers = 5
    n_dim_per_layer = 150
    #n_dim_per_layer = 100
    diffusivity_type = "spd"
    ACTIVATIONS = tf.nn.elu
    loaded_model = ModelBuilder.define_gaussian_process(
                                            n_input_dimensions=n_dimensions,
                                            n_output_dimensions=n_dimensions,
                                            n_layers=n_layers,
                                            n_dim_per_layer=n_dim_per_layer,
                                            name="Euler",
                                            diffusivity_type=diffusivity_type,
                                            activation=ACTIVATIONS)

    loaded_model.load_weights("/home/ece/utkarsh/fish_traj_extract/on_pydaddy_data/sim_vec_ternary/augmented_data/rot_mirror_75")

    #np.random.seed(42)
    #a,b = np.random.uniform(-0.002,0.03,[2])
    #x0 = np.array([[0, 0, 0, 0]]).astype(np.float32)
    #x0 = np.array([[a, b]]).astype(np.float32)
    x0 = np.array([data[0]]).astype(np.float32)

    def sample(x_old, mu, covariance_matrix):
        approx_normal = tfd.MultivariateNormalTriL(
        loc=(x_old +  0.04*mu),
        scale_tril=covariance_matrix,
        name="approx_normal")
        return np.array(approx_normal.sample())

    t = np.array(0.04).astype(np.float32) # weird that if you use dt rather than 0.04 it escalates to give an error
    path = [x0]
    #for i in tqdm(range(len_actual_data - 1)): # 
    for i in tqdm(range(sample_pts - 1)): # 
        mu, std = loaded_model(path[-1])
        mu = tf.cast(mu, tf.float32)
        std = tf.cast(std, tf.float32)
        #print(std)
        spd_step_size = tf.reshape(t, (-1, 1)) ##imp line
        spd_step_size = tf.math.sqrt(spd_step_size) # NO square root because we use cholesky below?
        n_dim = 2
        #n_dim = 4
        full_shape = n_dim * n_dim
        step_size_matrix = tf.broadcast_to(spd_step_size, [1, full_shape])
        step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

        # multiply with the step size
        covariance_matrix = tf.multiply(step_size_matrix , std)
        # square the matrix so that the cholesky decomposition does not change the eienvalues
        covariance_matrix = tf.linalg.matmul(covariance_matrix, tf.linalg.matrix_transpose(covariance_matrix))
        # perform cholesky to get the lower trianular matrix needed for MultivariateNormalTriL
        covariance_matrix = tf.linalg.cholesky(covariance_matrix)
        next_step = sample(path[-1], mu, covariance_matrix)
        while np.linalg.norm(next_step) > 1:
            next_step = sample(path[-1], mu, covariance_matrix)

        path.append(next_step)
        print(path[-1])
        

    path = np.array(path)
    print(path.shape)

    np.save("/home/ece/utkarsh/fish_traj_extract/on_pydaddy_data/sim_vec_ternary/augmented_data/rot_mirror_75/sampled_x_0_bc.npy", path.reshape(-1,2))



if __name__ == '__main__':
    main()

