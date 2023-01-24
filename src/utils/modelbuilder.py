import tensorflow as tf

from tensorflow.keras import layers

import tensorflow_probability as tfp

tfd = tfp.distributions


# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')
NUMBER_TYPE = tf.float64  # or tf.float32

STD_MIN_VALUE = 1e-13  # the minimal number that the diffusivity models can have




class ModelBuilder:
    """
    Constructs neural network models with specified topology.
    """
    DIFF_TYPES = ["diagonal", "triangular", "spd"]

    @staticmethod
    def define_forward_model(n_input_dimensions, n_output_dimensions, n_layers, n_dim_per_layer, name,
                             activation="tanh", dtype=tf.float64):
        
        inputs = layers.Input((n_input_dimensions,), dtype=dtype, name=name + '_inputs')
        network_x = inputs
        for i in range(n_layers):
            network_x = layers.Dense(n_dim_per_layer, activation=activation, dtype=dtype,
                                     name=name + "_hidden/dense_{}".format(i))(network_x)
        network_output = layers.Dense(n_output_dimensions, dtype=dtype,
                                      name=name + "_output_mean", activation=None)(network_x)

        network = tf.keras.Model(inputs=inputs, outputs=network_output,
                                 name=name + "_forward_model")
        return network

    @staticmethod
    def define_gaussian_process(n_input_dimensions, n_output_dimensions, n_layers, n_dim_per_layer, name,
                                diffusivity_type="diagonal", activation="tanh", dtype=tf.float64):
        
        def make_tri_matrix(z):
            # first, make all eigenvalues positive by changing the diagonal to positive values
            #print("in-make_tri_matrix",z.shape) # n(n + 1)/2
            z = tfp.math.fill_triangular(z)
            '''
            https://www.tensorflow.org/probability/api_docs/python/tfp/math/fill_triangular
            fill_triangular([1, 2, 3, 4, 5, 6])
            # ==> [[4, 0, 0],
            #      [6, 5, 0],
            #      [3, 2, 1]]
            '''
            #print("in-make_tri_matrix",z.shape) # (None, n, n)
            z2 = tf.linalg.diag(tf.linalg.diag_part(z)) # (None, n, n)
            """
            tf.linalg.diag_part --> see doc --> takes n*n shape and gives n shaped output
            tf.linalg.diag - takes n shapes in ---> n*n out with non diagonal terms zero
            
            
            """
            z = z - z2 + tf.abs(z2)  # this ensures the values on the diagonal are positive
            return z

        def make_spd_matrix(z):
            z = make_tri_matrix(z)
            return tf.linalg.matmul(z, tf.linalg.matrix_transpose(z))

        #def make_cov_matrix
        
        inputs = layers.Input((n_input_dimensions,), dtype=dtype, name=name + '_inputs')
        gp_x = inputs
        for i in range(n_layers):
            gp_x = layers.Dense(n_dim_per_layer,
                                activation=activation,
                                dtype=dtype,
                                name=name + "_mean_hidden_{}".format(i))(gp_x)
        gp_output_mean = layers.Dense(n_output_dimensions, dtype=dtype,
                                      name=name + "_output_mean", activation=None)(gp_x)

        # initialize with extremely small (not zero!) values so that it does not dominate the drift
        # estimation at the beginning of training
        small_init = 1e-2
        initializer = tf.keras.initializers.RandomUniform(minval=-small_init, maxval=small_init, seed=None)

        gp_x = inputs
        for i in range(n_layers):
            gp_x = layers.Dense(n_dim_per_layer,
                                activation=activation,
                                dtype=dtype,
                                kernel_initializer=initializer,
                                bias_initializer=initializer,
                                name=name + "_std_hidden_{}".format(i))(gp_x)
        if diffusivity_type=="diagonal":
            gp_output_std = layers.Dense(n_output_dimensions,
                                         kernel_initializer=initializer,
                                         bias_initializer=initializer,
                                         activation=lambda x: tf.nn.softplus(x) + STD_MIN_VALUE,
                                         name=name + "_output_std", dtype=dtype)(gp_x)
            #print("-diagonal-std-shape", gp_output_std.shape) # shape[none, n]
        elif diffusivity_type=="triangular":
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the lower triangular matrix with positive eigenvalues on the diagonal.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_cholesky", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_tri_matrix)(gp_output_tril)
            #print("-triangular-std-shape", gp_output_std.shape, gp_output_tril) # shape[none, n, n] shape[none, n(n+1)/2]
        elif diffusivity_type=="spd": # symmetric positive definite
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the SPD matrix C using C = L @ L.T to be used later.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_spd", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_spd_matrix)(gp_output_tril)
            print("-spd-std-shape", gp_output_std.shape, gp_output_tril)

        elif diffusivity_type=="spd_general": # symmetric positive definite
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the SPD matrix C using C = L @ L.T to be used later.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_spd", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_spd_matrix)(gp_output_tril)
            print("-spd-std-shape", gp_output_std.shape, gp_output_tril)
            # gp_output_std = layers.Lambda(lambda L: tf.linalg.matmul(L, tf.transpose(L)))(gp_output_tril)
        else:
            raise ValueError(f"Diffusivity type {diffusivity_type} not supported. Use one of {ModelBuilder.DIFF_TYPES}.")
        
        gp = tf.keras.Model(inputs,
                            [gp_output_mean, gp_output_std],
                            name=name + "_gaussian_process")
        return gp

    




