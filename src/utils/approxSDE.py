import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import keras.backend as K
import numpy as np



class SDEApproximationNetwork(tf.keras.Model):
    """
    Euler-Mauriyama based SDE estimation network
    """
    VALID_METHODS = ["euler"]

    def __init__(self,
                 sde_model: tf.keras.Model,
                 step_size=None,
                 method="euler",
                 diffusivity_type="diagonal",
                 scale_per_point=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = step_size
        self.method = method
        self.diffusivity_type = diffusivity_type
        self.scale_per_point = scale_per_point # only used if method="milstein approx"

        SDEApproximationNetwork.verify(self.method)

    @staticmethod
    def verify(method):
        if not (method in SDEApproximationNetwork.VALID_METHODS):
            raise ValueError(method + " is not a valid method. Use any of" + SDEApproximationNetwork.VALID_METHODS)

    def get_config(self):
        return {
            "sde_model": self.sde_model,
            "step_size": self.step_size,
            "method": self.method,
            "diffusivity_type": self.diffusivity_type
        }

    @staticmethod
    def euler_maruyama_pdf(ynp1_, yn_, step_size_, model_, diffusivity_type="diagonal"):
        """
        This implies a very simple sde_model, essentially just a Gaussian process
        on x_n that predicts the drift and diffusivity.
        Returns log P(y(n+1) | y(n)) for the Euler-Maruyama scheme.

        Parameters
        ----------
        ynp1_ next point in time.
        yn_ current point in time.
        step_size_ step size in time.
        model_ sde_model that returns a (drift, diffusivity) tuple.
        diffusivity_type defines which type of diffusivity matrix will be used. See ModelBuilder.DIFF_TYPES.

        Returns
        -------
        logarithm of p(ynp1_ | yn_) under the Euler-Maruyama scheme.

        """
        #print("reached_in euler_mauriyama_pdf")
        drift_, diffusivity_ = model_(yn_)
        #print("reached_in euler_mauriyama_pdf -- Neural net executed")
        #print(drift_.shape, diffusivity_.shape, step_size_.shape, ynp1_.shape, yn_.shape)
        #print("step_size----------------------------",tf.print(step_size_)) # 0.04

        if diffusivity_type=="diagonal":
            approx_normal = tfd.MultivariateNormalDiag(
                loc=(yn_ + step_size_ * drift_),
                scale_diag=tf.math.sqrt(step_size_) * diffusivity_,
                name="approx_normal"
            )
            '''
            covariance_matrix = tf.multiply(step_size_matrix, diffusivity_spd_)
            approx_normal = tfd.MultivariateNormalFullCovariance(
                loc=(yn_ + step_size_ * drift_),
                covariance_matrix = step_size_ * diffusivity_,
                validate_args = True,
                name="approx_normal")
            '''

        elif diffusivity_type=="triangular":
            diffusivity_tril_ = diffusivity_ # size[none, n,n]

            # a cumbersome way to multiply the step size scalar with the batch of matrices...
            # better use tfp.bijectors.FillScaleTriL()
            tril_step_size = tf.math.sqrt(step_size_)
            n_dim = K.shape(yn_)[-1] #2
            full_shape = n_dim * n_dim # 2*2
            step_size_matrix = tf.broadcast_to(tril_step_size, [K.shape(step_size_)[0], full_shape])
            step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

            # now form the normal distribution
            approx_normal = tfd.MultivariateNormalTriL(
                loc=(yn_ + step_size_ * drift_),
                scale_tril=tf.multiply(step_size_matrix, diffusivity_tril_),
                name="approx_normal"
            )
        elif diffusivity_type=="spd":
            diffusivity_spd_ = diffusivity_ # size[none, n,n]

            # a cumbersome way to multiply the step size scalar with the batch of matrices...
            # TODO: REFACTOR with diffusivity_type=="triangular"
            spd_step_size = tf.math.sqrt(step_size_) # NO square root because we use cholesky below?
            n_dim = K.shape(yn_)[-1] # 2
            full_shape = n_dim * n_dim
            step_size_matrix = tf.broadcast_to(spd_step_size, [K.shape(step_size_)[0], full_shape])
            #print("inside--spd",n_dim, full_shape, step_size_matrix.shape)
            step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))
            #print(n_dim, full_shape, step_size_matrix.shape)

            # multiply with the step size
            #covariance_matrix = tf.multiply(tf.math.sqrt(0.04), diffusivity_spd_)
            covariance_matrix = tf.multiply(step_size_matrix, diffusivity_spd_)
            # square the matrix so that the cholesky decomposition does not change the eienvalues
            covariance_matrix = tf.linalg.matmul(covariance_matrix, tf.linalg.matrix_transpose(covariance_matrix))
            # perform cholesky to get the lower trianular matrix needed for MultivariateNormalTriL
            covariance_matrix = tf.linalg.cholesky(covariance_matrix)
            # now form the normal distribution
            #tf.print(step_size_) --> 0.04
            approx_normal = tfd.MultivariateNormalTriL(
                loc=(yn_ + step_size_ * drift_),
                scale_tril=covariance_matrix,
                name="approx_normal"
            )
        
        else:
            raise ValueError(f"Diffusivity type <{diffusivity_type}> not supported. Use one of {ModelBuilder.DIFF_TYPES}.")

        #print("done-euler-mauriyama")
        return approx_normal.log_prob(ynp1_)

    @staticmethod
    def split_inputs(inputs, step_size=None):
        #print("reached in split_inputs", inputs)
        if step_size is None: # executed 
            n_size = (inputs.shape[1] - 1) // 2
            step_size, x_n, x_np1 = tf.split(inputs, num_or_size_splits=[1, n_size, n_size], axis=1)
            #step_size, x_n, x_np1 = tf.split(inputs, num_or_size_splits=[1, 2, 1], axis=1) #[#step_size col, #input_dim, #output_dim]
        else: # not executed
            step_size = step_size
            x_n, x_np1 = tf.split(inputs, num_or_size_splits=2, axis=1)
            #step,size, x_n, x_np1 = tf.split(inputs, num_or_size_splits=[1, 2, 1], axis=1)
        return step_size, x_n, x_np1

    def call_xn(self, inputs_xn):
        """
        Can be used to evaluate the drift and diffusivity
        of the sde_model. This is different than the "call" method
        because it only expects "x_k", not "x_{k+1}" as well.
        """
        return self.sde_model(inputs_xn)

    def call(self, inputs):
        """
        Expects the input tensor to contain all of (step_sizes, x_k, x_{k+1}).
        """
        #print("reached-here-call-of-this-class--sdeapproxiamationnetwork")
        #self.inputs = inputs, TensorShape([None, 4]) --
        step_size, x_n, x_np1 = SDEApproximationNetwork.split_inputs(inputs, self.step_size)
        #print("reached-here-call2-of-this-class--sdeapproxiamationnetwork")

        if self.method == "euler":
            log_prob = SDEApproximationNetwork.euler_maruyama_pdf(x_np1, x_n, step_size, self.sde_model,
                                                                  self.diffusivity_type)
        elif self.method == "milstein":
            log_prob = SDEApproximationNetwork.milstein_pdf_regularized(x_np1, x_n, step_size, self.sde_model)
        elif self.method == "milstein approx":
            log_prob = SDEApproximationNetwork.milstein_forward_approx(x_np1, x_n, step_size, self.sde_model, self.scale_per_point)
        else:
            raise ValueError(self.method + " not available")

        sample_distortion = -tf.reduce_mean(log_prob, axis=-1)
        distortion = tf.reduce_mean(sample_distortion)

        loss = distortion

        self.add_loss(loss)
        self.add_metric(distortion, name="distortion", aggregation="mean")

        return self.sde_model(x_n)
