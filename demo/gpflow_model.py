from copy import deepcopy

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import covariances, utilities
from gpflow.models import GPModel, InternalDataTrainingLossMixin

tf.keras.backend.set_floatx("float64")

class AttentiveKernel(gpflow.kernels.Kernel):
    def __init__(self, dim_input=1, dim_hidden=10, dim_output=10):
        super().__init__()
        lengthscales = np.linspace(0.01, 0.5, num=dim_output)
        self.amplitude = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscales = gpflow.Parameter(lengthscales, trainable=False)
        with tf.name_scope("nn"):
            self.nn = tf.keras.Sequential()
            self.nn.add(tf.keras.layers.Dense(dim_hidden, input_shape=(dim_input,), activation='tanh'))
            self.nn.add(tf.keras.layers.Dense(dim_hidden, activation='tanh'))
            self.nn.add(tf.keras.layers.Dense(dim_output, activation='softmax'))
            self.nn.build()

    def get_representations(self, X):
        Z = self.nn(X)
        representations = Z / tf.norm(Z, axis=1, keepdims=True)
        return representations

    def K_diag(self, X):
        return tf.fill((tf.shape(X)[0], 1), self.amplitude)

    def K(self, X1, X2=None):
        if X2 is None:
            X2 = X1

        dist = tf.norm(tf.expand_dims(X1, 1) - tf.expand_dims(X2, 0), axis=2)
        repre1 = self.get_representations(X1)
        repre2 = self.get_representations(X2)
        cov_mat = tf.zeros_like(dist, dtype=tf.float64)

        for i in range(self.num_lengthscales):
            attention_lengthscales = tf.linalg.matmul(
                tf.expand_dims(repre1[:, i], 1),
                tf.expand_dims(repre2[:, i], 0)
            )
            cov_mat += self.rbf(dist, self.lengthscales[i]) * attention_lengthscales

        attention_inputs = tf.linalg.matmul(repre1, repre2, transpose_b=True)
        cov_mat *= self.amplitude * attention_inputs

        return cov_mat

    @property
    def num_lengthscales(self):
        return self.lengthscales.shape[0]

    def rbf(self, X, lengthscale):
        return tf.exp(-0.5 * (X / lengthscale) ** 2)


class OSGPR(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data,
        kernel,
        Z,
        old_z,
        mu_old,
        Su_old,
        Kaa_old,
        mean_function=None,
        jitter=1e-4,
    ):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate gpflow objects
        mu_old, Su_old are mean and covariance of old q(u)
        old_z is the old inducing inputs
        This method only works with a Gaussian likelihood.
        """
        # numpy array to tensor and force to default float (float64)
        self.train_x, self.Y = self.data = gpflow.models.util.data_input_to_tensor(data)
        likelihood = gpflow.likelihoods.Gaussian()
        num_latent_gps = gpflow.models.GPModel.calc_num_latent_gps_from_data(
            data, kernel, likelihood
        )
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.inducing_variable = gpflow.inducing_variables.InducingPoints(Z)
        self.num_data = self.train_x.shape[0]
        self.jitter = utilities.to_default_float(jitter)

        self.mu_old = tf.Variable(
            mu_old,
            shape=tf.TensorShape(None),
            trainable=False,
        )
        self.M_old = old_z.shape[0]
        self.Su_old = tf.Variable(
            Su_old,
            shape=tf.TensorShape(None),
            trainable=False,
        )
        self.Kaa_old = tf.Variable(
            Kaa_old,
            shape=tf.TensorShape(None),
            trainable=False,
        )
        self.old_z = tf.Variable(
            old_z,
            shape=tf.TensorShape(None),
            trainable=False,
        )

    def maximum_log_likelihood_objective(self):
        sigma2 = self.likelihood.variance
        N = self.num_data

        Saa = self.Su_old
        ma = self.mu_old

        Kfdiag = self.kernel(self.train_x, full_cov=False)
        Mb = self.inducing_variable.num_inducing
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        Saa = self.Su_old
        ma = self.mu_old

        Kbf = covariances.Kuf(self.inducing_variable, self.kernel, self.train_x)
        Kbb = covariances.Kuu(self.inducing_variable, self.kernel, jitter=self.jitter)
        Kba = covariances.Kuf(self.inducing_variable, self.kernel, self.old_z)
        Kaa_cur = utilities.add_noise_cov(self.kernel(self.old_z), self.jitter)
        z_kernel_mat = utilities.add_noise_cov(self.Kaa_old, self.jitter)

        err = self.Y - self.mean_function(self.train_x)

        Sainv_ma = tf.linalg.solve(Saa, ma)
        Sinv_y = self.Y / sigma2
        c1 = tf.matmul(Kbf, Sinv_y)
        c2 = tf.matmul(Kba, Sainv_ma)
        c = c1 + c2

        Lb = tf.linalg.cholesky(Kbb)
        Lbinv_c = tf.linalg.triangular_solve(Lb, c, lower=True)
        Lbinv_Kba = tf.linalg.triangular_solve(Lb, Kba, lower=True)
        Lbinv_Kbf = tf.linalg.triangular_solve(Lb, Kbf, lower=True) / sigma
        d1 = tf.matmul(Lbinv_Kbf, Lbinv_Kbf, transpose_b=True)
        Qff = d1

        LSa = tf.linalg.cholesky(Saa)
        Kab_Lbinv = tf.linalg.matrix_transpose(Lbinv_Kba)
        LSainv_Kab_Lbinv = tf.linalg.triangular_solve(LSa, Kab_Lbinv, lower=True)
        d2 = tf.matmul(LSainv_Kab_Lbinv, LSainv_Kab_Lbinv, transpose_a=True)

        La = tf.linalg.cholesky(z_kernel_mat)
        Lainv_Kab_Lbinv = tf.linalg.triangular_solve(La, Kab_Lbinv, lower=True)
        d3 = tf.matmul(Lainv_Kab_Lbinv, Lainv_Kab_Lbinv, transpose_a=True)

        D = tf.eye(Mb, dtype=gpflow.default_float()) + d1 + d2 - d3
        D = utilities.add_noise_cov(D, self.jitter)
        LD = tf.linalg.cholesky(D)

        LDinv_Lbinv_c = tf.linalg.triangular_solve(LD, Lbinv_c, lower=True)

        LSa = tf.linalg.cholesky(Saa)
        Lainv_ma = tf.linalg.triangular_solve(LSa, ma, lower=True)

        Kaadiff = Kaa_cur - tf.matmul(Lbinv_Kba, Lbinv_Kba, transpose_a=True)
        Sainv_Kaadiff = tf.linalg.solve(Saa, Kaadiff)
        Kainv_Kaadiff = tf.linalg.solve(z_kernel_mat, Kaadiff)

        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / sigma2
        # bound += -0.5 * tf.reduce_sum(ma * Sainv_ma)
        bound += -0.5 * tf.reduce_sum(tf.square(Lainv_ma))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_Lbinv_c))
        # log det term
        bound += -0.5 * N * tf.reduce_sum(tf.math.log(sigma2))
        bound += -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))
        # delta 1: trace term
        bound += -0.5 * tf.reduce_sum(Kfdiag) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.linalg.diag_part(Qff))
        # delta 2: a and b difference
        bound += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(La)))
        bound += -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSa)))
        bound += -0.5 * tf.reduce_sum(
            tf.linalg.diag_part(Sainv_Kaadiff) - tf.linalg.diag_part(Kainv_Kaadiff)
        )
        return bound

    def predict_f(self, test_x, full_cov=False):
        Kbs = covariances.Kuf(self.inducing_variable, self.kernel, test_x)
        Kbb = covariances.Kuu(self.inducing_variable, self.kernel, jitter=self.jitter)
        Kbf = covariances.Kuf(self.inducing_variable, self.kernel, self.train_x)
        Kba = covariances.Kuf(self.inducing_variable, self.kernel, self.old_z)

        # Compute Lb
        Lb = tf.linalg.cholesky(Kbb)

        # Compute LD
        sigma = tf.sqrt(self.likelihood.variance)
        Lb = tf.linalg.cholesky(Kbb)
        Lbinv_Kba = tf.linalg.triangular_solve(Lb, Kba, lower=True)
        Lbinv_Kbf = tf.linalg.triangular_solve(Lb, Kbf, lower=True) / sigma
        d1 = tf.matmul(Lbinv_Kbf, Lbinv_Kbf, transpose_b=True)

        LSa = tf.linalg.cholesky(self.Su_old)
        Kab_Lbinv = tf.linalg.matrix_transpose(Lbinv_Kba)
        LSainv_Kab_Lbinv = tf.linalg.triangular_solve(LSa, Kab_Lbinv, lower=True)
        d2 = tf.matmul(LSainv_Kab_Lbinv, LSainv_Kab_Lbinv, transpose_a=True)

        z_kernel_mat = utilities.add_noise_cov(self.Kaa_old, self.jitter)
        La = tf.linalg.cholesky(z_kernel_mat)
        Lainv_Kab_Lbinv = tf.linalg.triangular_solve(La, Kab_Lbinv, lower=True)
        d3 = tf.matmul(Lainv_Kab_Lbinv, Lainv_Kab_Lbinv, transpose_a=True)

        Mb = self.inducing_variable.num_inducing
        D = tf.eye(Mb, dtype=gpflow.default_float()) + d1 + d2 - d3
        D = utilities.add_noise_cov(D, self.jitter)
        LD = tf.linalg.cholesky(D)

        # Compute LDinv_Lbinv_c
        Sinv_y = self.Y / self.likelihood.variance
        c1 = tf.matmul(Kbf, Sinv_y)

        Sainv_ma = tf.linalg.solve(self.Su_old, self.mu_old)
        c2 = tf.matmul(Kba, Sainv_ma)
        c = c1 + c2
        Lbinv_c = tf.linalg.triangular_solve(Lb, c, lower=True)
        LDinv_Lbinv_c = tf.linalg.triangular_solve(LD, Lbinv_c, lower=True)

        Lbinv_Kbs = tf.linalg.triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.linalg.triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_c, transpose_a=True)

        if full_cov:
            Kss = self.kernel(test_x) + self.jitter * tf.eye(
                tf.shape(test_x)[0], dtype=gpflow.default_float()
            )
            var1 = Kss
            var2 = -tf.matmul(Lbinv_Kbs, Lbinv_Kbs, transpose_a=True)
            var3 = tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_Kbs, transpose_a=True)
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(test_x, full_cov=False)
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), axis=0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), axis=0)
            var = var1 + var2 + var3
        return mean + self.mean_function(test_x), var


class GPflowModel:
    def __init__(
        self,
        num_inducing,
        # kernel=gpflow.kernels.RBF(variance=1.0, lengthscales=0.8),
        kernel=AttentiveKernel(),
        noise_variance=0.001,
    ):
        self.num_inducing = num_inducing
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.model = None
        self.z = None
        self.z_mean = None
        self.z_covar_mat = None
        self.z_kernel_mat = None
        self.is_init_fit = True

    def fit(self, train_x, train_y):
        if self.is_init_fit:
            self.is_init_fit = False
            new_z = train_x[
                np.random.permutation(train_x.shape[0])[: self.num_inducing], :
            ]
            self.model = gpflow.models.SGPR(
                (train_x, train_y),
                self.kernel,
                inducing_variable=new_z,
                noise_variance=self.noise_variance,
            )
        else:
            z = self.init_inducing_points(train_x)
            self.model = OSGPR(
                (train_x, train_y),
                deepcopy(self.model.kernel),
                z,
                self.z,
                self.z_mean,
                self.z_covar_mat,
                self.z_kernel_mat,
            )
            self.model.likelihood.variance.assign(self.model.likelihood.variance)
        gpflow.optimizers.Scipy().minimize(
            self.model.training_loss,
            self.model.trainable_variables,
            options={"disp": True},
        )
        self.summarize()

    def predict(self, test_x):
        test_mean, test_var = self.model.predict_f(test_x)
        if len(test_var.shape) == 3:
            test_var = test_var[:, 0]
        return test_mean.numpy(), test_var.numpy()

    @property
    def inducing_points(self):
        return self.model.inducing_variable.Z.numpy()

    def summarize(self):
        self.z = self.inducing_points
        self.z_mean, self.z_covar_mat = self.model.predict_f(self.z, full_cov=True)
        self.z_kernel_mat = self.model.kernel(self.z)

        if len(self.z_covar_mat.shape) == 3:
            self.z_covar_mat = self.z_covar_mat[0, :, :]

    def init_inducing_points(self, x_new, keep_rate=0.7):
        old_z = self.inducing_points
        num_z = old_z.shape[0]
        num_old_z = int(keep_rate * num_z)
        num_new_z = num_z - num_old_z
        old_z = old_z[np.random.permutation(num_z)[:num_old_z], :]
        new_z = x_new[np.random.permutation(x_new.shape[0])[:num_new_z], :]
        new_z = np.vstack((old_z, new_z))
        return new_z
