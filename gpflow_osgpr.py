import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from copy import deepcopy

import gpflow
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from gpflow import covariances, utilities
from gpflow.models import GPModel, InternalDataTrainingLossMixin


class OSGPR(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data,
        kernel,
        likelihood,
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
        # sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        # Saa = self.Su_old
        # ma = self.mu_old

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

        Qaa = tf.matmul(Lbinv_Kba, Lbinv_Kba, transpose_a=True)
        # Kaadiff = Kaa_cur - tf.matmul(Lbinv_Kba, Lbinv_Kba, transpose_a=True)
        Kaadiff = Kaa_cur - Qaa
        Sainv_Kaadiff = tf.linalg.solve(Saa, Kaadiff)
        Kainv_Kaadiff = tf.linalg.solve(z_kernel_mat, Kaadiff)

        constant_term = -0.5 * N * np.log(2 * np.pi)
        quadratic_term = (
            -0.5 * tf.reduce_sum(tf.square(err)) / sigma2
            + 0.5 * tf.reduce_sum(tf.square(LDinv_Lbinv_c))
            - 0.5 * tf.reduce_sum(tf.square(Lainv_ma))
        )
        log_terms = (
            -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))
            - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSa)))
            + tf.reduce_sum(tf.math.log(tf.linalg.diag_part(La)))
            - 0.5 * N * tf.reduce_sum(tf.math.log(sigma2))
        )

        trace_terms = (
            -0.5
            * tf.reduce_sum(
                tf.linalg.diag_part(Sainv_Kaadiff) - tf.linalg.diag_part(Kainv_Kaadiff)
            )
            - 0.5 * tf.reduce_sum(Kfdiag) / sigma2
            + 0.5 * tf.reduce_sum(tf.linalg.diag_part(Qff))
        )
        return constant_term + quadratic_term + log_terms + trace_terms
        # # constant term
        # bound = -0.5 * N * np.log(2 * np.pi)
        # # quadratic term
        # bound += -0.5 * tf.reduce_sum(tf.square(err)) / sigma2
        # # bound += -0.5 * tf.reduce_sum(ma * Sainv_ma)
        # bound += -0.5 * tf.reduce_sum(tf.square(Lainv_ma))
        # bound += 0.5 * tf.reduce_sum(tf.square(LDinv_Lbinv_c))
        # # log det term
        # bound += -0.5 * N * tf.reduce_sum(tf.math.log(sigma2))
        # bound += -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))
        # # delta 1: trace term
        # bound += -0.5 * tf.reduce_sum(Kfdiag) / sigma2
        # bound += 0.5 * tf.reduce_sum(tf.linalg.diag_part(Qff))
        # # delta 2: a and b difference
        # bound += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(La)))
        # bound += -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSa)))
        # bound += -0.5 * tf.reduce_sum(
        #     tf.linalg.diag_part(Sainv_Kaadiff) - tf.linalg.diag_part(Kainv_Kaadiff)
        # )
        # return bound

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
        kernel,
        noise_variance=0.001,
    ):
        self.num_inducing = num_inducing
        self.kernel = kernel
        self.init_noise_variance = noise_variance
        self.model = None
        self.z = None
        self.z_mean = None
        self.z_covar_mat = None
        self.z_kernel_mat = None
        self.is_init_fit = True

    @property
    def noise_variance(self):
        return self.model.likelihood.variance.numpy()

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
                noise_variance=self.init_noise_variance,
            )
        else:
            z = self.init_inducing_points(train_x)
            self.model = OSGPR(
                (train_x, train_y),
                deepcopy(self.model.kernel),
                deepcopy(self.model.likelihood),
                z,
                self.z,
                self.z_mean,
                self.z_covar_mat,
                self.z_kernel_mat,
            )
        gpflow.optimizers.Scipy().minimize(
            self.model.training_loss,
            self.model.trainable_variables,
            options={"disp": False},
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
        # new_z = np.linspace(-2, 12, self.num_inducing).reshape(-1, 1)
        old_z = self.inducing_points
        num_z = old_z.shape[0]
        num_old_z = int(keep_rate * num_z)
        num_new_z = num_z - num_old_z
        old_indices = np.random.permutation(num_z)[:num_old_z]
        old_z = old_z[old_indices, :]
        new_indices = np.random.permutation(x_new.shape[0])[:num_new_z]
        new_z = x_new[new_indices, :]
        new_z = np.vstack((old_z, new_z))
        return new_z


def get_data(is_iid_data):
    train_x = np.loadtxt("./data_x.txt", delimiter=",").reshape(-1, 1)
    train_y = np.loadtxt("./data_y.txt", delimiter=",").reshape(-1, 1)
    num_train = len(train_y)
    batch_size = num_train // 3
    # Move the first third of the data to the left by 1
    train_x[:batch_size, :] -= 1
    # Move the second third of the data to the right by 1
    train_x[2 * batch_size : 3 * batch_size, :] += 1
    if is_iid_data:
        indices = np.random.permutation(train_x)
        train_x = train_x[indices, :]
        train_y = train_y[indices, :]
    test_x = np.linspace(-2, 12, 100)[:, None]
    return train_x, train_y, batch_size, test_x


def plot_result(
    ax, train_x, train_y, old_x, old_y, test_x, test_mean, test_var, opt_z, z_mean
):
    ax.plot(train_x, train_y, "kx", mew=1, alpha=0.8)
    if old_x is not None and old_y is not None:
        ax.plot(old_x, old_y, "kx", mew=1, alpha=0.2)
    ax.plot(test_x, test_mean, "b", lw=2)
    ax.fill_between(
        test_x.ravel(),
        test_mean.ravel() - 2 * np.sqrt(test_var.ravel()),
        test_mean.ravel() + 2 * np.sqrt(test_var.ravel()),
        color="b",
        alpha=0.3,
    )
    ax.plot(opt_z, z_mean, "ro", mew=1)
    ax.set_ylim([-3.0, 3.0])
    ax.set_xlim([np.min(test_x), np.max(test_x)])
    plt.subplots_adjust(hspace=0.08)
    ax.set_ylabel("y")
    ax.yaxis.set_ticks(np.arange(-2, 3, 1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))


def main(is_iid_data, num_inducing=10):
    train_x, train_y, batch_size, test_x = get_data(is_iid_data)
    fig, axes = plt.subplots(4, 1, sharey=True, sharex=True)
    online_model = GPflowModel(
        num_inducing=num_inducing,
        kernel=gpflow.kernels.RBF(variance=1.0, lengthscales=0.8),
    )
    for batch_index in range(3):
        x_batch = train_x[batch_index * batch_size : (batch_index + 1) * batch_size, :]
        y_batch = train_y[batch_index * batch_size : (batch_index + 1) * batch_size, :]
        old_x = train_x[: batch_index * batch_size, :]
        old_y = train_y[: batch_index * batch_size, :]
        online_model.fit(x_batch, y_batch)
        test_mean, test_var = online_model.predict(test_x)
        plot_result(
            axes[batch_index],
            train_x,
            train_y,
            old_x,
            old_y,
            test_x,
            test_mean,
            test_var,
            online_model.inducing_points,
            online_model.z_mean,
        )
    # Train on the full dataset using SGPR
    batch_model = GPflowModel(
        num_inducing=num_inducing,
        kernel=gpflow.kernels.RBF(variance=1.0, lengthscales=0.8),
    )
    batch_model.fit(train_x, train_y)
    test_mean, test_var = batch_model.predict(test_x)
    plot_result(
        axes[3],
        train_x,
        train_y,
        None,
        None,
        test_x,
        test_mean,
        test_var,
        batch_model.inducing_points,
        batch_model.z_mean,
    )
    axes[3].set_xlabel("x")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(10)
    main(is_iid_data=False)
