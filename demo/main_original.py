import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings('ignore')

import gpflow
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf


class OSGPR_VFE(gpflow.models.GPModel,
                gpflow.models.InternalDataTrainingLossMixin):
    """
    Online Sparse Variational GP regression.
    
    Streaming Gaussian process approximations
    Thang D. Bui, Cuong V. Nguyen, Richard E. Turner
    NIPS 2017
    """

    def __init__(self,
                 data,
                 kernel,
                 mu_old,
                 Su_old,
                 Kaa_old,
                 Z_old,
                 Z,
                 mean_function=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate gpflow objects
        mu_old, Su_old are mean and covariance of old q(u)
        Z_old is the old inducing inputs
        This method only works with a Gaussian likelihood.
        """
        self.X, self.Y = self.data = gpflow.models.util.data_input_to_tensor(
            data)
        likelihood = gpflow.likelihoods.Gaussian()
        num_latent_gps = gpflow.models.GPModel.calc_num_latent_gps_from_data(
            data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.inducing_variable = gpflow.inducing_variables.InducingPoints(Z)
        self.num_data = self.X.shape[0]

        self.mu_old = tf.Variable(mu_old,
                                  shape=tf.TensorShape(None),
                                  trainable=False)
        self.M_old = Z_old.shape[0]
        self.Su_old = tf.Variable(Su_old,
                                  shape=tf.TensorShape(None),
                                  trainable=False)
        self.Kaa_old = tf.Variable(Kaa_old,
                                   shape=tf.TensorShape(None),
                                   trainable=False)
        self.Z_old = tf.Variable(Z_old,
                                 shape=tf.TensorShape(None),
                                 trainable=False)

    def _common_terms(self):
        Mb = self.inducing_variable.num_inducing
        Ma = self.M_old
        # jitter = gpflow.default_jitter()
        jitter = gpflow.utilities.to_default_float(1e-4)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbf = gpflow.covariances.Kuf(self.inducing_variable, self.kernel,
                                     self.X)
        Kbb = gpflow.covariances.Kuu(self.inducing_variable,
                                     self.kernel,
                                     jitter=jitter)
        Kba = gpflow.covariances.Kuf(self.inducing_variable, self.kernel,
                                     self.Z_old)
        Kaa_cur = gpflow.utilities.add_noise_cov(self.kernel(self.Z_old),
                                                 jitter)
        Kaa = gpflow.utilities.add_noise_cov(self.Kaa_old, jitter)

        err = self.Y - self.mean_function(self.X)

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

        LSa = tf.linalg.cholesky(Saa)
        Kab_Lbinv = tf.linalg.matrix_transpose(Lbinv_Kba)
        LSainv_Kab_Lbinv = tf.linalg.triangular_solve(LSa,
                                                      Kab_Lbinv,
                                                      lower=True)
        d2 = tf.matmul(LSainv_Kab_Lbinv, LSainv_Kab_Lbinv, transpose_a=True)

        La = tf.linalg.cholesky(Kaa)
        Lainv_Kab_Lbinv = tf.linalg.triangular_solve(La, Kab_Lbinv, lower=True)
        d3 = tf.matmul(Lainv_Kab_Lbinv, Lainv_Kab_Lbinv, transpose_a=True)

        D = tf.eye(Mb, dtype=gpflow.default_float()) + d1 + d2 - d3
        D = gpflow.utilities.add_noise_cov(D, jitter)
        LD = tf.linalg.cholesky(D)

        LDinv_Lbinv_c = tf.linalg.triangular_solve(LD, Lbinv_c, lower=True)

        return (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD, Lbinv_Kba,
                LDinv_Lbinv_c, err, d1)

    def maximum_log_likelihood_objective(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. 
        """

        Mb = self.inducing_variable.num_inducing
        Ma = self.M_old
        jitter = gpflow.default_jitter()
        # jitter = gpflow.utilities.to_default_float(1e-4)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        N = self.num_data

        Saa = self.Su_old
        ma = self.mu_old

        # a is old inducing points, b is new
        # f is training points
        Kfdiag = self.kernel(self.X, full_cov=False)
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD, Lbinv_Kba, LDinv_Lbinv_c,
         err, Qff) = self._common_terms()

        LSa = tf.linalg.cholesky(Saa)
        Lainv_ma = tf.linalg.triangular_solve(LSa, ma, lower=True)

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

        Kaadiff = Kaa_cur - tf.matmul(Lbinv_Kba, Lbinv_Kba, transpose_a=True)
        Sainv_Kaadiff = tf.linalg.solve(Saa, Kaadiff)
        Kainv_Kaadiff = tf.linalg.solve(Kaa, Kaadiff)

        bound += -0.5 * tf.reduce_sum(
            tf.linalg.diag_part(Sainv_Kaadiff) -
            tf.linalg.diag_part(Kainv_Kaadiff))

        return bound

    def predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. 
        """

        # jitter = gpflow.default_jitter()
        jitter = gpflow.utilities.to_default_float(1e-4)

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbs = gpflow.covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD, Lbinv_Kba, LDinv_Lbinv_c,
         err, Qff) = self._common_terms()

        Lbinv_Kbs = tf.linalg.triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.linalg.triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_c, transpose_a=True)

        if full_cov:
            Kss = self.kernel(Xnew) + jitter * tf.eye(
                tf.shape(Xnew)[0], dtype=gpflow.default_float())
            var1 = Kss
            var2 = -tf.matmul(Lbinv_Kbs, Lbinv_Kbs, transpose_a=True)
            var3 = tf.matmul(LDinv_Lbinv_Kbs,
                             LDinv_Lbinv_Kbs,
                             transpose_a=True)
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(Xnew, full_cov=False)
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), axis=0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), axis=0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var


def figsize(scale, ratio=None):
    fig_width_pt = 397.4849  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) -
                   1.0) / 4  # Aesthetic ratio (you could change this)
    if ratio is not None:
        golden_mean = ratio
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def init_Z(cur_Z, new_X, use_old_Z=True, first_batch=True):
    if use_old_Z:
        Z = np.copy(cur_Z)
    else:
        M = cur_Z.shape[0]
        M_old = int(0.7 * M)
        M_new = M - M_old
        old_Z = cur_Z[np.random.permutation(M)[0:M_old], :]
        new_Z = new_X[np.random.permutation(new_X.shape[0])[0:M_new], :]
        Z = np.vstack((old_Z, new_Z))
    return Z


def plot_model(model, ax, cur_x, cur_y, pred_x, seen_x=None, seen_y=None):
    mx, vx = model.predict_f(pred_x)
    Zopt = model.inducing_variable.Z.numpy()
    mu, Su = model.predict_f(Zopt, full_cov=True)
    if len(Su.shape) == 3:
        Su = Su[0, :, :]
        vx = vx[:, 0]
    ax.plot(cur_x, cur_y, 'kx', mew=1, alpha=0.8)
    if seen_x is not None:
        ax.plot(seen_x, seen_y, 'kx', mew=1, alpha=0.2)
    ax.plot(pred_x, mx, 'b', lw=2)
    ax.fill_between(pred_x[:, 0],
                    mx[:, 0] - 2 * np.sqrt(vx),
                    mx[:, 0] + 2 * np.sqrt(vx),
                    color='b',
                    alpha=0.3)
    ax.plot(Zopt, mu, 'ro', mew=1)
    ax.set_ylim([-2.4, 2])
    ax.set_xlim([np.min(pred_x), np.max(pred_x)])
    plt.subplots_adjust(hspace=.08)
    ax.set_ylabel('y')
    ax.yaxis.set_ticks(np.arange(-2, 3, 1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    return mu, Su, Zopt


def get_data(shuffle):
    X = np.loadtxt('./data/1d_x_train.txt', delimiter=',')
    y = np.loadtxt('./data/1d_y_train.txt', delimiter=',')
    X = X.reshape(X.shape[0], 1)
    y = y.reshape((y.shape[0], 1))
    N = y.shape[0]
    gap = N // 3
    X[:gap, :] = X[:gap, :] - 1
    X[2 * gap:3 * gap, :] = X[2 * gap:3 * gap, :] + 1
    if shuffle:
        idxs = np.random.permutation(N)
        X = X[idxs, :]
        y = y[idxs, :]
    return X, y


def optimize(model, **options):
    gpflow.optimizers.Scipy().minimize(model.training_loss,
                                       model.trainable_variables,
                                       options=options)


def main(M, use_old_Z, shuffle):
    fig, axs = plt.subplots(4,
                            1,
                            figsize=figsize(1, ratio=12.0 / 19.0),
                            sharey=True,
                            sharex=True)

    X, y = get_data(shuffle)

    N = X.shape[0]
    gap = N // 3

    # get the first portion and call sparse GP regression
    X1 = X[:gap, :]
    y1 = y[:gap, :]
    seen_x = None
    seen_y = None
    # Z1 = np.random.rand(M, 1)*L
    Z1 = X1[np.random.permutation(X1.shape[0])[0:M], :]

    model1 = gpflow.models.SGPR((X1, y1),
                                gpflow.kernels.RBF(variance=1.0,
                                                   lengthscales=0.8),
                                inducing_variable=Z1,
                                noise_variance=0.001)
    optimize(model1, disp=1)

    # plot prediction
    xx = np.linspace(-2, 12, 100)[:, None]
    mu1, Su1, Zopt = plot_model(model1, axs[0], X1, y1, xx, seen_x, seen_y)

    # now call online method on the second portion of the data
    X2 = X[gap:2 * gap, :]
    y2 = y[gap:2 * gap, :]
    seen_x = X[:gap, :]
    seen_y = y[:gap, :]

    Kaa1 = model1.kernel(model1.inducing_variable.Z)

    Zinit = init_Z(Zopt, X2, use_old_Z)
    model2 = OSGPR_VFE(
        (X2, y2),
        gpflow.kernels.RBF(variance=model1.kernel.variance,
                           lengthscales=model1.kernel.lengthscales), mu1, Su1,
        Kaa1, Zopt, Zinit)
    model2.likelihood.variance.assign(model1.likelihood.variance)
    optimize(model2, disp=1)

    # plot prediction
    mu2, Su2, Zopt = plot_model(model2, axs[1], X2, y2, xx, seen_x, seen_y)

    # now call online method on the third portion of the data
    X3 = X[2 * gap:3 * gap, :]
    y3 = y[2 * gap:3 * gap, :]
    seen_x = np.vstack((seen_x, X2))
    seen_y = np.vstack((seen_y, y2))

    Kaa2 = model2.kernel(model2.inducing_variable.Z)

    Zinit = init_Z(Zopt, X3, use_old_Z)
    model3 = OSGPR_VFE(
        (X3, y3),
        gpflow.kernels.RBF(variance=model2.kernel.variance,
                           lengthscales=model2.kernel.lengthscales), mu2, Su2,
        Kaa2, Zopt, Zinit)
    model3.likelihood.variance.assign(model2.likelihood.variance)
    optimize(model3, disp=1)
    mu3, Su3, Zopt = plot_model(model3, axs[2], X3, y3, xx, seen_x, seen_y)

    Z4 = X[np.random.permutation(X.shape[0])[0:M], :]
    model4 = gpflow.models.SGPR((X, y),
                                gpflow.kernels.RBF(variance=1.0,
                                                   lengthscales=0.8),
                                inducing_variable=Z4,
                                noise_variance=0.001)
    optimize(model4, disp=1)

    # plot prediction
    xx = np.linspace(-2, 12, 100)[:, None]
    mu4, Su4, Zopt4 = plot_model(model4, axs[3], X, y, xx, None, None)
    axs[3].set_xlabel('x')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    M = 10
    use_old_Z = False
    seed = 10
    shuffle = False
    np.random.seed(seed)
    main(M, use_old_Z, shuffle)
