import warnings

import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


class GreaterThanConstraint:
    def __init__(self, min_value=1e-6):
        self.min_value = min_value

    def transform(self, x):
        return F.softplus(x) + self.min_value

    def inv_transform(self, x):
        if torch.any(x <= self.min_value):
            raise ValueError(f"Input must be greater than {self.min_value}.")
        shifted_x = x - self.min_value
        return shifted_x + torch.log(-torch.expm1(-shifted_x))


constraint = GreaterThanConstraint()


def to_tensor(input):
    return torch.tensor(input, dtype=DTYPE, device=DEVICE)


def to_array(tensor):
    return tensor.detach().cpu().numpy()


def compute_tril(A, base_jitter=1e-6, num_attempts=3):
    L, info = torch.linalg.cholesky_ex(A)
    if not torch.any(info):
        # The decomposition was successful.
        return L
    if torch.isnan(A).any():
        raise ValueError("Input to `robust_cholesky` must not contain NaNs.")
    _A = A.clone()
    prev_jitter = 0.0
    jitter = base_jitter
    for i in range(num_attempts):
        not_positive_definite = info > 0
        jitter = base_jitter * (10**i)
        increment = not_positive_definite * (jitter - prev_jitter)
        _A.diagonal().add_(increment)
        prev_jitter = jitter
        warnings.warn(
            f"Added jitter of {jitter} to the diagonal of a matrix that was not"
            + "positive-definite. Attempting Cholesky decomposition again."
        )
        L, info = torch.linalg.cholesky_ex(_A)
        if not torch.any(info):
            return L
    raise ValueError(
        f"Cholesky decomposition failed after adding {jitter} to the diagonal."
        + "Try increasing the `base_jitter` or `num_attempts` arguments."
    )


def tril_solve(L, B):
    """
    Solve a linear system L @ X = B with a lower-triangular matrix L.
    """
    return torch.linalg.solve_triangular(L, B, upper=False)


def tril_logdet(L):
    """
    Computes the log determinant of a matrix A = L @ L.T using the
    Cholesky decomposition of A.
    """
    return 2.0 * L.diagonal().log().sum()


def tril_diag_quadratic(R):
    """
    returns a column vector of the diagonal elements of R^{T}R

    """
    return R.square().sum(dim=0).view(-1, 1)


def quadratic_form(x, y):
    return (x * y).sum()


def trace_quadratic(A):
    """
    returns the trace of A.T @ A or A @ A.T
    """
    return A.square().sum()


def plot_result(
    ax,
    x_train,
    y_train,
    x_test,
    mean,
    var,
    noise_variance,
    x_pseudo=None,
    f_pseudo=None,
    x_old=None,
    y_old=None,
    plot_legend=False,
):
    x_test = x_test.ravel()
    mean = mean.ravel()
    std = np.sqrt(var.ravel())
    noise_std = np.sqrt(noise_variance)

    ax.plot(x_train, y_train, "kx", mew=1, alpha=0.8, label="Training")
    if x_old is not None and y_old is not None:
        ax.plot(x_old, y_old, "kx", mew=1, alpha=0.2, label="Old")
    ax.plot(x_test, mean, "b", lw=2, label="Mean")
    ax.fill_between(
        x_test.ravel(),
        mean - 2 * std,
        mean + 2 * std,
        color="b",
        alpha=0.3,
        label="±2σ(f)",
    )
    ax.fill_between(
        x_test.ravel(),
        mean - 2 * std - 2 * noise_std,
        mean + 2 * std + 2 * noise_std,
        color="c",
        alpha=0.2,
        label="±2σ(y)",
    )
    if x_pseudo is not None and f_pseudo is not None:
        ax.plot(x_pseudo, f_pseudo, "ro", mew=1, alpha=0.8, label="Pseudo")
    ax.set_ylim([-3.0, 3.0])
    ax.set_xlim([np.min(x_test), np.max(x_test)])
    plt.subplots_adjust(hspace=0.08)
    ax.set_ylabel("y")
    ax.yaxis.set_ticks(np.arange(-2, 3, 1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    if plot_legend:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.6),
            fancybox=True,
            shadow=True,
            ncol=5,
        )


def get_data(shuffle=False):
    x_train = np.loadtxt("./data/1d_x.txt", delimiter=",").reshape(-1, 1)
    y_train = np.loadtxt("./data/1d_y.txt", delimiter=",").reshape(-1, 1)
    num_train = len(y_train)
    batch_size = num_train // 3
    # Move the first third of the data to the left by 1
    x_train[:batch_size, :] -= 1
    # Move the second third of the data to the right by 1
    x_train[2 * batch_size : 3 * batch_size, :] += 1
    if shuffle:
        indices = np.random.permutation(x_train)
        x_train = x_train[indices, :]
        y_train = y_train[indices, :]
    x_test = np.linspace(-2.0, 12.0, 100)[:, None]
    # Normalize the data
    # x_min = -2.0
    # x_max = 12.0
    # x_train = 2.0 * ((x_train - x_min) / (x_max - x_min) - 0.5)
    # y_mean = np.mean(y_train)
    # y_std = np.std(y_train)
    # y_train = (y_train - y_mean) / y_std
    # x_test = np.linspace(-1.0, 1.0, 100)[:, None]
    return x_train, y_train, batch_size, x_test


def make_prediction(model, x_test, is_gpflow):
    if is_gpflow:
        mean, var = model.predict_f(x_test)
        mean = mean.numpy()
        var = var.numpy()
        x_pseudo = model.inducing_variable.Z
        f_pseudo, _ = model.predict_f(x_pseudo)
        x_pseudo = x_pseudo.numpy()
        f_pseudo = f_pseudo.numpy()
    else:
        mean, var = model.predict(x_test, include_noise=False)
        x_pseudo = model.x_pseudo
        f_pseudo, _ = model.predict(x_pseudo, include_noise=False)
    return mean, var, x_pseudo, f_pseudo
