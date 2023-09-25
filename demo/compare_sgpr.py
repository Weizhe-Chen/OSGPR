import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from gpflow_model import GPflowModel
from gpytorch_sgpr import SGPR


def get_data(is_iid_data):
    train_x = np.loadtxt("./data/1d_x_train.txt", delimiter=",").reshape(-1, 1)
    train_y = np.loadtxt("./data/1d_y_train.txt", delimiter=",").reshape(-1, 1)
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
    ax.yaxis.set_ticks(np.arange(-2, 3, 1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))


def main(is_iid_data, num_inducing=10):
    train_x, train_y, batch_size, test_x = get_data(is_iid_data)

    fig, axes = plt.subplots(3, 1)

    print("Running SGPR using GPyTorch...")
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    kernel.outputscale = 1.0
    kernel.base_kernel.lengthscale = 0.8
    z_array = train_x[np.random.permutation(train_x.shape[0])[:num_inducing], :]
    z_tensor = torch.from_numpy(z_array).double()
    noise_variance = 0.001
    model = SGPR(train_x, train_y, kernel, z_tensor, noise_variance)
    model.fit(train_x, train_y)
    mean_pt, var_pt = model.predict(test_x)

    plot_result(
        axes[0],
        train_x,
        train_y,
        None,
        None,
        test_x,
        mean_pt,
        var_pt,
        model.inducing_points,
        model.z_mean,
    )
    axes[0].set_ylabel("GPyTorch")

    print("Running SGPR using GPflow...")
    model = GPflowModel(num_inducing=num_inducing)
    model.fit(train_x, train_y)
    mean_tf, var_tf = model.predict(test_x)

    plot_result(
        axes[1],
        train_x,
        train_y,
        None,
        None,
        test_x,
        mean_tf,
        var_tf,
        model.inducing_points,
        model.z_mean,
    )
    axes[1].set_ylabel("GPflow")

    mean_diff = np.abs(mean_pt - mean_tf.ravel())
    var_diff = np.abs(var_pt - var_tf.ravel())

    axes[2].plot(test_x, mean_diff, "b", lw=2, label="Mean")
    axes[2].plot(test_x, var_diff, "r", lw=2, label="Variance")
    axes[2].set_xlim([np.min(test_x), np.max(test_x)])
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("Abs. Diff.")
    axes[2].legend()
    fig.subplots_adjust(hspace=0.08)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(10)
    main(is_iid_data=False)
