import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import gpflow

from gpflow_model import GPflowModel


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
    batch_model = GPflowModel(
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
