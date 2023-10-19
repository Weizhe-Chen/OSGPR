import matplotlib.pyplot as plt
import numpy as np
import torch

from kernels import GaussianKernel
from osgpr import OSGPR
from utils import get_data, make_prediction, plot_result


def main():
    print("Running OSGPR...")
    np.random.seed(10)
    torch.manual_seed(10)

    init_length_scale = 0.8
    init_output_scale = 1.0
    init_noise_variance = 0.001
    num_inducing = 10

    x_train, y_train, batch_size, x_test = get_data()
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(7, 7))
    axes[0].set_title("OSGPR Prediction")
    for batch_index in range(3):
        x_batch = x_train[batch_index * batch_size : (batch_index + 1) * batch_size, :]
        y_batch = y_train[batch_index * batch_size : (batch_index + 1) * batch_size, :]

        if batch_index == 0:
            online_model = OSGPR(
                x_train=x_batch,
                y_train=y_batch,
                kernel=GaussianKernel(init_length_scale, init_output_scale),
                noise_variance=init_noise_variance,
                x_pseudo=x_batch[np.random.choice(batch_size, num_inducing), :],
            )
        else:
            online_model.update_data(x_batch, y_batch)

        print(f"Training on batch {batch_index}...")
        online_model.scipy_fit()

        mean, var, x_pseudo, f_pseudo = make_prediction(
            online_model, x_test, is_gpflow=False
        )

        plot_result(
            axes[batch_index],
            x_train,
            y_train,
            x_test,
            mean,
            var,
            online_model.noise_variance.item(),
            x_pseudo,
            f_pseudo,
            x_old=x_train[: batch_index * batch_size, :] if batch_index > 0 else None,
            y_old=y_train[: batch_index * batch_size, :] if batch_index > 0 else None,
        )

    # Train on the full dataset using SGPR
    print("Running SGPR on the full dataset for comparison...")
    batch_model = OSGPR(
        x_train=x_train,
        y_train=y_train,
        kernel=GaussianKernel(init_length_scale, init_output_scale),
        noise_variance=init_noise_variance,
        x_pseudo=x_train[np.random.choice(len(y_train), num_inducing), :],
    )
    batch_model.scipy_fit()
    mean, var, x_pseudo, f_pseudo = make_prediction(
        batch_model, x_test, is_gpflow=False
    )
    plot_result(
        axes[3],
        x_train,
        y_train,
        x_test,
        mean,
        var,
        batch_model.noise_variance.item(),
        x_pseudo,
        f_pseudo,
        plot_legend=True,
    )
    axes[3].set_xlabel("x")
    fig.tight_layout()

if __name__ == "__main__":
    main()
    plt.show()
