from gpytorch_sgpr import SGPR
import numpy as np
import gpytorch
import torch
from matplotlib import pyplot as plt

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


train_x, train_y, batch_size, test_x = get_data(is_iid_data=False)
num_inducing = 10

batch_index = 0
x_batch = train_x[batch_index * batch_size : (batch_index + 1) * batch_size, :]
y_batch = train_y[batch_index * batch_size : (batch_index + 1) * batch_size, :]

kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
kernel.outputscale = 1.0
kernel.base_kernel.lengthscale = 0.8
z_array = train_x[np.random.permutation(x_batch.shape[0])[:num_inducing], :]
z_tensor = torch.from_numpy(z_array).double()
noise_variance = 0.001
model = SGPR(x_batch, y_batch, kernel, z_tensor, noise_variance=0.01)
model.fit_scipy(x_batch, y_batch)
# mean, var = model.predict(test_x)
# z_mean = model.z_mean

# fig, ax = plt.subplots()
# ax.plot(train_x, train_y, "kx")
# ax.plot(test_x, mean, "b")
# ax.fill_between(
#     test_x.squeeze(),
#     mean.squeeze() - 2 * np.sqrt(var.squeeze()),
#     mean.squeeze() + 2 * np.sqrt(var.squeeze()),
#     color="b",
#     alpha=0.2,
# )
# ax.plot(model.inducing_points, z_mean, "ro")
# ax.set_ylim([-3, 3])
# plt.show()

batch_index = 1
x_batch = train_x[batch_index * batch_size : (batch_index + 1) * batch_size, :]
y_batch = train_y[batch_index * batch_size : (batch_index + 1) * batch_size, :]

x_batch = torch.from_numpy(x_batch).double()
y_batch = torch.from_numpy(y_batch).double().squeeze()
test_x = torch.from_numpy(test_x).double()

def init_inducing_points(old_z, x_new, keep_rate=0.7):
    num_z = old_z.shape[0]
    num_old_z = int(keep_rate * num_z)
    num_new_z = num_z - num_old_z
    old_z = old_z[np.random.permutation(num_z)[:num_old_z], :]
    new_z = x_new[np.random.permutation(x_new.shape[0])[:num_new_z], :]
    new_z = np.vstack((old_z, new_z))
    return new_z

jitter = 1e-4


z = model.covar_module.inducing_points
z_pred = model(z)
z_mean = z_pred.mean.detach().reshape(-1, 1)
z_covar_mat = z_pred.covariance_matrix.detach()
z_kernel_mat = model.covar_module(z).evaluate().detach()

new_z = init_inducing_points(z.detach().numpy(), x_batch)
new_z = torch.from_numpy(new_z).double()


Kbs = model.covar_module(new_z, test_x).evaluate()
Kbb = model.covar_module(new_z).evaluate()
Kbb.diagonal().add_(jitter)
Kbf = model.covar_module(new_z, x_batch).evaluate()
Kba = model.covar_module(new_z, z).evaluate()

Lb = torch.linalg.cholesky(Kbb)

# Compute LD
sigma = torch.sqrt(model.likelihood.noise)
Lb = torch.linalg.cholesky(Kbb)
Lbinv_Kbf = torch.linalg.solve_triangular(Lb, Kbf, upper=False)
d1 = Lbinv_Kbf @ Lbinv_Kbf.T

LSa = torch.linalg.cholesky(z_covar_mat)
Lbinv_Kba = torch.linalg.solve_triangular(Lb, Kba, upper=False)
Kab_Lbinv = Lbinv_Kba.T
LSainv_Kab_Lbinv = torch.linalg.solve_triangular(LSa, Kab_Lbinv, upper=False)
d2 = LSainv_Kab_Lbinv.T @ LSainv_Kab_Lbinv

z_kernel_mat.diagonal().add_(jitter)
La = torch.linalg.cholesky(z_kernel_mat)
Lainv_Kab_Lbinv = torch.linalg.solve_triangular(La, Kab_Lbinv, upper=False)
d3 = Lainv_Kab_Lbinv.T @ Lainv_Kab_Lbinv

Mb = len(new_z)
D = torch.eye(Mb).double() + d1 + d2 - d3
D.diagonal().add_(jitter)
LD = torch.linalg.cholesky(D)

# Compute LDinv_Lbinv_c
Sinv_y = y_batch.unsqueeze(-1) / model.likelihood.noise
c1 = Kbf @ Sinv_y

Sainv_ma = torch.linalg.solve_triangular(LSa, z_mean, upper=False)
c2 = Kba @ Sainv_ma
c = c1 + c2
Lbinv_c = torch.linalg.solve_triangular(Lb, c, upper=False)
LDinv_Lbinv_c = torch.linalg.solve_triangular(LD, Lbinv_c, upper=False)

Lbinv_Kbs = torch.linalg.solve_triangular(Lb, Kbs, upper=False)
LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(LD, Lbinv_Kbs, upper=False)
mean = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_c
mean = mean.squeeze()

var1 = model.covar_module(test_x, diag=True)
var2 = -torch.sum(torch.square(Lbinv_Kbs), dim=0)
var3 = torch.sum(torch.square(LDinv_Lbinv_Kbs), dim=0)
var = var1 + var2 + var3

mean = mean.detach().numpy()
var = var.detach().numpy()
z_mean = z_mean.detach().numpy()

fig, ax = plt.subplots()
ax.plot(train_x, train_y, "kx")
ax.plot(test_x, mean, "b")
ax.fill_between(
    test_x.squeeze(),
    mean.squeeze() - 2 * np.sqrt(var.squeeze()),
    mean.squeeze() + 2 * np.sqrt(var.squeeze()),
    color="b",
    alpha=0.2,
)
ax.plot(new_z, z_mean, "ro")
ax.set_ylim([-3, 3])
plt.show()
