from copy import deepcopy

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll_scipy

import gpytorch


class SGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, inducing_points, noise_variance):
        train_x = torch.tensor(train_x).double()
        train_y = torch.tensor(train_y).double().squeeze()
        inducing_points = torch.tensor(inducing_points).double()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise_variance
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            kernel, inducing_points=inducing_points, likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, train_x, train_y, num_steps=500):
        train_x = torch.from_numpy(train_x).double()
        train_y = torch.from_numpy(train_y).double().squeeze()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        parameters = [
            dict(params=self.likelihood.parameters(), lr=0.1),
            dict(params=self.covar_module.base_kernel.parameters(), lr=0.1),
            dict(params=self.covar_module.inducing_points, lr=0.5),
        ]
        optimizer = torch.optim.Adam(parameters)
        self.train()
        self.likelihood.train()
        for _ in range(num_steps):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        self.eval()
        self.likelihood.eval()

    def fit_scipy(self, train_x, train_y):
        train_x = torch.from_numpy(train_x).double()
        train_y = torch.from_numpy(train_y).double().squeeze()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        self.train()
        self.likelihood.train()
        fit_gpytorch_mll_scipy(mll)
        # fit_gpytorch_mll_torch(mll)
        self.eval()
        self.likelihood.eval()

    def predict_f(self, test_x, full_cov=False):
        test_x = torch.from_numpy(test_x).double()
        with torch.no_grad():
            pred_dist = self(test_x)
            mean = pred_dist.mean.numpy()
            if full_cov:
                var = pred_dist.covariance_matrix.numpy()
            else:
                var = pred_dist.variance.numpy()
        return mean, var

    @property
    def inducing_points(self):
        return self.covar_module.inducing_points.detach().cpu().numpy()

    @property
    def z_mean(self):
        z_mean, _ = self.predict(self.inducing_points)
        return z_mean

    @property
    def kernel(self):
        return self.covar_module.base_kernel


class OSGPR(gpytorch.models.GP):
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
        jitter=1e-4,
    ):
        super().__init__()
        self.train_x, self.Y = data
        self.train_x = torch.from_numpy(self.train_x).double()
        self.Y = torch.from_numpy(self.Y).double()
        self.kernel = kernel
        self.likelihood = likelihood
        self.inducing_variable = torch.from_numpy(Z).double()
        self.num_data = self.train_x.shape[0]
        self.jitter = jitter

        self.mu_old = torch.from_numpy(mu_old).double()
        self.M_old = old_z.shape[0]
        self.Su_old = torch.from_numpy(Su_old).double()
        self.Kaa_old = torch.from_numpy(Kaa_old).double()
        self.old_z = torch.from_numpy(old_z).double()

    @torch.no_grad()
    def predict_f(self, test_x, full_cov=False):
        test_x = torch.from_numpy(test_x).double()
        Kbs = self.kernel(self.inducing_variable, test_x).evaluate()
        Kbb = self.kernel(self.inducing_variable).evaluate()
        Kbb = Kbb + self.jitter * torch.eye(
            Kbb.shape[0], dtype=torch.double
        )
        Kbf = self.kernel(self.inducing_variable, self.train_x).evaluate()
        Kba = self.kernel(self.inducing_variable, self.old_z).evaluate()

        # Compute Lb
        Lb = torch.linalg.cholesky(Kbb)

        # Compute LD
        sigma = torch.sqrt(self.likelihood.noise)
        Lb = torch.linalg.cholesky(Kbb)
        Lbinv_Kba = torch.linalg.solve_triangular(Lb, Kba, upper=False)
        Lbinv_Kbf = torch.linalg.solve_triangular(Lb, Kbf, upper=False) / sigma
        d1 = torch.matmul(Lbinv_Kbf, Lbinv_Kbf.t())

        LSa = torch.linalg.cholesky(self.Su_old)
        Kab_Lbinv = torch.linalg.solve_triangular(LSa, Lbinv_Kba, upper=False)
        LSainv_Kab_Lbinv = torch.linalg.solve_triangular(
            LSa, Kab_Lbinv, upper=False
        )
        d2 = torch.matmul(LSainv_Kab_Lbinv.t(), LSainv_Kab_Lbinv)

        z_kernel_mat = self.Kaa_old + self.jitter * torch.eye(
            self.Kaa_old.shape[0], dtype=torch.double
        )
        La = torch.linalg.cholesky(z_kernel_mat)
        Lainv_Kab_Lbinv = torch.linalg.solve_triangular(La, Kab_Lbinv, upper=False)
        d3 = torch.matmul(Lainv_Kab_Lbinv.t(), Lainv_Kab_Lbinv)

        Mb = self.inducing_variable.shape[0]
        D = torch.eye(Mb, dtype=torch.double) + d1 + d2 - d3
        D = D + self.jitter * torch.eye(Mb, dtype=torch.double)
        LD = torch.linalg.cholesky(D)

        # Compute LDinv_Lbinv_c
        Sinv_y = self.Y / self.likelihood.noise
        c1 = torch.matmul(Kbf, Sinv_y)

        Sainv_ma = torch.linalg.solve(self.Su_old, self.mu_old)
        c2 = torch.matmul(Kba, Sainv_ma)
        c = c1 + c2
        Lbinv_c = torch.linalg.solve_triangular(Lb, c, upper=False)
        LDinv_Lbinv_c = torch.linalg.solve_triangular(LD, Lbinv_c, upper=False)

        Lbinv_Kbs = torch.linalg.solve_triangular(Lb, Kbs, upper=False)
        LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(LD, Lbinv_Kbs, upper=False)
        mean = torch.matmul(LDinv_Lbinv_Kbs.t(), LDinv_Lbinv_c)

        if full_cov:
            Kss = self.kernel(test_x).evaluate() + self.jitter * torch.eye(
                test_x.shape[0], dtype=torch.double
            )
            var1 = Kss
            var2 = -torch.matmul(Lbinv_Kbs.t(), Lbinv_Kbs)
            var3 = torch.matmul(LDinv_Lbinv_Kbs.t(), LDinv_Lbinv_Kbs)
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(test_x, diag=True)
            var2 = -torch.sum(torch.square(Lbinv_Kbs), axis=0)
            var3 = torch.sum(torch.square(LDinv_Lbinv_Kbs), axis=0)
            var = var1 + var2 + var3
        return mean, var

    def covariance_matrix(self, x):
        x = torch.from_numpy(x).double()
        with torch.no_grad():
            covar_mat = self.kernel(x).evaluate()
        return covar_mat.numpy()


class GPyTorchModel:
    def __init__(
        self,
        num_inducing,
        noise_variance=0.001,
    ):
        kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        kernel.outputscale = 1.0
        kernel.base_kernel.lengthscale = 0.8
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
            self.model = SGPR(
                train_x,
                train_y,
                self.kernel,
                inducing_points=new_z,
                noise_variance=self.noise_variance,
            )
            self.model.fit_scipy(train_x, train_y)
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
        self.summarize()

    def predict(self, test_x):
        test_mean, test_var = self.model.predict_f(test_x)
        if len(test_var.shape) == 3:
            test_var = test_var[:, 0]
        return test_mean, test_var

    @property
    def inducing_points(self):
        if isinstance(self.model, SGPR):
            return self.model.covar_module.inducing_points.detach().numpy()
        else:
            return self.model.inducing_variable.detach().numpy()

    def summarize(self):
        self.z = self.inducing_points
        self.z_mean, self.z_covar_mat = self.model.predict_f(self.z, full_cov=True)
        with torch.no_grad():
            self.z_kernel_mat = self.model.kernel(torch.from_numpy(self.z).double()).numpy() 

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
