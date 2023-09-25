import torch
import gpytorch
from botorch.fit import fit_gpytorch_mll_scipy

class SGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, inducing_points, noise_variance):
        train_x = torch.from_numpy(train_x).double()
        train_y = torch.from_numpy(train_y).double().squeeze()
        inducing_points = torch.from_numpy(inducing_points).double()
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
