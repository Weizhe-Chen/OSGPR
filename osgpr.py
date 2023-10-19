import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
from pytorch_minimize.optim import MinimizeWrapper
from torch.nn.parameter import Parameter
from tqdm import tqdm

import utils


class OSGPR(torch.nn.Module):
    def __init__(self, x_train, y_train, kernel, noise_variance, x_pseudo, jitter=1e-4):
        super().__init__()
        self.is_sgpr = True
        self.kernel = kernel
        self.free_noise_variance = Parameter(
            utils.constraint.inv_transform(utils.to_tensor(noise_variance))
        )
        self.train_x = utils.to_tensor(x_train)
        self.train_y = utils.to_tensor(y_train)
        self.pseudo_x = Parameter(utils.to_tensor(x_pseudo))
        self.num_pseudo = self.pseudo_x.size(0)
        self.jitter = jitter

    def update_data(self, x_train, y_train):
        self.summarize()
        self.train_x = utils.to_tensor(x_train)
        self.train_y = utils.to_tensor(y_train)
        self.update_pseudo_x()
        self.is_sgpr = False

    def summarize(self):
        pseudo_x = self.pseudo_x.data.clone()
        m, S = self.forward(pseudo_x, diag=False)
        K = self.kernel(pseudo_x)
        K.diagonal().add_(self.jitter)  # Debug
        self.old_pseudo_x = pseudo_x.detach()
        self.old_m = m.detach()
        self.old_S = S.detach()
        self.old_K = K.detach()

    def update_pseudo_x(self, keep_rate=0.7):
        # self.pseudo_x = Parameter(utils.to_tensor(np.linspace(-2, 12, self.num_pseudo).reshape(-1, 1)))
        num_kept = int(keep_rate * self.num_pseudo)
        num_added = self.num_pseudo - num_kept
        num_train = self.train_x.size(0)
        kept_indices = np.random.permutation(self.num_pseudo)[:num_kept]
        kept_pseudo_x = self.old_pseudo_x[kept_indices, :]
        added_indices = np.random.permutation(num_train)[:num_added]
        added_pseudo_x = self.train_x[added_indices, :]
        new_pseudo_x = torch.cat([kept_pseudo_x, added_pseudo_x], dim=0)
        self.pseudo_x = Parameter(new_pseudo_x)

    @property
    @torch.no_grad()
    def x_pseudo(self):
        return utils.to_array(self.pseudo_x)

    @property
    def noise_variance(self):
        var = utils.constraint.transform(self.free_noise_variance)
        return var

    @noise_variance.setter
    def noise_variance(self, value):
        self.free_noise_variance = Parameter(
            utils.constraint.inv_transform(utils.to_tensor(value))
        )

    def scipy_fit(self):
        self.train()
        minimizer_args = dict(
            method="L-BFGS-B", options={"disp": False, "maxiter": 100}
        )
        optimizer = MinimizeWrapper(self.parameters(), minimizer_args)

        def closure():
            optimizer.zero_grad()
            loss = -self.evidence()
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()

    def fit(self, num_steps, num_restarts=3, lr=0.01, verbose=True):
        self.train()
        losses = []
        for index_restart in range(num_restarts):
            print(f"Restart: {index_restart}")
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            progress = range(num_steps)
            if verbose:
                progress = tqdm(progress)
            for i in progress:
                optimizer.zero_grad()
                loss = -self.evidence()
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                if verbose:
                    progress.set_description(
                        f"Iter: {i:04d} " + f"Loss: {loss.item():.4f}"
                    )
        self.eval()
        return losses

    @torch.no_grad()
    def predict(self, test_x, diag=True, include_noise=True):
        test_x = utils.to_tensor(test_x)
        if self.is_sgpr:
            mean, covar = self.sgpr_forward(test_x, diag=diag)
        else:
            mean, covar = self.forward(test_x, diag=diag)
        if diag and include_noise:
            covar += self.noise_variance
        return utils.to_array(mean), utils.to_array(covar)

    def common_terms(self, Kvu, Kvf, Kvv):
        TrilKvv = utils.compute_tril(Kvv, self.jitter)
        TrilSuu = utils.compute_tril(self.old_S, self.jitter)
        TrilOldKuu = utils.compute_tril(self.old_K, self.jitter)

        InvTrilKvv_Kvf = utils.tril_solve(TrilKvv, Kvf)
        D1 = (InvTrilKvv_Kvf @ InvTrilKvv_Kvf.T).div(self.noise_variance)
        InvTrilKvv_Kvu = utils.tril_solve(TrilKvv, Kvu)
        Kuv_InvTrilKvvT = InvTrilKvv_Kvu.T
        InvTrilSuu_Kuv_InvTrilKvvT = utils.tril_solve(TrilSuu, Kuv_InvTrilKvvT)
        D2 = InvTrilSuu_Kuv_InvTrilKvvT.T @ InvTrilSuu_Kuv_InvTrilKvvT
        InvTrilOldKuu_Kuv_InvTrilKvvT = utils.tril_solve(TrilOldKuu, Kuv_InvTrilKvvT)
        D3 = InvTrilOldKuu_Kuv_InvTrilKvvT.T @ InvTrilOldKuu_Kuv_InvTrilKvvT
        D = torch.eye(D1.shape[0]).to(D1) + D1 + D2 - D3  # Debug
        D.diagonal().add_(self.jitter)  # Debug
        TrilD = utils.compute_tril(D, self.jitter)

        InvTrilSuu_Kuv = utils.tril_solve(TrilSuu, Kvu.T)
        InvTrilSuu_mu = utils.tril_solve(TrilSuu, self.old_m)
        c1 = (Kvf @ self.train_y).div(self.noise_variance)
        c2 = InvTrilSuu_Kuv.T @ InvTrilSuu_mu
        c = c1 + c2
        return c, TrilKvv, TrilD, TrilSuu, TrilOldKuu

    def forward(self, test_x, diag=True):
        if self.is_sgpr:
            return self.sgpr_forward(test_x, diag=diag)

        # Compute kernel matrics
        Kvs = self.kernel(self.pseudo_x, test_x)  # O(MS)
        Kvv = self.kernel(self.pseudo_x)  # O(M^2)
        Kvv.diagonal().add_(self.jitter)  # Debug
        Kvf = self.kernel(self.pseudo_x, self.train_x)  # O(MN)
        Kvu = self.kernel(self.pseudo_x, self.old_pseudo_x)  # O(MM)

        c, TrilKvv, TrilD, _, _ = self.common_terms(Kvu, Kvf, Kvv)

        # Prepare common terms for computing mean and (co)variance
        InvTrilKvv_Kvs = utils.tril_solve(TrilKvv, Kvs)
        InvTrilD_InvTrilKvv_Kvs = utils.tril_solve(TrilD, InvTrilKvv_Kvs)
        InvTrilKvv_c = utils.tril_solve(TrilKvv, c)
        InvTrilD_InvTrilKvv_c = utils.tril_solve(TrilD, InvTrilKvv_c)

        mean = InvTrilD_InvTrilKvv_Kvs.T @ InvTrilD_InvTrilKvv_c

        if diag:
            kss = self.kernel(test_x, diag=True)
            var = (
                kss
                - utils.tril_diag_quadratic(InvTrilKvv_Kvs)
                + utils.tril_diag_quadratic(InvTrilD_InvTrilKvv_Kvs)
            )
            return mean, var
        else:
            Kss = self.kernel(test_x)
            cov = (
                Kss
                - InvTrilKvv_Kvs.T @ InvTrilKvv_Kvs
                + InvTrilD_InvTrilKvv_Kvs.T @ InvTrilD_InvTrilKvv_Kvs
            )
            return mean, cov

    def evidence(self):
        if self.is_sgpr:
            return self.sgpr_evidence()

        num_train = self.train_x.size(0)

        # Compute kernel matrics
        Kvv = self.kernel(self.pseudo_x)  # O(M^2)
        Kvv.diagonal().add_(self.jitter)  # Debug
        Kvf = self.kernel(self.pseudo_x, self.train_x)  # O(MN)
        Kvu = self.kernel(self.pseudo_x, self.old_pseudo_x)  # O(MM)
        Kuu = self.kernel(self.old_pseudo_x)  # O(M^2)
        Kuu.diagonal().add_(self.jitter)  # Debug
        kff = self.kernel(self.train_x, diag=True)  # O(N)

        c, TrilKvv, TrilD, TrilSuu, TrilOldKuu = self.common_terms(Kvu, Kvf, Kvv)

        InvTrilKvv_c = utils.tril_solve(TrilKvv, c)
        InvTrilD_InvTrilKvv_c = utils.tril_solve(TrilD, InvTrilKvv_c)
        InvTrilSuu_mu = utils.tril_solve(TrilSuu, self.old_m)

        # Prepare for computing trace terms
        InvTrilKvv_Kvu = utils.tril_solve(TrilKvv, Kvu)
        Quu = InvTrilKvv_Kvu.T @ InvTrilKvv_Kvu
        Euu = Kuu - Quu
        InvSuu_Euu = torch.cholesky_solve(Euu, TrilSuu)
        InvOldKuu_Euu = torch.cholesky_solve(Euu, TrilOldKuu)
        InvTrilKvv_Kvf = utils.tril_solve(TrilKvv, Kvf)

        constant_term = -num_train * np.log(2 * np.pi)
        quadratic_terms = (
            -(self.train_y.square().sum().div(self.noise_variance))
            + InvTrilD_InvTrilKvv_c.square().sum()
            - InvTrilSuu_mu.square().sum()
        )
        log_terms = (
            -utils.tril_logdet(TrilD)
            - utils.tril_logdet(TrilSuu)
            + utils.tril_logdet(TrilOldKuu)
            - num_train * torch.log(self.noise_variance)
        )
        trace_terms = (
            -InvSuu_Euu.trace()
            + InvOldKuu_Euu.trace()
            - kff.sum().div(self.noise_variance)
            + utils.trace_quadratic(InvTrilKvv_Kvf).div(self.noise_variance)
        )
        elbo = 0.5 * (constant_term + quadratic_terms + log_terms + trace_terms)
        return elbo

    def sgpr_common_terms(self):
        sigma = self.noise_variance.sqrt()
        Kuu = self.kernel(self.pseudo_x)  # O(M^2)
        Kuf = self.kernel(self.pseudo_x, self.train_x)  # O(MN)
        Lk = utils.compute_tril(Kuu, self.jitter)  # O(M^3)
        # A = sigma^{-1} Lk^{-1} Kuf
        A = utils.tril_solve(Lk, Kuf).div(sigma)  # O(M^2 N)
        # B = I + A A.T
        B = torch.eye(self.num_pseudo).to(A) + A @ A.t()  # O(M^2 N)
        Lb = utils.compute_tril(B, base_jitter=0.0)  # O(M^3)
        # c = sigma^{-1} Lk^{-1} Kuf y -> shape (M, 1)
        c = utils.tril_solve(Lb, A @ self.train_y).div(sigma)  # O(M^2 N)
        return Kuu, Kuf, Lk, A, B, Lb, c

    def sgpr_forward(self, test_x, diag=True):
        _, _, Lk, _, _, Lb, c = self.sgpr_common_terms()
        Ksu = self.kernel(test_x, self.pseudo_x)  # O(SM)
        # Lk^{-1} Ku* -> shape (M, S)
        invLk_Kus = utils.tril_solve(Lk, Ksu.t())  # O(SM^2)
        # Lb^{-1} Lk^{-1} Ku* -> shape (M, S)
        invLb_invLk_Kus = utils.tril_solve(Lb, invLk_Kus)  # O(SM^2)
        # K*u Lk^{-T} Lb^{-T} c -> shape (S, 1)
        mean = invLb_invLk_Kus.t() @ c  # O(SM)
        if diag:
            kss = self.kernel(test_x, diag=True)  # O(S)
            var = (
                kss
                - utils.tril_diag_quadratic(invLk_Kus)  # O(SM)
                + utils.tril_diag_quadratic(invLb_invLk_Kus)  # O(SM)
            )
            return mean, var
        else:
            Kss = self.kernel(test_x)  # O(S^2)
            cov = (
                Kss
                - invLk_Kus.t() @ invLk_Kus  # O(SM^2)
                + invLb_invLk_Kus.t() @ invLb_invLk_Kus  # O(SM^2)
            )
            return mean, cov

    def sgpr_evidence(self):
        num_train = self.train_x.size(0)
        _, _, _, A, _, Lb, c = self.sgpr_common_terms()
        return 0.5 * (
            # -N log(2pi)
            -num_train * np.log(2 * np.pi)
            # -log|B|
            - utils.tril_logdet(Lb)
            # -N log(sigma^{2})
            - num_train * torch.log(self.noise_variance)
            # -sigma^{-2} y.T y
            - self.train_y.square().sum().div(self.noise_variance)
            # +sigma^{-2} y.T A.T B^{-1} A y = sigma^{-2} c.T c
            + c.square().sum()
            # -sigma^{-2} trace(Kff)
            - self.kernel(self.train_x, diag=True).sum().div(self.noise_variance)
            # + trace(A A.T)
            + utils.trace_quadratic(A)
        )
