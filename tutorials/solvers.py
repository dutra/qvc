# cg_solver.py

from typing import Callable
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax.scipy.sparse.linalg import cg
from tinygp.solvers.solver import Solver
from tinygp.kernels.base import Kernel
from typing import Any

from tinygp import kernels
from tinygp.helpers import JAXArray
from tinygp.noise import Noise


class DirectFullRank(Solver):

    X: JAXArray
    variance_value: JAXArray
    covariance_value: JAXArray

    def __init__(
        self,
        kernel: kernels.Kernel,
        X: JAXArray,
        noise: Noise,
        *,
        covariance: Any | None = None,
        ):
        # Initialize the kernel, data, and other hyperparameters
        self.X = X
        self.variance_value = None
        if covariance is None:
            jax.debug.print("shape {}", kernel(X, X).shape)
            covariance = kernel(X, X) + noise
        self.covariance_value = jnp.asarray(covariance)
        self.variance_value = jnp.diag(self.covariance_value)
        #jax.debug.print("covariance_value {s}", s=self.covariance_value)


    def condition(
        self, kernel: kernels.Kernel, X_test: jnp.ndarray | None, noise: Any
    ) -> Any:
        """Compute the covariance matrix for a conditional GP

        Args:
            kernel: The kernel for the covariance between the observed and
                predicted data.
            X_test: The coordinates of the predicted points. Defaults to the
                input coordinates.
            noise: The noise model for the predicted process.
        """
        if X_test is None:
            Ks = kernel(self.X, self.X)
            Kss = Ks + noise
        else:
            Ks = kernel(self.X, X_test)
            Kss = kernel(X_test, X_test) + noise

        A = self.solve_triangular(Ks)
        return Kss - A.transpose() @ A

    def variance(self) -> JAXArray:
        return self.variance_value

    def covariance(self) -> JAXArray:
        return self.covariance_value

    def normalization(self) -> JAXArray:
        sign, logdet = jnp.linalg.slogdet(self.covariance_value)
        return 0.5 * logdet + 0.5 * self.covariance_value.shape[0] * jnp.log(2 * jnp.pi)

    def solve_triangular(self, y: JAXArray, *, transpose: bool = False) -> JAXArray:
        if transpose:
            return jax.scipy.linalg.solve(self.covariance_value.T, y, assume_a="gen")
        else:
            return jax.scipy.linalg.solve(self.covariance_value, y, assume_a="gen")

    def dot_triangular(self, y: JAXArray) -> JAXArray:
        return jnp.dot(self.covariance_value, y)
