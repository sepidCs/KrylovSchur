"""Matrix SVD and third-order tensor SVD using Fourier-domain slices.

Matrix functions return (U, S, V).
Matrix reconstruction uses U @ S @ V.conj().T. Tensor reconstruction uses
tProduct(U, tProduct(S, kilmer_transposs(V))).
"""

import warnings

import numpy as np
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from .krylov_schur import *
import torch 




def fftKilmer(A):
    """Transform along the third (tube) axis."""
    A = np.fft.fft(A, axis=2)
    return(A)


def ifftKilmer(A):
    """Invert the Fourier transform along the tube axis."""
    A = np.fft.ifft(A, axis=2)
    return(A)


def tProduct(A,B):
    """Multiply tensors by matrix multiplication of their Fourier slices.

    Intermediate factors can be complex even for a real input tensor. Keep
    their imaginary components; discard only numerical roundoff at the end.
    """
    Abar = fftKilmer(A)
    Bbar = fftKilmer(B)
    n1,ell,n3=A.shape
    n2=B.shape[1]
    Cbar = np.zeros((n1,n2,n3), dtype=np.complex128)
    for i in range(n3):
        Cbar[:,:,i]=np.dot(Abar[:,:,i],Bbar[:,:,i])
    C=ifftKilmer(Cbar)
    return np.real_if_close(C)


def our_svd(B,k, m, maxIt,tol):
    """Return rank-k factors using a Krylov-Schur Gram-matrix subspace.

    Requires 1 <= k < m <= min(B.shape). A small SVD inside the computed
    subspace recovers the factors without rounding eigenvalues or dividing by
    singular values. This also handles zero and rank-deficient inputs.
    """
    B = np.asarray(B)
    if B.ndim != 2:
        raise ValueError('B must be a two-dimensional matrix')
    n1,n2=B.shape
    if n1<n2:
        A=np.dot(B,B.conj().T)
        v1=np.random.rand(n1)
        Q, H, isC, flag, nc, ni = KrylovSchur(A, v1, n1, k, m, maxIt, tol)
    else:
        A=np.dot(B.conj().T,B)
        v1=np.random.rand(n2)
        Q, H, isC, flag, nc, ni = KrylovSchur(A, v1, n2, k, m, maxIt, tol)

    if flag:
        warnings.warn(
            f'Krylov-Schur did not converge after {ni} iterations '
            f'({nc} of {k} eigenvalues converged); returning an approximation',
            RuntimeWarning,
            stacklevel=2,
        )

    basis = Q[:, :k]
    if n1 < n2:
        small_u, sigma, vh = np.linalg.svd(basis.conj().T @ B, full_matrices=False)
        return basis @ small_u, np.diag(sigma), vh.conj().T
    u, sigma, small_vh = np.linalg.svd(B @ basis, full_matrices=False)
    return u, np.diag(sigma), basis @ small_vh.conj().T


def scipy_svd(B,k):
    """scikit-learn comparison; returns (U, S, VT), unlike our_svd."""
    svd = TruncatedSVD(k)
    X_transformed = svd.fit_transform(B)
    U = X_transformed / svd.singular_values_
    Sigma_matrix = np.diag(svd.singular_values_)
    VT = svd.components_
    return U,Sigma_matrix,VT


def tSVD(A,k):
    """Return tensor factors using PyTorch low-rank SVD on each FFT slice."""
    Abar = fftKilmer(A)
    n1,n2,n3=A.shape
    Ubar=np.zeros((n1,k ,n3), dtype=np.complex128)
    Sbar=np.zeros((k,k ,n3), dtype=np.complex128)
    Vbar=np.zeros((n2,k ,n3), dtype=np.complex128)

    for i in range(Abar.shape[-1]):
        u, singular_values, v = torch.svd_lowrank(torch.tensor(Abar[:,:,i]), q=k)
        # PyTorch can return a lazy conjugate view for V. Resolve it before
        # converting to NumPy instead of relying on implicit tensor conversion.
        Ubar[:, :, i] = u.resolve_conj().numpy()
        Sbar[:, :, i] = np.diag(singular_values.numpy())
        Vbar[:, :, i] = v.resolve_conj().numpy()

    U = ifftKilmer(Ubar)
    S = ifftKilmer(Sbar)
    Vt = ifftKilmer(Vbar)
    return U,S,Vt


def our_tSVD(A,k,m, maxIt,tol):
    """Return tensor factors using our_svd on each Fourier-domain slice."""
    Abar = fftKilmer(A)
    n1,n2,n3=A.shape
    Ubar=np.zeros((n1,k ,n3), dtype=np.complex128)
    Sbar=np.zeros((k,k ,n3))
    Vbar=np.zeros((n2,k ,n3), dtype=np.complex128)

    for i in range(Abar.shape[-1]):
        Ubar[:,:,i],Sbar[:,:,i],Vbar[:,:,i]= our_svd(Abar[:,:,i],k, m, maxIt,tol)
    U = ifftKilmer(Ubar)
    S = ifftKilmer(Sbar)
    Vt = ifftKilmer(Vbar)
    return U,S,Vt


def kilmer_transposs(A):
    """Tensor adjoint: conjugate-transpose slices and reverse nonfirst slices.

    In the Fourier domain, each slice becomes its conjugate transpose.
    """
    [n1, n2, n3] = A.shape
    Atrans = np.zeros((n2, n1, n3), dtype=np.complex128)
    Atrans[:, :, 0] = A[:, :, 0].conj().T
    for k in range(1, n3):
        Atrans[:, :, k] = A[:, :, n3-k].conj().T

    return Atrans
