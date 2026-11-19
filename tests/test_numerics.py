"""Numerical checks against dense SVD and direct tensor convolution."""

import unittest
import numpy as np
from numpy.testing import assert_allclose
import torch

from krylov_tsvd.krylov_schur import KrylovSchur
from krylov_tsvd.tensor_svd import our_svd, our_tSVD, tSVD, tProduct, kilmer_transposs


class NumericalTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)

    def check_matrix(self, matrix, rank):
        u, s, v = our_svd(matrix, rank, min(matrix.shape), 100, 2e-16)
        self.assertEqual(u.shape, (matrix.shape[0], rank))
        self.assertEqual(s.shape, (rank, rank))
        self.assertEqual(v.shape, (matrix.shape[1], rank))
        for factor in (u, s, v):
            self.assertTrue(np.isfinite(factor).all())
        reference = np.linalg.svd(matrix, compute_uv=False)
        assert_allclose(np.diag(s), reference[:rank], rtol=1e-10, atol=1e-10)
        assert_allclose(u.conj().T @ u, np.eye(rank), rtol=0, atol=1e-10)
        assert_allclose(v.conj().T @ v, np.eye(rank), rtol=0, atol=1e-10)
        error = np.linalg.norm(matrix - u @ s @ v.conj().T)
        assert_allclose(error, np.linalg.norm(reference[rank:]), rtol=1e-9, atol=1e-10)

    def test_matrix_tall_and_wide(self):
        rng = np.random.RandomState(0)
        for shape in ((10, 7), (7, 10)):
            for rank in (4, 6):
                with self.subTest(shape=shape, rank=rank):
                    self.check_matrix(rng.rand(*shape), rank)

    def test_small_singular_values_are_not_rounded_to_zero(self):
        matrix = np.vstack((np.diag([3, 1, .4, .2, .05, .01, .001]), np.zeros((3, 7))))
        self.check_matrix(matrix, 6)

    def test_complex_matrix(self):
        rng = np.random.RandomState(0)
        for shape in ((10, 7), (7, 10)):
            with self.subTest(shape=shape):
                self.check_matrix(rng.rand(*shape) + 1j * rng.rand(*shape), 6)

    def test_zero_and_rank_deficient_matrix(self):
        rng = np.random.RandomState(0)
        for matrix in (np.zeros((10, 7)), rng.randn(10, 2) @ rng.randn(2, 7)):
            with self.subTest(norm=np.linalg.norm(matrix)):
                self.check_matrix(matrix, 6)

    def test_tensor_product_matches_direct_complex_convolution(self):
        rng = np.random.RandomState(0)
        a = rng.rand(3, 2, 3) + 1j * rng.rand(3, 2, 3)
        b = rng.rand(2, 4, 3) + 1j * rng.rand(2, 4, 3)
        expected = np.zeros((3, 4, 3), dtype=np.complex128)
        for t in range(3):
            for j in range(3):
                expected[:, :, t] += a[:, :, j] @ b[:, :, (t - j) % 3]
        assert_allclose(tProduct(a, b), expected, rtol=1e-12, atol=1e-12)

    def test_tensor_adjoint_in_fourier_domain(self):
        rng = np.random.RandomState(0)
        for depth in (2, 3, 4):
            with self.subTest(depth=depth):
                a = rng.rand(5, 3, depth) + 1j * rng.rand(5, 3, depth)
                expected = np.fft.fft(a, axis=2).conj().transpose(1, 0, 2)
                assert_allclose(np.fft.fft(kilmer_transposs(a), axis=2), expected, rtol=1e-12, atol=1e-12)

    def test_tensor_svd_against_per_slice_dense_svd(self):
        rng = np.random.RandomState(0)
        for depth, rank in ((2, 6), (3, 3), (4, 3)):
            with self.subTest(depth=depth, rank=rank):
                a = rng.rand(10, 7, depth)
                u, s, v = our_tSVD(a, rank, 7, 100, 2e-16)
                for factor in (u, s, v):
                    self.assertTrue(np.isfinite(factor).all())
                estimate = tProduct(u, tProduct(s, kilmer_transposs(v)))
                abar = np.fft.fft(a, axis=2)
                squared_error = sum(
                    np.linalg.norm(np.linalg.svd(abar[:, :, i], compute_uv=False)[rank:]) ** 2
                    for i in range(depth)
                ) / depth
                assert_allclose(np.linalg.norm(a - estimate), np.sqrt(squared_error), rtol=1e-9, atol=1e-10)

    def test_pytorch_tensor_svd_full_rank_reconstruction(self):
        a = np.random.RandomState(0).rand(6, 4, 3)
        u, s, v = tSVD(a, 4)
        assert_allclose(tProduct(u, tProduct(s, kilmer_transposs(v))), a, rtol=1e-10, atol=1e-10)

    def test_krylov_convergence_and_invariant_subspace(self):
        a = np.diag([10., 8., 6., 4., 2., 1., .5])
        q, h, is_complex, flag, converged, iterations = KrylovSchur(
            a, np.random.rand(7), 7, 3, 5, 100, 1e-12
        )
        self.assertEqual(flag, 0)
        self.assertEqual(converged, 3 + is_complex)
        self.assertLessEqual(iterations, 100)
        basis = q[:, :-1]
        assert_allclose(basis.conj().T @ basis, np.eye(basis.shape[1]), rtol=0, atol=1e-10)
        assert_allclose(a @ basis, q @ h, rtol=1e-10, atol=1e-10)
        assert_allclose(np.linalg.eigvals(h[:-1]), [10, 8, 6], rtol=1e-10, atol=1e-9)

    def test_krylov_zero_and_repeated_eigenvalues(self):
        for a in (np.zeros((7, 7)), np.eye(7)):
            with self.subTest(norm=np.linalg.norm(a)):
                q, h, _, flag, converged, _ = KrylovSchur(a, np.random.rand(7), 7, 3, 5, 100, 2e-16)
                self.assertEqual(flag, 0)
                self.assertEqual(converged, 3)
                assert_allclose(q[:, :-1].conj().T @ q[:, :-1], np.eye(3), rtol=0, atol=1e-10)
                assert_allclose(a @ q[:, :-1], q @ h, rtol=1e-10, atol=1e-10)

    def test_nonconvergence_is_reported(self):
        a = np.random.RandomState(0).rand(10, 7)
        with self.assertWarnsRegex(RuntimeWarning, "converg"):
            our_svd(a, 3, 5, 0, 2e-16)


if __name__ == "__main__":
    unittest.main()
