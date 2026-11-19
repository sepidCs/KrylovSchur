"""Run small matrix and tensor approximations with dense SVD references."""

import argparse

import numpy as np

from krylov_tsvd.tensor_svd import our_svd, our_tSVD, tProduct, kilmer_transposs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)

    matrix = np.random.rand(10, 7)
    u, s, v = our_svd(matrix, args.rank, 7, 100, 2e-16)
    error = np.linalg.norm(matrix - u @ s @ v.conj().T)
    reference = np.linalg.norm(np.linalg.svd(matrix, compute_uv=False)[args.rank:])
    print(f'Matrix (10, 7), rank {args.rank}')
    print(f'  reconstruction error: {error:.12f}')
    print(f'  dense SVD reference:  {reference:.12f}')

    tensor = np.random.rand(10, 7, 2)
    u, s, v = our_tSVD(tensor, args.rank, 7, 100, 2e-16)
    estimate = tProduct(u, tProduct(s, kilmer_transposs(v)))
    abar = np.fft.fft(tensor, axis=2)
    reference = np.sqrt(sum(
        np.linalg.norm(np.linalg.svd(abar[:, :, i], compute_uv=False)[args.rank:]) ** 2
        for i in range(2)
    ) / 2)
    print(f'Tensor (10, 7, 2), tubal rank {args.rank}')
    print(f'  reconstruction error: {np.linalg.norm(tensor - estimate):.12f}')
    print(f'  dense SVD reference:  {reference:.12f}')
    print('A finite, nonzero error is expected when a lower-rank approximation discards information.')


if __name__ == '__main__':
    main()
