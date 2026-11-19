"""Compare matrix/tensor SVD accuracy and CPU time.

Run: python -m experiments.compare --suite all --repeats 5
Outputs: results/metrics.csv, results/metrics.json, and comparison plots.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import matplotlib
matplotlib.use('Agg')  # Save figures without opening a graphical window.
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
import torch

from krylov_tsvd.tensor_svd import (
    our_svd, our_tSVD, scipy_svd, tSVD, tProduct, kilmer_transposs,
)


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple
    rank: int
    restart: int
    group: str
    max_iterations: int = 150


# Scaling cases hold q/p=10 and k/p=0.2 fixed; depth cases change only axis 2.
SCALING = [
    Case(f'matrix-{p}', (10*p, p), p//5, 2*p//5, 'matrix-size')
    for p in (25, 50, 100, 200)
] + [
    Case(f'tensor-{p}', (10*p, p, 4), p//5, 2*p//5, 'tensor-size')
    for p in (25, 50, 100)
] + [
    Case(f'depth-{depth}', (500, 50, depth), 10, 20, 'tensor-depth')
    for depth in (2, 8)
]

# Small and large cases exercise matrix/tensor approximation at several ranks.
ORIGINAL = [
    Case('matrix-large', (1000, 100), 30, 60, 'original'),
    Case('matrix-small', (10, 7), 4, 7, 'original'),
    Case('matrix-high-rank', (10, 7), 6, 7, 'original'),
    Case('tensor-large', (1000, 100, 10), 30, 60, 'original'),
    Case('tensor-small', (10, 7, 2), 3, 5, 'original', 100),
    Case('tensor-high-rank', (10, 7, 2), 6, 7, 'original', 100),
]
COLORS = {'Dense SVD': '#6f7d8c', 'PyTorch': '#d28b25',
          'Krylov-Schur': '#136f83', 'scikit-learn': '#9568ae'}


def dense_svd(matrix, rank):
    """Full dense SVD, truncated to rank k: the accuracy reference."""
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :rank], np.diag(s[:rank]), vh[:rank].conj().T


def decompose(data, case, method):
    """Use the same rank and factor convention (U, S, V) for every method."""
    k = case.rank
    if data.ndim == 2:
        if method == 'Dense SVD':
            return dense_svd(data, k)
        if method == 'scikit-learn':
            u, s, vh = scipy_svd(data, k)
            return u, s, vh.conj().T
        if method == 'PyTorch':
            u, s, v = torch.svd_lowrank(torch.tensor(data), q=k)
            return u.resolve_conj().numpy(), np.diag(s.numpy()), v.resolve_conj().numpy()
        return our_svd(data, k, case.restart, case.max_iterations, 2e-16)
    if method == 'Krylov-Schur':
        return our_tSVD(data, k, case.restart, case.max_iterations, 2e-16)
    if method == 'PyTorch':
        return tSVD(data, k)

    # Dense t-SVD applies an ordinary dense SVD to every Fourier slice.
    transformed = np.fft.fft(data, axis=2)
    u = np.empty((data.shape[0], k, data.shape[2]), dtype=np.complex128)
    s = np.empty((k, k, data.shape[2]), dtype=np.complex128)
    v = np.empty((data.shape[1], k, data.shape[2]), dtype=np.complex128)
    for i in range(data.shape[2]):
        u[:, :, i], s[:, :, i], v[:, :, i] = dense_svd(transformed[:, :, i], k)
    return tuple(np.fft.ifft(factor, axis=2) for factor in (u, s, v))


def benchmark(case, repeats, seed):
    """Time factorization only; input generation, seeds, and errors are outside."""
    data = np.random.RandomState(seed).rand(*case.shape)
    methods = ['Dense SVD', 'PyTorch', 'Krylov-Schur']
    if data.ndim == 2:
        methods.append('scikit-learn')
    records = []
    for method in methods:
        times = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for repeat in range(repeats + 1):
                # Identical inputs and RNG state separate runtime variation from randomness.
                np.random.seed(seed + 1)
                torch.manual_seed(seed + 1)
                start = perf_counter()
                u, s, v = decompose(data, case, method)
                elapsed = perf_counter() - start
                if repeat:  # Discard one warm-up for each method and input size.
                    times.append(elapsed)

        if not all(np.isfinite(factor).all() for factor in (u, s, v)):
            raise ArithmeticError(f'{case.name}: {method} returned nonfinite factors')
        if data.ndim == 2:
            approximation = u @ s @ v.conj().T
        else:
            approximation = tProduct(u, tProduct(s, kilmer_transposs(v)))
        absolute_error = float(np.linalg.norm(data - approximation))
        record = {
            'case': case.name, 'group': case.group,
            'shape': 'x'.join(map(str, case.shape)), 'rank': case.rank,
            'restart': case.restart, 'method': method,
            'median_ms': 1000 * float(np.median(times)),
            'min_ms': 1000 * min(times), 'max_ms': 1000 * max(times),
            'absolute_error': absolute_error,
            'relative_error': absolute_error / float(np.linalg.norm(data)),
            'warnings': sorted(set(str(w.message) for w in caught)),
            'samples_ms': [1000 * value for value in times],
        }
        records.append(record)
        print(f'{case.name:22s} {method:13s} {record["median_ms"]:10.3f} ms'
              f'  relative error {record["relative_error"]:.6%}', flush=True)
    reference = records[0]['relative_error']
    for record in records:
        record['error_over_dense'] = record['relative_error'] / reference
    return records


def save_plot(records, output, groups, filename, compact=False):
    """Show runtime and approximation error at the same dimension and rank."""
    if compact:
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
        axes = axes.reshape(2, 1)
    else:
        figure, axes = plt.subplots(2, len(groups), figsize=(11, 6.3), squeeze=False)
    for column, (title, chosen) in enumerate(groups):
        if not chosen:
            continue
        names = [case.name for case in chosen]
        labels = ['x'.join(map(str, case.shape)) + f'\nk={case.rank}' for case in chosen]
        methods = [method for method in COLORS if any(
            row['method'] == method and row['case'] in names for row in records)]
        for method in methods:
            rows = [next(row for row in records if row['case'] == name
                         and row['method'] == method) for name in names]
            for axis, key, multiplier in ((axes[0, column], 'median_ms', 1),
                                          (axes[1, column], 'relative_error', 100)):
                axis.plot(range(len(names)), [row[key]*multiplier for row in rows],
                          marker='o', markersize=4, linewidth=1.6,
                          color=COLORS[method], label=method)
        axes[0, column].set_title(title, weight='bold', pad=10, fontsize=9 if compact else 12)
        axes[0, column].set_yscale('log')
        for axis in axes[:, column]:
            axis.set_xticks(range(len(names)), labels, fontsize=7 if compact else 8)
            axis.tick_params(axis='y', labelsize=8 if compact else 10)
            axis.grid(alpha=.2)
            axis.spines[['top', 'right']].set_visible(False)
        axes[0, column].set_ylabel('Time (ms, log scale)' if compact else 'Factorization time (ms, log scale)', fontsize=8 if compact else 10)
        axes[1, column].set_ylabel('Relative reconstruction error (%)', fontsize=8 if compact else 10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=len(labels),
                  frameon=False, bbox_to_anchor=(.5, 1.01), fontsize=8 if compact else 10)
    figure.tight_layout(rect=(0, 0, 1, .94), h_pad=2.0)
    for extension in ('png', 'svg'):
        metadata = {'Date': None} if extension == 'svg' else None
        figure.savefig(output / f'{filename}.{extension}', dpi=240,
                       bbox_inches='tight', metadata=metadata)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('quick', 'original', 'scaling', 'all'), default='all')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be at least 1')
    suites = {'quick': [ORIGINAL[1], ORIGINAL[4]], 'original': ORIGINAL,
              'scaling': SCALING, 'all': SCALING + ORIGINAL}
    cases = suites[args.suite]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):  # One CPU thread makes the comparison controlled.
        records = [row for case in cases for row in benchmark(case, args.repeats, args.seed)]
        pools = [{key: pool.get(key) for key in ('internal_api', 'num_threads', 'version')}
                 for pool in threadpool_info()]
    metadata = {
        'seed': args.seed, 'repeats': args.repeats, 'warmups': 1, 'suite': args.suite,
        'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(), 'torch_threads': 1, 'thread_pools': pools,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'scipy', 'scikit-learn', 'torch', 'matplotlib')},
        'cases': [asdict(case) for case in cases],
        'settings': {'krylov_tolerance': 2e-16, 'pytorch_q': 'rank',
                     'pytorch_power_iterations': 2, 'sklearn_power_iterations': 5},
        'timing_scope': 'factorization including Gram/FFT/factor conversion; excludes generation, RNG resets, reconstruction, errors, imports, plots',
    }
    (args.output / 'metrics.json').write_text(
        json.dumps({'metadata': metadata, 'results': records}, indent=2), encoding='utf-8')
    keys = [key for key in records[0] if key not in ('warnings', 'samples_ms')]
    with (args.output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in records)

    matrix = [case for case in cases if case.group == 'matrix-size']
    tensor = [case for case in cases if case.group == 'tensor-size']
    if not matrix:
        matrix = [case for case in cases if len(case.shape) == 2]
    if not tensor:
        tensor = [case for case in cases if len(case.shape) == 3]
    save_plot(records, args.output,
              [('Matrices', matrix), ('Tensors (depth 4)' if args.suite in ('all', 'scaling')
                                      else 'Tensors', tensor)], 'comparison')
    if args.suite in ('all', 'scaling'):
        save_plot(records, args.output, [('Tensor size sweep (depth 4)', tensor)],
                  'tensor_comparison', compact=True)
    depth = sorted([case for case in cases if len(case.shape) == 3
                    and case.shape[:2] == (500, 50)], key=lambda case: case.shape[2])
    if depth:
        save_plot(records, args.output, [('Changing only tensor depth', depth)], 'depth')
    print(f'\nSaved measurements and plots to {args.output.resolve()}')


if __name__ == '__main__':
    main()
