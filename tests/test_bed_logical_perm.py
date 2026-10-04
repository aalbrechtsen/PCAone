#!/usr/bin/env python3
"""--perm-mem: out-of-core BED winSVD read in interleaved -w bands straight from
the input, without a permuted copy. The PCs must equal those of a physically
reordered copy in the same order (run with -S), for any window size, chunk
(--perm-chunk), rotation (--perm-rotate) and --perm-adapt; the input must never
be written or removed; outputs stay in input order."""
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# reuse the fixture and the comparison of the random-band permutation test
source_text = (ROOT / 'tests' / 'test_bed_permutation.py').read_text()
helpers = source_text[:source_text.index('with tempfile.TemporaryDirectory')]
ns = {'__file__': str(ROOT / 'tests' / 'test_bed_permutation.py')}
exec(compile(helpers, 'test_bed_permutation.py', 'exec'), ns)
N, M, fixture, reordered, compare = ns['N'], ns['M'], ns['fixture'], ns['reordered'], ns['compare']


def interleave(bucket, W, c=1):
    """Chunk t (c source SNPs) to band t mod W; a full band passes its share on.
    Band b = logical SNPs [b * bucket, (b + 1) * bucket)."""
    start = [min(M, b * bucket) for b in range(W + 1)]
    fill, order, j, t = [0] * W, [None] * M, 0, 0
    while j < M:
        b, n = t % W, min(c, M - j)
        while n:
            while fill[b] == start[b + 1] - start[b]:
                b = (b + 1) % W
            take = min(n, start[b + 1] - start[b] - fill[b])
            for _ in range(take):
                order[start[b] + fill[b]] = j
                fill[b] += 1
                j += 1
            n -= take
        t += 1
    return order


with tempfile.TemporaryDirectory(prefix='pcaone-logical-') as name:
    directory = Path(name)
    source, bed, bim, fam = fixture(directory)
    inputs = {s: Path(str(source) + s).read_bytes() for s in ['.bed', '.bim', '.fam']}

    def pcaone(*args):
        return subprocess.run([str(ROOT / 'PCAone'), *map(str, args)], capture_output=True, text=True)

    def run(label, bfile=source, memory='0.000003', extra=(), verbose=3):
        prefix = directory / label
        result = pcaone('--bfile', bfile, '-m', memory, '-k', 2, '--oversamples', 2, '-n', 2, '-w', 4,
                        '--maxp', 5, '-v', verbose, '-V', '-o', prefix, *extra)
        assert result.returncode == 0, result.stdout + result.stderr
        for s in ['bed', 'bim', 'fam']:
            assert not Path(str(prefix) + '.perm.' + s).exists()
        return prefix, Path(str(prefix) + '.log').read_text()

    def logical_order(prefix):  # written at -v 3
        return [int(x) for x in Path(str(prefix) + '.perm.idx').read_text().split()]

    def bucket_of(log):
        blocksize, factor = map(int, re.search(
            r'after adjustment by PCAone: .*blocksize = (\d+) , nblocks = \d+ , factor = (\d+)', log).groups())
        return blocksize * factor

    for label, memory, extra in [('many', '0.000003', ()), ('few', '0.0002', ()),
                                 ('emu', '0.000003', ('--emu', '--maxiter', 2))]:
        results = []
        # whole BED in one window, and windows of one band (the next read in the background)
        for budget in ['1', '0.00000001']:
            prefix, log = run(f'{label}-{budget}', memory=memory, extra=('--perm-mem', budget, *extra))
            assert 'no .perm.bed' in log
            k = int(re.search(r'window = (\d+) of', log)[1])
            assert (k == 4) == (budget == '1')
            mbim = Path(str(prefix) + '.mbim').read_text().splitlines()
            assert [line.split()[:6] for line in mbim] == [line.split()[:6] for line in bim]
            results.append(prefix)
        order = interleave(bucket_of(log), 4)
        assert sorted(order) == list(range(M)) and logical_order(results[0]) == order
        copy = reordered(directory, label + '-interleaved', order, bed, bim, fam)
        physical, _ = run(label + '-physical', copy, memory=memory, extra=('-S', *extra))
        for prefix in results:
            compare(prefix, physical, order)

    # chunks and rotations: every band keeps its size, the PCs equal those of a
    # copy in the same order, and --perm-adapt (small, growing windows) changes nothing
    plain, log = run('plain', extra=('--perm-mem', '1'))
    bucket = bucket_of(log)
    orders = {}
    for label, extra in [('chunk3', ('--perm-chunk', 3)), ('rotate', ('--perm-rotate',)),
                         ('rotate-again', ('--perm-rotate',)), ('rotate-seed', ('--perm-rotate', '--seed', 7)),
                         ('chunk3-rotate', ('--perm-chunk', 3, '--perm-rotate'))]:
        prefix, _ = run(label, extra=('--perm-mem', '1', *extra))
        order = orders[label] = logical_order(prefix)
        assert sorted(order) == list(range(M))
        copy = reordered(directory, label + '-copy', order, bed, bim, fam)
        # --seed also draws winSVD's test matrix: the copy runs with the same seed
        seed = extra[extra.index('--seed'):extra.index('--seed') + 2] if '--seed' in extra else ()
        physical, _ = run(label + '-physical', copy, extra=('-S', *seed))
        compare(prefix, physical, order)
        small, _ = run(label + '-adapt', extra=('--perm-mem', '0.00000001', '--perm-adapt', *extra))
        for suffix in ['.eigvals', '.eigvecs', '.loadings']:
            assert Path(str(small) + suffix).read_bytes() == Path(str(prefix) + suffix).read_bytes(), suffix
    assert orders['chunk3'] == interleave(bucket, 4, 3) != logical_order(plain)
    assert orders['rotate'] == orders['rotate-again'] != orders['rotate-seed']
    assert orders['rotate'] != logical_order(plain)

    # the default cleanup (-v 1) removes <out>.perm.* only; the input stays
    run('cleanup', extra=('--perm-mem', '1'), verbose=1)
    for s, data in inputs.items():
        assert Path(str(source) + s).read_bytes() == data, s
    # ignored, with a warning, where it does not apply
    result = pcaone('-b', source, '-k', 2, '-n', 2, '--perm-mem', '1', '-o', directory / 'incore')
    assert result.returncode == 0 and 'perm-mem only applies' in result.stdout + result.stderr
print('BED logical permutation (--perm-mem, chunk, rotate, adapt): equals a reordered copy, input untouched')
