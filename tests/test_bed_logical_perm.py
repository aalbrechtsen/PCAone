#!/usr/bin/env python3
"""--perm-mem: out-of-core BED winSVD read in interleaved -w bands straight from
the input, without a permuted copy. The PCs must equal those of a physically
reordered copy in the same order (run with -S), for any window size; the input
must never be written or removed; outputs stay in input order."""
import importlib.util
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# reuse the fixture and the comparison of the random-band permutation test
spec = importlib.util.spec_from_file_location('bedperm', ROOT / 'tests' / 'test_bed_permutation.py')
source_text = (ROOT / 'tests' / 'test_bed_permutation.py').read_text()
helpers = source_text[:source_text.index('with tempfile.TemporaryDirectory')]
ns = {'__file__': str(ROOT / 'tests' / 'test_bed_permutation.py')}
exec(compile(helpers, 'test_bed_permutation.py', 'exec'), ns)
N, M, fixture, reordered, compare = ns['N'], ns['M'], ns['fixture'], ns['reordered'], ns['compare']


def interleave(bucket, W):
    """SNP j to band j mod W, skipping full bands; band b = [b * bucket, (b + 1) * bucket)."""
    start = [min(M, b * bucket) for b in range(W + 1)]
    fill, order, b = [0] * W, [None] * M, 0
    for j in range(M):
        while fill[b] == start[b + 1] - start[b]:
            b = (b + 1) % W
        order[start[b] + fill[b]] = j
        fill[b] += 1
        b = (b + 1) % W
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
        blocksize, factor = map(int, re.search(
            r'after adjustment by PCAone: .*blocksize = (\d+) , nblocks = \d+ , factor = (\d+)', log).groups())
        order = interleave(blocksize * factor, 4)
        assert sorted(order) == list(range(M))
        copy = reordered(directory, label + '-interleaved', order, bed, bim, fam)
        physical, _ = run(label + '-physical', copy, memory=memory, extra=('-S', *extra))
        for prefix in results:
            compare(prefix, physical, order)

    # the default cleanup (-v 1) removes <out>.perm.* only; the input stays
    run('cleanup', extra=('--perm-mem', '1'), verbose=1)
    for s, data in inputs.items():
        assert Path(str(source) + s).read_bytes() == data, s
    # ignored, with a warning, where it does not apply
    result = pcaone('-b', source, '-k', 2, '-n', 2, '--perm-mem', '1', '-o', directory / 'incore')
    assert result.returncode == 0 and 'perm-mem only applies' in result.stdout + result.stderr
print('BED logical permutation (--perm-mem): equals a reordered copy for any window, input untouched')
