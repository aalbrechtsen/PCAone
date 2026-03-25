#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


BED_HEADER = bytes((0x6C, 0x1B, 0x01))
BED_MISSING = 0b01


def read_bim(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines()]


def read_fam_count(path: Path) -> int:
    return sum(1 for _ in path.open())


def flip_bed_row(row: bytes, nsamples: int) -> bytes:
    out = bytearray(row)
    for sample_idx in range(nsamples):
        byte_idx = sample_idx // 4
        shift = (sample_idx % 4) * 2
        genotype = (out[byte_idx] >> shift) & 0b11
        if genotype == 0b00:
            new_genotype = 0b11
        elif genotype == 0b11:
            new_genotype = 0b00
        else:
            new_genotype = genotype
        out[byte_idx] &= ~(0b11 << shift)
        out[byte_idx] |= new_genotype << shift
    return bytes(out)


def missing_bed_row(nsamples: int) -> bytes:
    row = bytearray((nsamples + 3) // 4)
    for sample_idx in range(nsamples):
        byte_idx = sample_idx // 4
        shift = (sample_idx % 4) * 2
        row[byte_idx] |= BED_MISSING << shift
    return bytes(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a PLINK dataset aligned to a reference BIM by inserting missing SNPs."
    )
    parser.add_argument("--reference-prefix", required=True, help="Reference PLINK prefix used for BIM layout.")
    parser.add_argument("--target-prefix", required=True, help="Target PLINK prefix to be padded/aligned.")
    parser.add_argument("--output-prefix", required=True, help="Output PLINK prefix.")
    args = parser.parse_args()

    ref_prefix = Path(args.reference_prefix)
    tgt_prefix = Path(args.target_prefix)
    out_prefix = Path(args.output_prefix)

    ref_bim = read_bim(ref_prefix.with_suffix(".bim"))
    tgt_bim = read_bim(tgt_prefix.with_suffix(".bim"))
    fam_path = tgt_prefix.with_suffix(".fam")
    nsamples = read_fam_count(fam_path)
    bytes_per_snp = (nsamples + 3) // 4

    with tgt_prefix.with_suffix(".bed").open("rb") as f:
        header = f.read(3)
        if header != BED_HEADER:
            raise RuntimeError("Target BED file does not have the expected PLINK SNP-major header.")
        tgt_rows = f.read()

    expected_size = len(tgt_bim) * bytes_per_snp
    if len(tgt_rows) != expected_size:
        raise RuntimeError(
            f"Target BED payload size mismatch: expected {expected_size} bytes, found {len(tgt_rows)}."
        )

    target_index: dict[str, tuple[int, bool]] = {}
    exact = 0
    flipped = 0
    for idx, row in enumerate(tgt_bim):
        target_index[row[1]] = (idx, False)

    for ref_row in ref_bim:
        snp_id = ref_row[1]
        if snp_id not in target_index:
            continue
        tgt_idx, _ = target_index[snp_id]
        tgt_row = tgt_bim[tgt_idx]
        same_pos = tgt_row[0] == ref_row[0] and tgt_row[3] == ref_row[3]
        same_alleles = tgt_row[4] == ref_row[4] and tgt_row[5] == ref_row[5]
        swapped = tgt_row[4] == ref_row[5] and tgt_row[5] == ref_row[4]
        if not same_pos or (not same_alleles and not swapped):
            raise RuntimeError(f"Cannot harmonize SNP {snp_id}: incompatible BIM rows.")
        if swapped:
            target_index[snp_id] = (tgt_idx, True)
            flipped += 1
        else:
            exact += 1

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fam_path, out_prefix.with_suffix(".fam"))
    out_prefix.with_suffix(".bim").write_text(
        "\n".join("\t".join(row) for row in ref_bim) + "\n"
    )

    missing_row = missing_bed_row(nsamples)
    inserted = 0
    with out_prefix.with_suffix(".bed").open("wb") as out:
        out.write(BED_HEADER)
        for ref_row in ref_bim:
            snp_id = ref_row[1]
            target_info = target_index.get(snp_id)
            if target_info is None:
                out.write(missing_row)
                inserted += 1
                continue
            tgt_idx, needs_flip = target_info
            start = tgt_idx * bytes_per_snp
            row = tgt_rows[start : start + bytes_per_snp]
            out.write(flip_bed_row(row, nsamples) if needs_flip else row)

    print(f"reference_snps={len(ref_bim)}")
    print(f"target_snps={len(tgt_bim)}")
    print(f"shared_exact={exact}")
    print(f"shared_flipped={flipped}")
    print(f"inserted_missing={inserted}")
    print(f"output_prefix={out_prefix}")


if __name__ == "__main__":
    main()
