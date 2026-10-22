"""Command line interface for training and exporting LP tokenizers."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and export an LP tokenizer")
    p.add_argument("--data", required=True, type=Path, help="UTF-8 text file, saved Dataset, or Parquet file/directory")
    p.add_argument("--vocab-size", required=True, type=int, action="append", dest="vocab_sizes", metavar="K")
    p.add_argument("--output-dir", type=Path, default=Path("lp-tokeniser-output"))
    p.add_argument("--text-column", help="Text column in a saved Dataset or Parquet input")
    p.add_argument("--pretokenizer", choices=("nanochat", "apertus", "split_bytelevel", "pythia"), default="nanochat")
    p.add_argument("--training-mode", choices=("standard", "boundless", "super"), default="standard")
    p.add_argument("--rounding", choices=("det", "bias", "all_ones"), default="det")
    p.add_argument("--max-token-bytes", type=int)
    p.add_argument("--super-base-vocab-size", type=int)
    p.add_argument("--pently-rho", type=float, default=0.0)
    p.add_argument("--celex-dir", type=Path)
    p.add_argument("--vocab-utilisation-weight", type=float, default=0.0)
    p.add_argument("--compression-divisor", type=float, action="append", dest="compression_divisors",
                   help="Repeat once per --vocab-size when using utilization weighting")
    p.add_argument("--num-proc", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--num-gpus", type=int, default=4,
                   help="GPUs used together by each PDLP solve (default: 4)")
    return p


def validate(args: argparse.Namespace) -> None:
    from lp_tokenizer.document_tokenizer import TrainingOptions

    if not args.data.exists():
        raise ValueError(f"Data path does not exist: {args.data}")
    if not args.vocab_sizes or any(k <= 0 for k in args.vocab_sizes):
        raise ValueError("--vocab-size must be a positive integer")
    for name in ("num_proc", "batch_size", "num_gpus"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.pently_rho) or args.pently_rho < 0:
        raise ValueError("--pently-rho must be finite and nonnegative")
    if not math.isfinite(args.vocab_utilisation_weight) or args.vocab_utilisation_weight < 0:
        raise ValueError("--vocab-utilisation-weight must be finite and nonnegative")
    if args.pently_rho > 0 and (args.celex_dir is None or not args.celex_dir.is_dir()):
        raise ValueError("--celex-dir must point to CELEX data when --pently-rho is positive")
    if args.vocab_utilisation_weight > 0 and (
        not args.compression_divisors
        or len(args.compression_divisors) != len(args.vocab_sizes)
        or any(not math.isfinite(value) or value <= 0 for value in args.compression_divisors)
    ):
        raise ValueError("Supply one positive --compression-divisor for each --vocab-size")
    options = TrainingOptions(args.training_mode, args.max_token_bytes, args.super_base_vocab_size)
    options.validate_penalties(args.pently_rho, args.vocab_utilisation_weight)
    reserved = 256 + (10 if args.pretokenizer == "nanochat" else 6)
    if any(k <= reserved for k in args.vocab_sizes):
        raise ValueError(f"--vocab-size must exceed {reserved} reserved byte and special tokens")
    if args.training_mode == "super":
        for k in args.vocab_sizes:
            options.base_size(k, reserved)


def check_gpu_backend() -> None:
    if importlib.util.find_spec("cuopt") is None:
        raise RuntimeError("Training requires NVIDIA cuOpt. Install cuOpt in a CUDA-enabled environment before running this command.")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Training requires an NVIDIA GPU; nvidia-smi was not found.")
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("Training requires a visible NVIDIA GPU; nvidia-smi -L found none.")


def load_data(path: Path, text_column: str | None = None):
    from datasets import Dataset, load_dataset
    from lp_tokenizer.training import load_training_dataset, normalize_to_text_column

    if path.is_file() and path.suffix.lower() in {".txt", ".text"}:
        def records():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    value = line.rstrip("\r\n")
                    if value.strip():
                        yield {"text": value}
        dataset = Dataset.from_generator(records)
    elif path.is_file() and path.suffix.lower() == ".parquet":
        dataset = load_dataset("parquet", data_files=str(path), split="train")
    else:
        dataset = load_training_dataset(str(path), text_column=text_column)
    if path.is_file():
        dataset = normalize_to_text_column(dataset, text_column=text_column)
    if not len(dataset):
        raise ValueError("Training data has no nonempty records")
    return dataset


def run(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        validate(args)
        check_gpu_backend()
    except (ValueError, RuntimeError) as error:
        p.error(str(error))
    os.environ["NUM_PROC"] = str(args.num_proc)
    os.environ["BATCH_SIZE"] = str(args.batch_size)
    os.environ["LP_NUM_GPUS"] = str(args.num_gpus)
    os.environ["PRETOKENIZER_MODE"] = args.pretokenizer

    from lp_tokenizer.export import export_raw_vocabulary
    from lp_tokenizer.training import (
        BYTE_LEVEL_ALPHABET, build_pretokenizer, get_special_tokens,
        train_lp_tokenizer_sweep,
    )
    from lp_tokenizer.document_tokenizer import mode_directory

    dataset = load_data(args.data, args.text_column)
    divisors = (dict(zip(args.vocab_sizes, args.compression_divisors))
                if args.compression_divisors else None)
    raw_dir = args.output_dir / "raw"
    train_lp_tokenizer_sweep(
        dataset, list(BYTE_LEVEL_ALPHABET), args.vocab_sizes, str(raw_dir),
        build_pretokenizer(args.pretokenizer), get_special_tokens(args.pretokenizer),
        pently_rho=args.pently_rho,
        celex_dir=str(args.celex_dir) if args.celex_dir else None,
        vocab_utilisation_weight=args.vocab_utilisation_weight,
        training_mode=args.training_mode,
        max_token_bytes=args.max_token_bytes,
        super_base_vocab_size=args.super_base_vocab_size,
        compression_divisor=divisors,
    )
    for k in sorted(set(args.vocab_sizes)):
        raw_file = mode_directory(raw_dir, args.training_mode) / f"lp_tokens_{k}.pkl"
        dest = args.output_dir / f"vocab_{k}" / "tokenizer"
        export_raw_vocabulary(raw_file, k, dest,
                              pretokenizer_mode=args.pretokenizer,
                              rounding_scheme=args.rounding)
        print(f"Saved raw LP vocabulary: {raw_file}")
        print(f"Saved tokenizer: {dest}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
