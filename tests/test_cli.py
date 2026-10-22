from pathlib import Path
import pickle
from unittest import mock

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from lp_tokenizer.cli import load_data, parser, run, validate
from lp_tokenizer.datastructures import possibleToken
from lp_tokenizer.export import NANOCHAT_SPECIAL_TOKENS
from lp_tokenizer.lp_tokenizer import BYTE_LEVEL_ALPHABET


def raw_payload():
    return {
        "possible_tokens": [possibleToken("he", 0.9), possibleToken("ll", 0.8)],
        "unique_chars": list(BYTE_LEVEL_ALPHABET),
        "special_tokens": list(NANOCHAT_SPECIAL_TOKENS),
    }


def test_text_lines_and_saved_dataset(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello\n\n  \nworld  \n", encoding="utf-8")
    assert load_data(path)["text"] == ["hello", "world  "]
    saved = tmp_path / "saved"
    Dataset.from_dict({"content": ["one", "two"]}).save_to_disk(saved)
    assert load_data(saved, "content")["text"] == ["one", "two"]
    parquet = tmp_path / "sample.parquet"
    Dataset.from_dict({"content": ["three"]}).to_parquet(parquet)
    assert load_data(parquet, "content")["text"] == ["three"]


def test_validation_requires_divisor_and_celex(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello\n", encoding="utf-8")
    args = parser().parse_args(["--data", str(data), "--vocab-size", "268",
                                "--vocab-utilisation-weight", "0.1"])
    with pytest.raises(ValueError, match="compression-divisor"):
        validate(args)
    args = parser().parse_args(["--data", str(data), "--vocab-size", "268",
                                "--pently-rho", "0.1"])
    with pytest.raises(ValueError, match="celex-dir"):
        validate(args)


def test_command_exports_loadable_tokenizer(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello\nworld\n", encoding="utf-8")
    output = tmp_path / "out"

    def fake_train(dataset, unique_chars, sizes, raw_dir, *args, **kwargs):
        assert dataset["text"] == ["hello", "world"]
        assert sizes == [268]
        directory = Path(raw_dir)
        directory.mkdir(parents=True)
        with (directory / "lp_tokens_268.pkl").open("wb") as handle:
            pickle.dump(raw_payload(), handle)

    with mock.patch("lp_tokenizer.cli.check_gpu_backend"), mock.patch(
        "lp_tokenizer.training.train_lp_tokenizer_sweep", side_effect=fake_train
    ):
        assert run(["--data", str(data), "--vocab-size", "268",
                    "--output-dir", str(output), "--num-proc", "1"]) == 0
    assert (output / "raw/lp_tokens_268.pkl").is_file()
    tokenizer = AutoTokenizer.from_pretrained(output / "vocab_268/tokenizer", local_files_only=True)
    assert len(tokenizer) == 268
    sample = "hello, world!\n"
    assert tokenizer.decode(tokenizer(sample, add_special_tokens=False)["input_ids"]) == sample


def test_document_export_roundtrip(tmp_path):
    from lp_tokenizer.document_tokenizer import (
        DocumentLPTokenizer, FORMAT_VERSION, serialize_pretokenizer,
    )
    from lp_tokenizer.export import export_raw_vocabulary
    from lp_tokenizer.training import build_pretokenizer

    metadata = {
        "format_version": FORMAT_VERSION,
        "training_mode": "boundless",
        "max_token_bytes": 64,
        "pretokenizer": serialize_pretokenizer(build_pretokenizer("nanochat")),
        "special_tokens": list(NANOCHAT_SPECIAL_TOKENS),
        "vocab_size": 268,
    }
    raw = raw_payload()
    raw["metadata"] = metadata
    raw_path = tmp_path / "lp_tokens_268.pkl"
    with raw_path.open("wb") as handle:
        pickle.dump(raw, handle)
    output = export_raw_vocabulary(raw_path, 268, tmp_path / "tokenizer")
    loaded = DocumentLPTokenizer.from_pretrained(output)
    sample = "hello world!\n"
    assert loaded.decode(loaded.encode(sample)) == sample
    assert len(loaded) == 268


def test_utilisation_uses_requested_divisor_without_mutating_base_costs():
    import numpy as np
    from lp_tokenizer.lp_functions import _cuopt_objective

    data = {
        "c": np.array([12.0, 6.0, 0.0]),
        "num_f": 1, "num_g": 1, "num_t": 1,
        "vocab_utilisation_weight": 0.2,
        "utilisation_coefficients": np.array([0.5]),
        "compression_divisor": {268: 12.0},
    }
    result = _cuopt_objective(data, 2, 268)
    assert np.allclose(result, [0.95, 0.5, 0.1])
    assert np.array_equal(data["c"], [12.0, 6.0, 0.0])


def test_default_gpu_count():
    args = parser().parse_args(["--data", "data.txt", "--vocab-size", "268"])
    assert args.num_gpus == 4


@pytest.mark.parametrize("gpu_count", [None, "2"])
def test_cuopt_forces_pdlp_and_configures_gpus(monkeypatch, gpu_count):
    import sys
    from types import SimpleNamespace
    from lp_tokenizer.lp_functions import solve_cuopt_problem

    monkeypatch.delenv("LP_NUM_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if gpu_count is not None:
        monkeypatch.setenv("LP_NUM_GPUS", gpu_count)
    settings = mock.Mock()
    parameters = SimpleNamespace(
        CUOPT_CROSSOVER="crossover", CUOPT_METHOD="method", CUOPT_NUM_GPUS="num_gpus",
    )
    solver_settings = SimpleNamespace(
        SolverSettings=lambda: settings, SolverMethod=SimpleNamespace(PDLP=object()),
    )
    monkeypatch.setitem(sys.modules, "cuopt.linear_programming.solver.solver_parameters", parameters)
    monkeypatch.setitem(sys.modules, "cuopt.linear_programming.solver_settings", solver_settings)
    problem = mock.Mock(Status=SimpleNamespace(name="Optimal"), SolveTime=0.1)
    model = {
        "problem": problem, "cuopt_lp_data": {},
        "current_budget": 2, "current_vocab_size": 268,
        "num_f": 0, "num_g": 0, "num_t": 1,
        "variables": [SimpleNamespace(Value=1.0)],
    }
    solve_cuopt_problem(model, 2, vocab_size=268, verbose=False)
    problem.solve.assert_called_once_with(settings)
    settings.set_parameter.assert_has_calls([
        mock.call("crossover", False),
        mock.call("method", solver_settings.SolverMethod.PDLP),
        mock.call("num_gpus", int(gpu_count or "4")),
    ])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="exposes fewer GPUs"):
        solve_cuopt_problem(model, 2, vocab_size=268, verbose=False)


def test_sweep_keeps_all_gpus_visible_for_each_budget(tmp_path, monkeypatch):
    from lp_tokenizer import training

    monkeypatch.setenv("LP_NUM_GPUS", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2,1,0")
    tokenizer = mock.Mock(unique_chars=["a"], special_tokens_list=[])
    observed = []

    def solve(*args):
        import os
        observed.append((args[3], os.environ["CUDA_VISIBLE_DEVICES"]))

    with mock.patch.object(training, "Tokenizer", return_value=tokenizer), mock.patch.object(
        training, "_solve_and_save_lp_vocab", side_effect=solve,
    ):
        training.train_lp_tokenizer_sweep(
            ["hello"], ["a"], [269, 268], str(tmp_path), None, [],
        )
    assert observed == [(268, "3,2,1,0"), (269, "3,2,1,0")]
