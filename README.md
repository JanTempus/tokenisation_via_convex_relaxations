# lp-tokeniser

Train and export LP tokenizers from text or Hugging Face datasets. Python 3.11+, [NVIDIA cuOpt with multi-GPU PDLP support](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/convex/convex-examples.html) (26.10+), and four visible NVIDIA GPUs are required for training. Install cuOpt using the instructions for your CUDA environment before starting a training run.

```bash
python -m pip install -e .
python main.py --data mydataset.txt --vocab-size 8192 --output-dir output
# The installed command works too:
lp-tokeniser --data mydataset.txt --vocab-size 8192 --output-dir output
```

Each nonempty UTF-8 line in a `.txt` file becomes one training record. Saved Hugging Face Datasets and Parquet inputs are also accepted. The default is standard mode, nanochat pretokenization, and deterministic rounding. Each size writes `output/raw/lp_tokens_K.pkl` and `output/vocab_K/tokenizer/`. Document modes store raw files under `output/raw/<mode>/` and export a `DocumentLPTokenizer` that must be loaded from `lp_tokenizer.document_tokenizer`.

Training always selects cuOpt PDLP with crossover disabled and uses four GPUs together per solve by default. `--num-gpus` overrides the GPU count; direct Python calls use `LP_NUM_GPUS` (default: `4`). Keep all requested GPUs visible in `CUDA_VISIBLE_DEVICES`.

Repeat `--vocab-size` for a sweep. Vocabulary sizes are solved sequentially, with each solve using the configured GPUs. Run `python main.py --help` for pretokenizer, document mode, penalty, processing, and rounding options. `--pently-rho` requires `--celex-dir`; a positive `--vocab-utilisation-weight` requires one `--compression-divisor` per `--vocab-size`. Paper-specific divisors live in the experiments repository.
