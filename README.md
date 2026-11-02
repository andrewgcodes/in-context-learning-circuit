# A Circuit for In-Context Learning

Code, prompt batteries, figure scripts and paper source for *A Circuit for In-Context Learning* (Andrew Gao, William Gao; Stanford University).

The paper traces how a language model learns an arbitrary label mapping from a few demonstrations inside a single forward pass. Label-token positions act as slots: the sentence class before each label is written into the slot as a key, the label identity is the slot's value, query-side heads produce a code for the query sentence, reader heads match that code against the stored keys, a duplicate-label signal gates the readers, brake heads suppress recent-label bias, and late copier heads write the answer logits. The repository contains everything needed to run those measurements on a model and to regenerate the figures.

## Layout

- `icl/run.py` — loads a model, builds a battery, scores every head, runs the interventions (knockouts, twin-prompt handoffs, pushes, holds) and writes per-prompt rows to `knock.jsonl` together with the attention-row and direct-logit measurements.
- `icl/score.py` — per-sequence scores: label log-probs, surprisal, flag readout, query-attention and duplicate-label head scores, slot rows, MLP direct effects.
- `icl/loupes.py` — the finer measurements: MLP neuron writers, attention logit split, value readout.
- `icl/hooks.py` — forward hooks: head mean-ablation, Q/K/V/output handoff from a twin prompt, residual push, hold, attention-row and direct-logit capture.
- `icl/tasks.py` — the prompt batteries (`concept`, `concept12`, `labelswap`, `manyclass`, the function and correction families) and their token-aligned twins.
- `icl/tokspans.py` — token-span bookkeeping (label tokens, arrows, query), including the Llama BOS handling.
- `icl/check.py` — post-run invariant check (row counts, intact margin, twin alignment, attention positions).
- `icl/effects.py` — paired effects with bootstrap intervals from `knock.jsonl`.
- `icl/allheads_law.py` — fits the three-constant reader-mass law per head and predicts held-out regimes.
- `icl/predict_ndemo18.py` — the eighteen-demonstration prediction.
- `icl/modal_app.py` — launcher for a single H100 on Modal.
- `batteries/` — `icl_tasks.json` (task pairs and label sets) and `concept_tasks.json` (sentence pools per concept).
- `data/` — the summary numbers four figure scripts need: the fitted read-law constants (`D139_law_qwen.txt`, `D161_law_llama.txt`, written by `icl/allheads_law.py`), the eighteen-demonstration prediction-vs-measurement files (`D162_*_pred_vs_meas.json`, from `icl/predict_ndemo18.py`), the gate logits (`D66b_gate_logits.json`) and the attention-row / direct-logit fields of the `knock.jsonl` rows behind Figures 7 and 12 (`D145_reader_rows.jsonl`, `D159_dla_*.jsonl`).
- `figures/` — one script per paper figure; `figstyle.py` holds the shared style, `figcheck.py` is the collision check for the hand-drawn diagrams, `data_headgrid.json` feeds Figure 17.
- `paper/` — `main.tex`, `refs.bib`, the ICLR style files, the figure PDFs and a `Makefile` that builds the PDF and an arXiv tarball.
- `tests/` — battery construction and twin-alignment checks that run on CPU.

## Running

```
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or the CPU wheel
pip install -r requirements.txt
python icl/run.py Qwen/Qwen3-4B-Base concept12 seed=1 out=runs/example
python icl/check.py runs/example
python icl/effects.py effects.tsv runs/example
```

`run.py <model> <families> [key=value ...]` takes the HuggingFace model id, a comma-separated list of battery families (`concept12`, or `std,redund,correct,bind,ctl`, …), and then space-separated options: `out=` (output directory, otherwise `$GRAIL_OUT`), `seed=`, `smoke` (a few prompts per task, for a quick check), `intact` (no interventions). The family names are the `fam_*` functions in `tasks.py`; the options are documented at the top of `run.py`. Models load in bfloat16 on CUDA; a full battery on a 4B model takes about an hour on one H100.

The natural-text documents used to set the mean-ablation scale are taken from `NeelNanda/pile-10k` by default; set `ICL_DOCS=path.json` (a JSON list of strings) to use your own.

On Modal (needs the `huggingface` secret and the `mi-grails` / `mi-hfcache` volumes):

```
modal run --detach icl/modal_app.py --models Qwen/Qwen3-4B-Base --mode concept12 --extra "seed=1 out=/vol/grails/example"
```

## Figures

Every script writes `figures/<name>.pdf` relative to the working directory, so run them from `paper/` to refresh the paper's figures:

```
cd paper && for f in ../figures/fig*.py; do python $f; done
```

Fourteen of the eighteen figures carry their numbers inline (the values reported in the paper). Four read from `data/`: `fig3_law.py`, `fig4_gate.py`, `fig7_accounting.py` and `fig12_reader_rows.py` look under `$ICL_DATA` (default `data/`), so from `paper/` run them with `ICL_DATA=../data`. To regenerate those inputs from your own runs, point `icl/allheads_law.py` and `icl/predict_ndemo18.py` at the `knock.jsonl` directories written by `run.py`.

## Paper

```
make -C paper          # main.pdf (pdflatex + bibtex)
make -C paper arxiv    # arxiv.tar.gz: main.tex, main.bbl, refs.bib, style files, figures
```

## Citation

```
@article{gao2026iclcircuit,
  title  = {A Circuit for In-Context Learning},
  author = {Gao, Andrew and Gao, William},
  year   = {2026}
}
```

MIT license.
