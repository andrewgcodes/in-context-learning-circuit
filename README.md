# A Circuit for In-Context Learning

Code, prompt batteries, figure scripts, paper source and an interactive walkthrough for *A Circuit for In-Context Learning* (Andrew Gao and William Gao, equal contributors; Stanford University).

The paper traces how a language model learns an arbitrary label mapping from a few demonstrations inside a single forward pass. Label-token positions act as slots: the sentence class before each label is written into the slot as a key, the label identity is the slot's value, query-side heads produce a code for the query sentence, reader heads match that code against the stored keys, a duplicate-label signal gates the readers, brake heads suppress recent-label bias, and late copier heads write the answer logits. The repository contains everything needed to run those measurements on a model and to regenerate the figures.

## Interactive walkthrough

`docs/index.html` is a self-contained, step-by-step 3D walkthrough of the circuit (one HTML file, no server or network needed): download it and open it in a browser, or serve the repo with GitHub Pages. It follows one prompt through Qwen3-4B's 36 layers — writer heads filling the label slots, the duplicate-label gate, the first token as the off switch, the reader heads' lookup with the fitted read law, the brakes and the copiers — with controls for repeated versus unique labels, the query class and the number of demonstrations.

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
- `icl/analyze_D170.py` — paired margin effects with bootstrap intervals for any run's `knock.jsonl` (the connection cuts of the "Cutting the arrows" section); `icl/analyze_D174.py` ranks the complement screen and prints the extended head set; `icl/analyze_D185.py` / `analyze_D185_decomp.py` split a word-label margin into the part that follows the demonstrations ($R$) and the part that follows word meaning ($P$); `icl/analyze_D188_scale.py` merges the four-size census into `data/D188_summary.json` and `data/D188_heads_<run>.json`; `icl/analyze_D188_gate.py` reads the gate-dose runs.
- `icl/fit_gate.py` — fits the gate direction from unique- versus recurring-label prompt pairs and reports held-out sign agreement by layer.
- `icl/trace_view.py` — layer-by-layer account of one position from a `trace=<n>` run (the token observatory).
- `toy/` — the hand-set three-layer circuit (`toy_circuit.py`, writes `toy_results.json`), the from-scratch toy trained on the slot-gate task (`train_toy.py --cond mix|noone|gate`), its loupes (`analyze_toy.py RUNDIR`) and the Modal launcher (`modal_toy.py`). A run trains in about ten minutes on one GPU.
- `batteries/` — `icl_tasks.json` (task pairs and label sets) and `concept_tasks.json` (sentence pools per concept).
- `data/` — the summary numbers the figure scripts and the appendix tables rest on: the fitted read-law constants (`D139_law_qwen.txt`, `D161_law_llama.txt`, written by `icl/allheads_law.py`), the eighteen-demonstration prediction-vs-measurement files (`D162_*_pred_vs_meas.json`, from `icl/predict_ndemo18.py`), the gate logits (`D66b_gate_logits.json`) and the attention-row / direct-logit fields of the `knock.jsonl` rows behind Figures 7 and 12 (`D145_reader_rows.jsonl`, `D159_dla_*.jsonl`). Added for the connection cuts, completeness screens, replications, natural-label closure, instruct comparison and scale census: the paired-effect summaries `D170*_effects.txt`, `D171_*.txt`, `D173*`–`D178*`, `D180*`–`D184*`, `D186*`, `D187*` (outputs of `icl/analyze_D170.py`), the screens `D174_screen*.txt`, `D174_subbands*.txt`, `D181_screen_*.txt`, the $R$/$P$ decompositions `D18*_decomp.txt`, and the census files `D188_summary.json`, `D188_heads_*.json`.
- `figures/` — one script per paper figure (Figures 19 and 21–23 are new in this version: the token observatory, natural-label closure, instruct comparison and scale census); `figstyle.py` holds the shared style, `figcheck.py` is the collision check for the hand-drawn diagrams, `data_headgrid.json` feeds Figure 17.
- `paper/` — `main.tex`, `refs.bib`, the ICLR style files, the figure PDFs and a `Makefile` that builds the PDF and an arXiv tarball.
- `tests/` — battery construction and twin-alignment checks that run on CPU.
- `docs/` — the interactive walkthrough (`index.html`).

## Running

```
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or the CPU wheel
pip install -r requirements.txt
python icl/run.py Qwen/Qwen3-4B-Base concept12 seed=1 out=runs/example
python icl/check.py runs/example
python icl/effects.py effects.tsv runs/example
```

`run.py <model> <families> [key=value ...]` takes the HuggingFace model id, a comma-separated list of battery families (`concept12`, or `std,redund,correct,bind,ctl`, …), and then space-separated options: `out=` (output directory, otherwise `$GRAIL_OUT`), `seed=`, `smoke` (a few prompts per task, for a quick check), `intact` (no interventions). The family names are the `fam_*` functions in `tasks.py`; the options are documented at the top of `run.py`. Among them: `circ=<heads> circmean=battery` keeps only the named heads at the query (every other head replaced by its battery mean, a pre-pass), `circrs` resamples the others from another prompt instead, `circbands=a-b,...` and `circscreen=a-b` cut or screen the complement one layer band at a time, `sinkv=`, `rkcond=`, `raknock`, `radd=<gate npz>:<layer>:<scale>` and `twinset=` are the connection cuts, `hgain=` scales reader outputs, `trace=<n>` writes the token-observatory shards, and `kinds=` / `every=` select label kinds and prompt subsets. `check.py` and the analysis scripts read `knock.jsonl` or `knock.jsonl.gz`. Models load in bfloat16 on CUDA; a full battery on a 4B model takes about an hour on one H100.

The natural-text documents used to set the mean-ablation scale are taken from `NeelNanda/pile-10k` by default; set `ICL_DOCS=path.json` (a JSON list of strings) to use your own.

On Modal (needs the `huggingface` secret and the `mi-grails` / `mi-hfcache` volumes):

```
modal run --detach icl/modal_app.py --models Qwen/Qwen3-4B-Base --mode concept12 --extra "seed=1 out=/vol/grails/example"
```

## Figures

Every script writes `figures/<name>.pdf` relative to the working directory, so run them from `paper/` to refresh the paper's figures:

```
cd paper && for f in ../figures/fig*.py; do ICL_DATA=../data python $f; done
```

Most figures carry their numbers inline (the values reported in the paper). Seven read from `data/`: `fig3_law.py`, `fig4_gate.py`, `fig7_accounting.py`, `fig12_reader_rows.py`, `fig21_closure.py`, `fig22_instruct.py` and `fig23_scale.py` look under `$ICL_DATA` (default `../data`, i.e. the repo's `data/` when run from `paper/` or `figures/`). `fig19_trace.py` (the token observatory) reads the two trace shards written by `run.py ... trace=1` (intact, and with `circ=<heads>`) from `$ICL_DATA/trace/`; the shards are 29 MB each and are not in the repo, so the script skips itself when they are absent. Figure 16 (the flowchart) is hand-drawn and shipped as a PDF. To regenerate those inputs from your own runs, point `icl/allheads_law.py` and `icl/predict_ndemo18.py` at the `knock.jsonl` directories written by `run.py`.

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
