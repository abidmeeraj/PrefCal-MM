# PrefCal-MM

**Preference-Calibrated Evaluation of Multimodal Summaries**

PrefCal-MM is a reference-free metric for *Multimodal Summarization with Multimodal Output* (MSMO): a system receives a document with text and images and returns a text summary plus a selection of images. PrefCal-MM scores such outputs on three interpretable pillars and combines them with Bradley–Terry models trained on pairwise human preferences.

This repository contains the reference implementation for the paper *Preference-Calibrated Evaluation of Multimodal Summaries* (Ali, Mollá-Aliod and Naseem, under review), including the trained aggregation weights.

---

## Contents

- [How the metric works](#how-the-metric-works)
- [Installation](#installation)
- [Data](#data)
- [Quick start: reproduce the paper on mLLM-EVAL](#quick-start-reproduce-the-paper-on-mllm-eval)
- [Scoring your own summaries](#scoring-your-own-summaries)
- [Using PrefCal-MM as a library](#using-prefcal-mm-as-a-library)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Expected results](#expected-results)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## How the metric works

| Pillar | Sub-score | What it measures | Judge (paper setting) |
|---|---|---|---|
| 1. Text | `sfact_unified` (S_fact†) | Fraction of atomic facts in the summary supported by the source **text or source images** | Qwen2.5-7B-Instruct (decomposition, text verification) + Qwen2.5-VL-7B-Instruct (visual verification) |
| | `srel`, `scoh`, `sflu` | G-Eval relevance, coherence, fluency (1–5 → [0, 1]) | Qwen2.5-7B-Instruct |
| 2. Alignment | `srelevance` | Do the **selected** images support the summary text? (1–5 → [0, 1]) | Qwen2.5-VL-72B-Instruct |
| 3. Diversity | `sdiversity` | Semantic diversity of the selected images: DPP log-determinant over CLIP features | CLIP ViT-B/16 |

**Cascaded multimodal fact verification.** The summary is decomposed into atomic facts. Every fact is first checked against the source text (Stage A). Only facts the text does not support are sent, together with *all* source images, to a visual verifier (Stage B). A fact counts as supported if either stage supports it. Facts whose verdict cannot be parsed are left out of the denominator.

**Two-stage Bradley–Terry aggregation.** For every article, each pair of systems is labelled with the system that humans rated higher on Overall Quality (ties are dropped). A logistic regression on feature differences, with no intercept, learns the weights. Negative weights are clipped to zero and the weights are normalised to sum to one.

```
S_text†    = w_fact·S_fact† + w_rel·S_rel + w_coh·S_coh + w_flu·S_flu          (Stage 1)
PrefCal-MM = w_text·S_text† + w_relevance·S_relevance + w_diversity·S_div      (Stage 2)
```

Released weights (trained on mLLM-EVAL, 5,197 preference pairs):

| Stage 1 | w_fact 0.253 | w_rel 0.249 | w_coh 0.233 | w_flu 0.264 | CV acc. 0.849 ± 0.013 |
|---|---|---|---|---|---|
| **Stage 2** | **w_text 0.801** | **w_relevance 0.175** | **w_diversity 0.024** | | **CV acc. 0.852 ± 0.009** |

**Calibration gate.** A sub-metric enters the composite only if its Kendall τ_b with its matched human dimension is at least 0.10 (S_fact† ↔ Consistency, S_rel ↔ Relevance, S_coh ↔ Coherence, S_flu ↔ Fluency, S_relevance ↔ Text–Images Relevance).

Diversity is undefined when a summary has fewer than two images. Those records get `NaN` for `sdiversity` and `prefcal_mm`, and are excluded from training and from any comparison that involves diversity.

---

## Installation

Python ≥ 3.10 and a CUDA GPU are required for the judge models.

```bash
git clone https://github.com/abidmeeraj/PrefCal-MM.git
cd PrefCal-MM
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[lmdb]" to read images from LMDB, ".[dev]" for pytest
```

Install it in editable mode (`-e`): the code reads `configs/` and `weights/` from the repository root.

**Models.** All judges come from the Hugging Face Hub. You can download them up front (gated models such as Llama need `huggingface-cli login`):

```bash
python -m prefcal_mm.cli.download_models               # the four models used in the paper
python -m prefcal_mm.cli.download_models --role vafv   # a single role
```

**Hardware.** The 7B judges need about 16 GB of GPU memory each in fp16. The 72B alignment judge needs about 145 GB in fp16 and is spread across all visible GPUs with `device_map="auto"`; with less GPU memory, layers are offloaded to CPU, which works but is much slower. You can swap in a smaller judge (e.g. `--model Qwen/Qwen2.5-VL-7B-Instruct`), but the released weights were fitted to the 72B judge's scores. The pillars run independently, so each one can go on a different machine.

If your compute nodes have no internet access, download the models first and then set `HF_HUB_OFFLINE=1` (the loaders then read only from the local cache).

---

## Data

### mLLM-EVAL (used in the paper)

The benchmark is described in Zhuang et al. (2024). It has 142 news articles, 11 summarisation systems and 1,562 (article, system) pairs, with human ratings on seven dimensions averaged over three annotators. Obtain it from the benchmark repository, [hjzhuang/MSMO-Eval](https://github.com/hjzhuang/MSMO-Eval). Then arrange it as below, or point the config at your copy:

```
data/mllm_eval/
├── MM_Eval_annotations.csv                      # one row per pair
└── meta_eval_data/
    └── <article_id>_<system_name>/              # e.g. 00c93d...bdbc_gpt4
        ├── source_doc.txt                       # source article text
        ├── source_img/                          # source article images (used by the visual verifier)
        ├── summary.txt                          # system summary
        └── summary_img/                         # images selected by the system (used by Pillars 2 and 3)
```

The annotation CSV has the columns `ID` (`<article_id>_<system_name>`), `Relevance`, `Coherence`, `Consistency`, `Fluency`, `Summary_Images_Quality`, `Text_Summary_Summary_Images_Relevance` and `Overall_Quality`. The paths and column names are set in `configs/config.yaml` under `datasets.mllm_eval`.

---

## Quick start: reproduce the paper on mLLM-EVAL

Each pillar is scored separately, so it loads only the model it needs. Runs can resume: if one is interrupted, rerun the same command and records already scored are skipped. Results go to `outputs/mllm_eval/`.

```bash
# 1. Score the four pillars (independent; run them in any order or in parallel)
python -m prefcal_mm.cli.score --pillar factual       # decomposition + text and visual verification
python -m prefcal_mm.cli.score --pillar qualitative   # G-Eval coherence / fluency / relevance
python -m prefcal_mm.cli.score --pillar relevance     # image-text alignment (72B judge)
python -m prefcal_mm.cli.score --pillar diversity     # CLIP + DPP

# 2. Join the pillars and compute S_text† and PrefCal-MM with the released weights
python -m prefcal_mm.cli.consolidate

# 3. Agreement with human Overall Quality + calibration gate
python -m prefcal_mm.cli.evaluate
```

Add `--max 5` to any `score` command for a quick smoke test.

To **re-learn the weights** from your scores rather than using the released ones:

```bash
python -m prefcal_mm.cli.train                                  # -> outputs/mllm_eval/weights/
python -m prefcal_mm.cli.evaluate --weights-dir outputs/mllm_eval/weights
```

`train` runs the calibration gate, fits Stage 1, recomputes S_text† with the new Stage-1 weights, fits Stage 2, and reports the cross-validated pairwise accuracy (5-fold `GroupKFold` by article). The released `weights/` directory is never overwritten.

---

## Scoring your own summaries

Put one row per summary in a CSV:

| column | content |
|---|---|
| `id` | article identifier |
| `source_text` | source article text |
| `summary` | generated summary text |
| `source_images` | JSON list of the article's image paths, e.g. `["a1/1.jpg", "a1/2.jpg"]` |
| `selected_images` | JSON list of the images the system selected |
| *(optional)* system column | name of the system that produced the summary, when the CSV holds several systems; set `system_column` in the config (otherwise every row gets `system_name`) |

Relative image paths are resolved against `image_root`. To use different column names, or to read images from an LMDB store, copy the `custom_csv` block in `configs/config.yaml` under a new name and edit it.

```bash
python -m prefcal_mm.cli.score --pillar factual     --csv data/custom/summaries.csv
python -m prefcal_mm.cli.score --pillar qualitative --csv data/custom/summaries.csv
python -m prefcal_mm.cli.score --pillar relevance   --csv data/custom/summaries.csv
python -m prefcal_mm.cli.score --pillar diversity   --csv data/custom/summaries.csv
python -m prefcal_mm.cli.consolidate --dataset custom_csv   # -> outputs/custom_csv/scores.csv
```

If you have human ratings, `evaluate` and `train` accept `--dataset custom_csv --annotations ratings.csv`. The ratings file needs `article_id` and `system_name` (or `pair_id`) plus one column per rating, e.g. `overall_quality`. `train` needs several systems per article, because the preferences are built within each article.

The released weights were learned on news summaries. The paper argues they reflect that domain (text dominates Overall Quality), so for other domains, re-learn them if you have human ratings.

---

## Using PrefCal-MM as a library

```python
from PIL import Image
from prefcal_mm.pipeline import score_record

record = {
    "pair_id": "article1_mysystem", "article_id": "article1", "system_name": "mysystem",
    "article": {"source_text": open("article.txt").read(),
                "source_images": [Image.open(p).convert("RGB") for p in ["s1.jpg", "s2.jpg"]]},
    "summary": {"text": open("summary.txt").read(),
                "selected_images": [Image.open(p).convert("RGB") for p in ["s1.jpg", "s2.jpg"]]},
}
scores = score_record(record)
print(scores["prefcal_mm"], scores["sfact_unified"], scores["srelevance"])
```

`score_record` loads every judge in one process, so it needs enough GPU memory for all of them at once. For datasets, use the command-line tools. You can also use the individual pillars directly: `prefcal_mm.pillars.factual.score(record)`, `qualitative.compute(summary, source)`, `relevance.compute(summary, images)` and `diversity.compute(images)`.

---

## Outputs

`outputs/<dataset>/pillars/<pillar>.csv` holds one row per record. A record that failed has `NaN` scores and the message in the `error` column.

`outputs/<dataset>/scores.csv` (from `consolidate`):

| column | meaning |
|---|---|
| `prefcal_mm` | final composite score; `NaN` if fewer than two images were selected |
| `stext_unified` | text-pillar score S_text† |
| `sfact_unified` | unified factual consistency S_fact† |
| `srel`, `scoh`, `sflu` | G-Eval relevance, coherence, fluency in [0, 1] |
| `srelevance`, `raw_srelevance` | image–text alignment in [0, 1], and the raw 1–5 judgement |
| `sdiversity`, `raw_diversity` | DPP diversity in [0, 1], and the raw log-determinant |
| `n_atomic_facts` | number of decomposed facts |
| `n_supported_text_only` | facts supported by the source text |
| `n_supported_image_only` | facts rejected by text but supported by the source images |
| `n_supported_neither`, `n_unk` | unsupported facts; unparseable facts (excluded from S_fact†) |

The `factual` pillar CSV also stores `atomic_facts_json`: every fact with its text and visual verdicts, for auditing.

`evaluate` writes `correlation.csv` and `calibration_gate.csv` to `outputs/<dataset>/evaluation/`. These hold segment-level Kendall τ_b with a bootstrap 95 % CI, Spearman ρ and Pearson r, and system-level τ_b and r over per-system means.

---

## Configuration

| file | what it controls |
|---|---|
| `configs/models.yaml` | judge model per role (`final`) and tested alternatives (`candidates`) |
| `configs/prompts.yaml` | every prompt template |
| `configs/config.yaml` | decoding, diversity formula, normalisation bounds, Bradley–Terry settings, calibration gate, dataset paths and column mappings |

You can override a single run without editing files:

```bash
python -m prefcal_mm.cli.score --pillar factual --model meta-llama/Llama-3.1-8B-Instruct --vafv-model Qwen/Qwen2.5-VL-72B-Instruct
python -m prefcal_mm.cli.score --pillar diversity --formula tce        # conference formulation
```

Non-default `--model` or `--formula` choices are added to the output file name, so variants don't overwrite each other. All judges decode greedily (temperature 0). `decoding.geval_scoring_mode: probability_weighted` switches G-Eval to the expectation over score tokens.

---

## Expected results

On mLLM-EVAL, against human Overall Quality (the paper's headline agreement table). Segment level is one value per (article, system) pair; system level uses the 11 per-system means. Rows involving diversity use the 1,307 pairs with at least two selected images.

| metric | n_seg | segment τ_b [95 % CI] | system τ_b | system r |
|---|---|---|---|---|
| **PrefCal-MM** | 1,307 | **0.607** [0.583, 0.631] | **0.927** | **0.989** |
| S_text† | 1,562 | 0.598 [0.571, 0.624] | 0.891 | 0.983 |
| S_rel | 1,562 | 0.609 [0.579, 0.639] | 0.855 | 0.965 |
| S_coh | 1,562 | 0.603 [0.575, 0.629] | 0.818 | 0.936 |
| S_fact† | 1,562 | 0.458 [0.422, 0.494] | 0.709 | 0.961 |
| S_flu | 1,562 | 0.454 [0.420, 0.488] | 0.844 | 0.747 |
| S_relevance | 1,562 | 0.402 [0.363, 0.436] | 0.771 | 0.907 |

Calibration gate (τ_b with the matched dimension, n = 1,562): S_rel 0.660, S_fact† 0.563, S_flu 0.469, S_relevance 0.430, S_coh 0.371. All pass.

Decoding is deterministic, but GPU kernels, library versions and hardware can still change low-order digits of the judge scores. Expect small deviations from these numbers when rescoring from scratch. Given the paper's pillar scores, the aggregation and evaluation code reproduces the weights and correlations above exactly.

---

## Repository layout

```
configs/                  models.yaml, prompts.yaml, config.yaml
weights/                  released Stage-1 and Stage-2 Bradley-Terry weights
prefcal_mm/
  pillars/                factual.py (cascade + S_fact†), qualitative.py, relevance.py, diversity.py
  aggregation/            preferences.py, bradley_terry.py, composite.py
  evaluation/             calibration.py (gate), correlation.py
  data/                   mLLM-EVAL loader, generic CSV loader, image loading
  cli/                    score, consolidate, train, evaluate, download_models
  pipeline.py             score_record(): single-record API
  models.py, generation.py, parsing.py, normalisation.py, config.py
tests/                    unit tests (pytest tests/)
```

---

## Limitations

- **Judge robustness.** The results come from one set of Qwen judges with deterministic decoding. Sensitivity to the prompts, the judge family and the order in which images are presented has not been measured.
- **Single benchmark.** The weights and correlations come from one English news benchmark with 11 systems. Text dominates the Stage-2 weights (0.801), which reflects that domain.
- **Diversity.** S_div passes its calibration gate against pairwise diversity annotations, but has no detectable effect on Overall Quality on mLLM-EVAL. It is kept as a diagnostic.
- **Segment level.** At segment level the composite is not significantly better than the strongest text sub-metric. Its advantage is at system level.

See the paper for a full discussion.

---

## Citation

The accompanying article, *Preference-Calibrated Evaluation of Multimodal Summaries*, is under review. A citation will be added here on publication. Until then, please cite this repository; GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff).

If you use the mLLM-EVAL benchmark, please also cite Zhuang et al. (2024); see [hjzhuang/MSMO-Eval](https://github.com/hjzhuang/MSMO-Eval).

---

## License

The code is released under the [MIT License](LICENSE). Judge models and datasets keep their own licences.

## Acknowledgements

This research was undertaken with the assistance of resources from the National Computational Infrastructure (NCI Australia), an NCRIS enabled capability supported by the Australian Government.
