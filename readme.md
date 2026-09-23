# Tests exps SAE x Tabular PFM

Embedding-invariance experiments on tabular foundation models: a model is fitted on a
perturbed version of a dataset, its test-row embeddings are read at a chosen layer, and
they are compared to the embeddings obtained without perturbation (`--metric`, cosine
similarity by default).

## Environment

On Jean Zay the venv sits on top of the IDRIS `pytorch-gpu` module, which provides torch
built against the CUDA/NCCL of the machine. The module has to be loaded before the venv is
activated, `setup_env.sh` does both:

```bash
source setup_env.sh
```

`job.slurm` is the matching batch script (`mkdir -p logs` once, then `sbatch job.slurm`).

## Usage

```bash
python run_exps.py --model tabpfn_v3 --dataset breast_cancer --exp perturbations --layer last
```

Every run writes `results/<model>_<dataset>_<exp>_layer-<layer>/`, where `<model>` includes the
version (`tabpfn_v3`, `tabpfn_v2`, `tabicl_v2`, `tabicl_v1`, `tabfm_v1`, `mitra_v1`, `mitra_v2`), with a json file of the
raw curves and the matching figure. `--metric mi`/`cka` add a `_metric-<metric>` suffix to
that name, so they never collide with or overwrite a `cosine` run of the same layer;
`cosine` itself is left unsuffixed, matching every result computed before `--metric` existed.

| Argument | Values | Default |
| --- | --- | --- |
| `--model` | `tabpfn_v3`, `tabpfn_v2`, `tabicl_v2`, `tabicl_v1`, `tabfm_v1`, `mitra_v1`, `mitra_v2` | `tabpfn_v3` |
| `--dataset` | `breast_cancer`, `iris`, `wine`, `digits` | `breast_cancer` |
| `--exp` | one experiment or a group (see below) | `perturbations` |
| `--layer` | `last`, an index in the layer stack, or a dotted submodule name | `last` |
| `--metric` | `cosine`, `mi`, `cka` (see `utils/metrics.py`) | `cosine` |
| `--n_levels` | number of perturbation intensities swept | `20` |
| `--seed` / `--split_seed` | perturbation seed / train-test split seed | `0` / `42` |
| `--eval` | also report accuracy and ROC AUC on the clean data | off |

### Metric

`--metric` picks how the perturbed embeddings are compared to the unperturbed baseline, in
`utils/metrics.py`:

- `cosine` (default): cosine similarity, computed per test row then averaged. Needs nothing
  beyond one row's two vectors.
- `mi`: mutual information, estimated one embedding dimension at a time (a k-NN estimator,
  `sklearn.feature_selection.mutual_info_regression`) over the whole test set, then averaged
  over dimensions. Unlike cosine similarity, MI is a property of a distribution, not of a
  single pair of vectors, so it needs the test set as its sample; past `MI_MAX_DIMS` (64)
  embedding dimensions a fixed random subset is used, since cost scales with how many
  dimensions are estimated. This does not capture dependence that only shows up jointly
  across dimensions - `cka` does, see its own docstring for why MI does not attempt that
  here (the short version: not enough test rows relative to the embedding width for a joint
  k-NN density estimate to be reliable).
- `cka`: Centered Kernel Alignment (Kornblith et al. 2019), built on HSIC, reads every
  embedding dimension jointly rather than one at a time, and does not need a density
  estimate, so it stays reliable at far smaller sample-to-dimension ratios than `mi`'s
  per-dimension estimate does. Its values are bounded to `[0, 1]` like cosine similarity, but
  have a known finite-sample bias: two literally independent embeddings still read as
  CKA > 0 (noticeably so at a few hundred test rows), tightening toward 0 only as the test
  set grows - this is the standard, literature-conventional CKA estimator, not a bug.

`--metric` takes several values (`--metric cosine mi cka`): they are all computed from the
same embeddings in one sweep, and each is saved to its own result folder. The campaign
scripts (`scripts/all_runs*.sh`) compute every metric of `METRICS_ALL` (in
`scripts/config.sh`, default `cosine mi cka`), and for each layer only ask for the metrics
whose result is still missing, so existing results are never recomputed.

`fuse_plot.py` also takes `--metric` (default: `cosine` for a single plot, every metric found
for `--all`); results of different metrics are never mixed into the same figure, their values
are not on comparable scales.

### Experiments

`row_shuffle`, `col_shuffle`, `random_labels`, `col_remove`, `row_remove`, `value_noise`,
`alt_target` (predict another column of X instead of y), plus two groups: `perturbations`
(the 2x3 grid of `test.py`) and `all`.

### Layer position

`--layer last` uses the embeddings the model exposes itself (`get_embeddings` for TabPFN,
the end of the ICL transformer for TabICL). Any other value reads the activations with a forward
hook, either by index in the layer stack or by submodule name. The names depend on the
installed version of the library, so list them first:

```bash
python run_exps.py --model tabpfn_v3 --list_layers
python run_exps.py --model tabpfn_v3 --layer 5
python run_exps.py --model tabicl_v2 --layer row_interactor.layers.3
```

TabICL (`tabicl_v2`, `tabicl_v1`) and TabPFN v3 (`tabpfn_v3`) have three named layer
positions, in the order the data flows through the network:

| `--layer` | Reads | Shape per row |
| --- | --- | --- |
| `tf_col` | column-wise stage output (TabICL: `col_embedder.tf_col`, before the affine map to cell embeddings; TabPFN: `feature_distribution_embedder`) | `features x dim`, reduced by `--feature_slot` |
| `tf_row` | row-wise stage output, the concatenated CLS tokens (TabICL: `row_interactor`; TabPFN: `column_aggregator`) | `n_cls x dim` |
| `tf_icl` | end of the in-context-learning transformer, after its final norm, what the decoder reads (TabICL: `icl_predictor.ln`; TabPFN: `output_norm`) | `dim` |

For TabICL only, `tf_col` and `tf_row` are each a stack of blocks, and `--layer tf_col_early` /
`tf_col_middle` and `tf_row_early` / `tf_row_middle` read an earlier block instead of the last
one (closing the first quarter / half of that stack), the same way as the stage's own final
output: `tf_col_*` reduces the block's per-cell output the same way `tf_col` does, `tf_row_*`
reads the same CLS-token slots `tf_row` reads, just refined less. `tf_row`'s last block does
the CLS-only pooling that `tf_row` itself reads, so `tf_row_early` / `tf_row_middle` are
computed over its other blocks only.

TabFM v1 (`tabfm_v1`) alternates column and row stages (`col_embedder`, `row_interactor`,
`col_embedder_2`, `row_interactor_2`, then the ICL transformer), so a bare `tf_col` or `tf_row`
would be ambiguous. Its layers are numbered in forward order: `tf_col_1`, `tf_row_1`, `tf_col_2`,
`tf_row_2` (the row output the ICL stage reads) and `tf_icl`. `tf_col_*` are the set-transformer
outputs, before the linear map and norm that follow them.

Mitra (`mitra_v1`, `mitra_v2`) is a stack of 12 identical layers, each attending across the rows
and then across the features, so it has no column or row stage to name. Read it with
`--layer <i>` (0 to 11, the output of layer `i` for the query rows), `--layer early` / `middle`
(the layers closing the first quarter and the first half of the stack, layers 2 and 5) or
`--layer last` (after the final norm, right before the prediction head). Each row is a set of `features + 1` tokens, the
first one being the target token: `--feature_slot target` (the default) keeps it, `mean` averages
the feature tokens and `flatten` concatenates them. The classifier runs in-context, without
fine-tuning.

For TabICL and TabFM, `tf_icl` is also what `--layer last` reads. For `tabpfn_v3`, `last` still means
the `get_embeddings` output of the library. `tabpfn_v2` has one flat stack of alternating
attention blocks and no such stages: read it with `--layer <index>` on its `blocks`, or by
depth with `--layer early` (first quarter of the blocks), `middle` (first half) and `last`.

For `tf_col` there is no target cell (TabICL and TabPFN v3), so `--feature_slot` is `mean` (also what `target` is
read as) or `flatten`. For TabICL the 4 columns reserved for the CLS tokens are dropped. Do not hook
`row_interactor.tf_row` itself: its last block is called directly, so the hook never fires.

## Structure

```
setup_env.sh                       # module + venv, to be sourced
job.slurm                          # batch script for the compute nodes
run_exps.py                        # argparse entry point
models/model_importer.py           # model_name (with version) + params -> model
models/base_model.py               # fit / predict / get_embeddings(layer) + hook machinery
models/tabpfn_model.py             # shared TabPFN wrapper + HF checkpoint lookup
models/tabpfn_v3_model.py          # TabPFNV3: default checkpoint
models/tabpfn_v2_model.py          # TabPFNV2: default checkpoint
models/tabicl_model.py             # shared TabICL wrapper
models/tabicl_v2_model.py          # TabICLV2: pinned checkpoint version
models/tabicl_v1_model.py          # TabICLV1: pinned checkpoint version
models/tabfm_v1_model.py           # TabFMV1: Google TabFM, PyTorch backend
models/mitra_model.py              # shared Mitra wrapper (AutoGluon), layer hooks
models/mitra_v1_model.py           # MitraV1: autogluon/mitra-classifier
models/mitra_v2_model.py           # MitraV2: autogluon/mitra-classifier-2
datasets/dataset_importer.py       # dataset_name -> TabularDataset
datasets/tabular_dataset.py        # train/test split container
experiments/experiment_importer.py # exp_name -> list of experiments
experiments/base_experiment.py     # Experiment interface + run_experiment loop
experiments/perturbations.py       # the six perturbation sweeps
experiments/target_column.py       # alternative-target sweep
utils/metrics.py                   # cosine / mi / cka, the --metric choices
utils/                             # permutations, plots, io
```

Adding a model version, a dataset or an experiment means adding one file and one entry in the
matching importer.

## Notes

- The TabPFN v3 (`tabpfn_v3`) and TabPFN v2 (`tabpfn_v2`) checkpoints are read from `$WORK/hf_cache`, where `down_weight.py` puts them;
  the compute nodes have no internet access, so run it on a login node first. `TABPFN_V3_CKPT` /
  `TABPFN_V2_CKPT` override the resolved path, `--model_path` overrides it per run.
- `n_estimators=1` keeps a single deterministic forward pass, so the captured
  representations are not an average over shuffled ensembles.
- `test.py`, `test_bis.py` and `test_tabicl.py` are the original standalone scripts this
  repo was factored out of.
- Never let pip change the torch of the module (`pip install` of a package that pins a newer
  torch, such as `autogluon.tabular[mitra]`, installs its own torch and CUDA libraries in the
  venv and breaks every model). Install such packages with `--no-deps`, see requirements.txt.
  If it happens, `pip freeze --local | grep -iE '^(torch|triton|nvidia|cuda)'` shows the
  intruders, and uninstalling them brings back the module's version.
