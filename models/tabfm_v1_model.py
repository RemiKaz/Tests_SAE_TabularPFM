import sys

from models.base_model import BaseTabularPFM


def _import_tabfm():
    """Import tabfm with its PyTorch backend only.

    tabfm tries its JAX backend first and only tolerates an ImportError. The flax of the
    IDRIS module is too old (no `nnx.dataclass`) and fails with an AttributeError instead,
    so flax is hidden while tabfm is imported, which makes that attempt fail cleanly.
    """
    missing = object()
    flax = sys.modules.get("flax", missing)
    sys.modules["flax"] = None  # `import flax` now raises ImportError
    try:
        import tabfm
    finally:
        if flax is missing:
            del sys.modules["flax"]
        else:
            sys.modules["flax"] = flax
    return tabfm


class TabFMV1(BaseTabularPFM):
    """TabFM v1.0.0 (Google), PyTorch backend, read with forward hooks.

    The forward pass alternates column and row stages before the ICL transformer:
    cell_embedder -> col_embedder -> row_interactor -> col_embedder_2 -> row_interactor_2
    -> icl_predictor. A bare `tf_col` or `tf_row` would not say which of the two stages it
    is, so every stage is numbered in forward order:

    - `tf_col_1`, `tf_col_2`: the output of the set transformer of the first / second
      column stage, one vector per table cell, before the linear map and norm that follow
      it. `--feature_slot` reduces it to one vector per row, "mean" (the default here) or
      "flatten"; "target" has no meaning here and is read as "mean". The CLS slots and
      the padded columns are dropped.
    - `tf_row_1`, `tf_row_2`: the CLS tokens after the first / second row transformer,
      concatenated. The second is what the ICL transformer reads.
    - `tf_icl` (the default, also `--layer last`): the end of the ICL transformer, after
      its final norm, what the decoder reads.

    Only the first ensemble member is read, `n_estimators=1` keeps it the only one.
    """

    TAKES_SEED = True
    LAYER_CONTAINERS = ("icl_predictor.tf_icl.blocks",)
    DEFAULT_LAYER = "last"
    DEFAULT_MODEL_PATH = None
    LAYER_ALIASES = {
        "tf_col_1": "col_embedder.tf_col",
        "tf_row_1": "row_interactor",
        "tf_col_2": "col_embedder_2.tf_col",
        "tf_row_2": "row_interactor_2",
        "tf_icl": "icl_predictor.ln",
    }
    # The set transformers of the column stages, read by `_tf_col_embeddings`.
    COL_STAGES = ("col_embedder.tf_col", "col_embedder_2.tf_col")

    def __init__(self, model_path=None, n_estimators=1, seed=0, dict_params=None, **kwargs):
        """
        Args:
            model_path (str, optional): A local directory holding the checkpoint (with a
                `classification` subfolder or a config.json), None reads the Hugging Face
                cache of `google/tabfm-1.0.0-pytorch`, filled by down_weight.py, the
                compute nodes have no internet access. Defaults to None.
            n_estimators (int, optional): The size of the ensemble. 1 keeps a single
                member, without feature shuffle or class shift, so one deterministic
                forward pass. Defaults to 1.
            seed (int, optional): The random state of the classifier. Defaults to 0.
            dict_params (dict, optional): Extra parameters forwarded to TabFMClassifier.
                Defaults to None.
            kwargs (dict): The parameters of BaseTabularPFM (device, layer, feature_slot).
        """
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.seed = seed
        self.dict_params = {} if dict_params is None else dict_params
        super().__init__(**kwargs)

    def _build_classifier(self):
        tabfm = _import_tabfm()

        network = tabfm.tabfm_v1_0_0_pytorch.load(
            model_type="classification", checkpoint_path=self.model_path, device=self.device
        )
        return tabfm.TabFMClassifier(
            model=network,
            n_estimators=self.n_estimators,
            random_state=self.seed,
            **self.dict_params,
        )

    def _torch_model(self):
        return self.clf.model

    def _native_embeddings(self, X_test):
        return self._hooked_embeddings(X_test, self.LAYER_ALIASES["tf_icl"])

    def _hooked_embeddings(self, X_test, layer):
        name, _ = self._resolve_layer(layer)
        if name in self.COL_STAGES:
            return self._tf_col_embeddings(X_test, name)
        return super()._hooked_embeddings(X_test, layer)

    def _reduce_activations(self, activations, n_test, layer_name=None):
        if layer_name == self.LAYER_ALIASES["tf_row_1"]:
            # The first row stage outputs the CLS tokens and the features, keep the CLS
            # tokens: (batch, rows, cls + features, dim) -> (batch, rows, cls * dim).
            n_cls = self._torch_model().cls_tokens.shape[0]
            activations = activations[..., :n_cls, :].flatten(start_dim=-2)
        return super()._reduce_activations(activations, n_test, layer_name)

    def _tf_col_embeddings(self, X_test, name):
        """Capture the output of a column-wise set transformer during a forward pass.

        The columns are processed in chunks, which call the set transformer once per chunk,
        so the calls of the last forward pass are gathered and put back together.
        """
        import torch

        model = self._torch_model()
        stage_name = name.rsplit(".", 1)[0]
        stage = model.get_submodule(stage_name)
        # The second column stage also sees the CLS slots, the first one does not.
        n_reserved = model.cls_tokens.shape[0] if stage_name == "col_embedder_2" else 0

        def on_tf_col(_module, _inputs, output):
            output = output[0] if isinstance(output, tuple) else output
            events.append(("tf_col", output.detach().to("cpu", torch.float32, copy=True)))

        # ("tf_col", output) per call, ("stage", output shape) once the stage is assembled,
        # ("pass", n_features) at the end of each forward pass.
        events = []
        handles = [
            stage.tf_col.register_forward_hook(on_tf_col),
            stage.register_forward_hook(
                lambda _module, _inputs, output: events.append(("stage", tuple(output.shape)))
            ),
            model.register_forward_hook(
                lambda _module, _args, kwargs, _output: events.append(("pass", kwargs.get("d"))),
                with_kwargs=True,
            ),
        ]
        try:
            self.clf.predict_proba(X_test)
        finally:
            for handle in handles:
                handle.remove()

        ends = [i for i, (kind, _) in enumerate(events) if kind == "pass"]
        if not ends:
            raise RuntimeError(f"The hook on {name} never fired during predict_proba")
        start = ends[-2] + 1 if len(ends) > 1 else 0
        window = events[start : ends[-1]]
        chunks = [value for kind, value in window if kind == "tf_col"]
        n_tables, _, n_columns, _ = next(value for kind, value in window if kind == "stage")
        n_features = events[ends[-1]][1]

        # Each call is shaped (columns of the chunk, rows, dim), the columns of all the
        # tables being stacked table after table.
        columns = torch.cat(chunks)
        if columns.shape[0] != n_tables * n_columns:
            raise RuntimeError(
                f"Captured {columns.shape[0]} column embeddings from {name}, expected "
                f"{n_tables * n_columns} ({n_tables} tables x {n_columns} columns)"
            )

        # First table only: (columns, rows, dim) -> (1, rows, features, dim), leaving out
        # the CLS slots and the columns padded past the active features.
        n_active = int(n_features[0]) if n_features is not None else n_columns - n_reserved
        columns = columns.reshape(n_tables, n_columns, *columns.shape[-2:])[0]
        activations = columns[n_reserved : n_reserved + n_active].transpose(0, 1).unsqueeze(0)
        if self.verbose:
            print(f"  captured {name}: {tuple(activations.shape)}")
            self.verbose = False  # only report the first capture of a run

        activations = self._reduce_cells_without_target(activations, "tf_col")
        return self._reduce_activations(activations, len(X_test))
