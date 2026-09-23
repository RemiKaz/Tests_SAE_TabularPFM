from models.base_model import BaseTabularPFM


class TabICL(BaseTabularPFM):
    """Shared TabICL wrapper, whose row representations only exist inside the forward pass.

    The versions live in tabicl_v1_model.py and tabicl_v2_model.py.

    Three representations are read with a forward hook, named as in the TabICL papers
    (the stages are the same in v1 and v2):

    - `tf_col`: the output of the column-wise set transformer, one vector per table cell,
      shaped (rows, features, dim). It is read before the affine map that turns it into
      the cell embeddings fed to the row transformer. `--feature_slot` reduces it to one
      vector per row, "mean" (the default here) or "flatten"; "target" has no meaning
      here, since TabICL has no target cell, and is read as "mean".
    - `tf_row`: the output of the row-wise transformer, the concatenated CLS tokens, one
      vector of 4 x dim per row.
    - `tf_icl` (the default, also `--layer last`): the end of the in-context-learning
      transformer, i.e. its output after the final LayerNorm, which is what the decoder
      reads to predict. One vector per row.

    `tf_col` and `tf_row` are each a stack of blocks (`col_embedder.tf_col.blocks`,
    `row_interactor.tf_row.blocks`), read at their very end. `tf_col_early` / `tf_col_middle`
    and `tf_row_early` / `tf_row_middle` read them at an earlier block instead (closing the
    first quarter / half of that stack), the same way as the stage's own final output: the
    per-cell mean/flatten for `tf_col_*`, the CLS-token slots for `tf_row_*` (every block
    writes into the same reserved slots, refining them further each time, so reading them
    mid-stack gives an earlier snapshot of the same representation `tf_row` itself reads).
    `tf_row`'s last block does the CLS-only pooling and has no equivalent mid-stack reading,
    so `tf_row_early` / `tf_row_middle` are computed over its other blocks only.
    """

    TAKES_SEED = True
    LAYER_CONTAINERS = (
        "row_interactor.layers",
        "row_interactor",
        "tf_icl.layers",
        "icl_predictor.layers",
    )
    # "last" is the end of TF_icl, see `_native_embeddings`.
    DEFAULT_LAYER = "last"
    # The checkpoint TabICL downloads when no model_path is given, None keeps the library
    # default. Set by the version subclasses to pin the version.
    CHECKPOINT_VERSION = None
    DEFAULT_MODEL_PATH = None
    LAYER_ALIASES = {"tf_row": "row_interactor", "tf_col": "col_embedder.tf_col"}
    TF_COL_NAME = "col_embedder.tf_col"
    # Fractional depths of tf_col and tf_row, read like their stage's own final output.
    STAGE_DEPTHS = {"early": 0.25, "middle": 0.5}

    def __init__(self, model_path=None, n_estimators=1, seed=0, dict_params=None, **kwargs):
        """
        Args:
            model_path (str, optional): The path of the checkpoint, None lets TabICL
                download it from HuggingFace, which does not work from a compute node
                without internet access. Defaults to None.
            n_estimators (int, optional): The size of the ensemble. 1 keeps a single
                deterministic forward pass, so the captured representations are not an
                average over shuffled ensembles. Defaults to 1.
            seed (int, optional): The random state of the classifier. Defaults to 0.
            dict_params (dict, optional): Extra parameters forwarded to TabICLClassifier.
                Defaults to None.
            kwargs (dict): The parameters of BaseTabularPFM (device, layer, feature_slot).
        """
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.seed = seed
        self.dict_params = {} if dict_params is None else dict_params
        super().__init__(**kwargs)

    def _build_classifier(self):
        from tabicl import TabICLClassifier

        params = dict(self.dict_params)
        if self.CHECKPOINT_VERSION is not None:
            params.setdefault("checkpoint_version", self.CHECKPOINT_VERSION)
        return TabICLClassifier(
            model_path=self.model_path,
            allow_auto_download=self.model_path is None,
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.seed,
            **params,
        )

    def _torch_model(self):
        model = getattr(self.clf, "model_", None) or getattr(self.clf, "model", None)
        if model is None:
            raise RuntimeError("The classifier has to be fitted before its layers can be read")
        return model

    def _native_embeddings(self, X_test):
        return self._hooked_embeddings(X_test, self._tf_icl_module_name())

    def _tf_icl_module_name(self):
        """The module whose output is the end of TF_icl, just before the decoder."""
        # The final LayerNorm only exists for pre-norm models, otherwise TF_icl ends itself.
        has_final_norm = hasattr(self._torch_model().icl_predictor, "ln")
        return "icl_predictor.ln" if has_final_norm else "icl_predictor.tf_icl"

    def _hooked_embeddings(self, X_test, layer):
        for stage in ("tf_col", "tf_row"):
            for depth, fraction in self.STAGE_DEPTHS.items():
                if layer == f"{stage}_{depth}":
                    return self._stage_depth_embeddings(X_test, stage, fraction)
        if layer == "tf_icl":
            layer = self._tf_icl_module_name()
        name, _ = self._resolve_layer(layer)
        if name == self.TF_COL_NAME:
            return self._tf_col_embeddings(X_test)
        return super()._hooked_embeddings(X_test, layer)

    def _stage_depth_embeddings(self, X_test, stage, fraction):
        """Dispatch `tf_col_early` / `middle` and `tf_row_early` / `middle` to their stage."""
        model = self._torch_model()
        if stage == "tf_col":
            blocks = model.col_embedder.tf_col.blocks
            index = self._depth_index(len(blocks), fraction)
            return self._tf_col_embeddings(X_test, f"{self.TF_COL_NAME}.blocks.{index}")

        blocks = model.row_interactor.tf_row.blocks
        n_regular = len(blocks) - 1  # the last block does the CLS-only pooling, see tf_row
        if n_regular <= 0:
            raise ValueError(
                f"row_interactor.tf_row has only {len(blocks)} block(s), no block left to "
                f"read tf_row_early / tf_row_middle at, distinct from tf_row itself"
            )
        index = self._depth_index(n_regular, fraction)
        return self._row_block_embeddings(X_test, index)

    def _tf_col_embeddings(self, X_test, module_name=None):
        """Capture the output of a submodule of the column-wise stage during a forward pass.

        Defaults to the whole `col_embedder.tf_col` (TF_COL_NAME), its output after every
        block; `_stage_depth_embeddings` instead passes one of its own blocks' dotted name,
        for `tf_col_early` / `tf_col_middle`. Either way the inference manager may split the
        columns over several calls of that submodule, and a prediction may run several
        forward passes (one per ensemble batch), so the calls of the last pass are gathered
        and put back together.
        """
        import torch

        module_name = module_name or self.TF_COL_NAME
        model = self._torch_model()
        stage = model.get_submodule(module_name)
        col_embedder = model.col_embedder
        # ("tf_col", output) per call of `stage`, ("col", output shape) once the whole column
        # stage is assembled - marks the end of one pass, whatever `stage` is.
        events = []
        handles = [
            stage.register_forward_hook(
                lambda _module, _inputs, output: events.append(
                    ("tf_col", output.detach().float().cpu())
                )
            ),
            col_embedder.register_forward_hook(
                lambda _module, _inputs, output: events.append(("col", tuple(output.shape)))
            ),
        ]
        try:
            self.clf.predict_proba(X_test)
        finally:
            for handle in handles:
                handle.remove()

        ends = [i for i, (kind, _) in enumerate(events) if kind == "col"]
        if not ends:
            raise RuntimeError(f"The hook on {module_name} never fired during predict_proba")
        start = ends[-2] + 1 if len(ends) > 1 else 0
        chunks = [value for kind, value in events[start : ends[-1]] if kind == "tf_col"]
        n_tables, _, n_columns, _ = events[ends[-1]][1]

        # Each call is shaped (..., rows, dim), the leading dims index the columns.
        columns = torch.cat([chunk.reshape(-1, *chunk.shape[-2:]) for chunk in chunks])
        if columns.shape[0] == n_tables * n_columns:
            columns = columns.reshape(n_tables, n_columns, *columns.shape[-2:])
        elif columns.shape[0] == n_columns:
            # Feature-shuffled ensembles embed the columns once and reorder them per table.
            columns = columns.unsqueeze(0)
        else:
            raise RuntimeError(
                f"Captured {columns.shape[0]} column embeddings from {module_name}, "
                f"expected {n_tables * n_columns} ({n_tables} tables x {n_columns} columns)"
            )

        # (tables, reserved + features, rows, dim) -> (tables, rows, features, dim), where
        # the first columns are reserved for the CLS tokens and carry no feature.
        n_reserved = col_embedder.reserve_cls_tokens
        activations = columns[:, n_reserved:].transpose(1, 2)
        if self.verbose:
            print(f"  captured {module_name}: {tuple(activations.shape)}")
            self.verbose = False  # only report the first capture of a run

        activations = self._reduce_cells_without_target(activations, "tf_col")
        return self._reduce_activations(activations, len(X_test))

    def _row_block_embeddings(self, X_test, index):
        """Capture the CLS-token slots after one (non-final) block of the row transformer.

        Read the same way as `tf_row` itself (the concatenated CLS tokens), just at an
        earlier depth: every block writes into the same reserved slots the column stage set
        aside, refining them further with each subsequent block, so slicing them out of an
        intermediate block's raw (rows, cls + features, dim) output gives an earlier,
        comparable snapshot of the same representation. Long tables may be processed in row
        chunks, gathered from the last forward pass, same idea as `_tf_col_embeddings`.
        """
        import torch

        model = self._torch_model()
        row_interactor = model.row_interactor
        block = row_interactor.tf_row.blocks[index]
        module_name = f"row_interactor.tf_row.blocks.{index}"
        n_cls = row_interactor.num_cls

        # ("block", output) per call of `block`, ("pass", None) once row_interactor itself
        # returns - marks the end of one pass, its own forward hook fires after every nested
        # block hook it triggered.
        events = []
        handles = [
            block.register_forward_hook(
                lambda _module, _inputs, output: events.append(
                    ("block", output.detach().float().cpu())
                )
            ),
            row_interactor.register_forward_hook(lambda *_: events.append(("pass", None))),
        ]
        try:
            self.clf.predict_proba(X_test)
        finally:
            for handle in handles:
                handle.remove()

        ends = [i for i, (kind, _) in enumerate(events) if kind == "pass"]
        if not ends:
            raise RuntimeError(f"The hook on {module_name} never fired during predict_proba")
        start = ends[-2] + 1 if len(ends) > 1 else 0
        chunks = [value for kind, value in events[start : ends[-1]] if kind == "block"]
        if not chunks:
            raise RuntimeError(f"The hook on {module_name} never fired during the last forward pass")

        activations = torch.cat(chunks, dim=1)  # (tables, rows, cls + features, dim)
        activations = activations[..., :n_cls, :].flatten(start_dim=-2)  # the CLS tokens
        if self.verbose:
            print(f"  captured {module_name}: {tuple(activations.shape)}")
            self.verbose = False  # only report the first capture of a run

        return self._reduce_activations(activations, len(X_test))
