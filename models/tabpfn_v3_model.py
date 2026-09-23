from models.tabpfn_model import TabPFN, find_hf_checkpoint

# TABPFN_V3_CKPT overrides the path.
DEFAULT_MODEL_PATH = find_hf_checkpoint(
    "Prior-Labs/tabpfn_3", "tabpfn-v3-classifier-v3_default.ckpt", "TABPFN_V3_CKPT"
)


class TabPFNV3(TabPFN):
    """TabPFN v3, whose stages mirror the ones of TabICL and are read with a forward hook.

    - `tf_col`: the feature distribution embedder, one vector per table cell, shaped
      (rows, feature groups, dim). `--feature_slot` reduces it to one vector per row,
      "mean" (the default here) or "flatten"; "target" has no meaning here and is read
      as "mean".
    - `tf_row`: the column aggregator, the CLS tokens concatenated as the model does
      before the ICL stage, one vector per row.
    - `tf_icl`: the end of the ICL transformer, after its output norm, what the decoder
      reads. One vector per row.
    """

    # The main (in-context) transformer, addressed by `--layer <index>`. The column
    # aggregator and the feature embedder have their own stacks, reachable by name.
    LAYER_CONTAINERS = ("icl_blocks",) + TabPFN.LAYER_CONTAINERS
    DEFAULT_MODEL_PATH = DEFAULT_MODEL_PATH
    STAGES = {
        "tf_col": "feature_distribution_embedder",
        "tf_row": "column_aggregator",
        "tf_icl": "output_norm",
    }
    LAYER_ALIASES = STAGES

    def _hooked_embeddings(self, X_test, layer):
        name, _ = self._resolve_layer(layer)
        if name in self.STAGES.values():
            return self._stage_embeddings(X_test, name)
        return super()._hooked_embeddings(X_test, layer)

    def _stage_embeddings(self, X_test, name):
        """Capture the output of one stage during a forward pass on `X_test`.

        Long tables are processed in row chunks, which call the stage once per chunk, so
        the calls of the last forward pass are gathered and put back together.
        """
        import torch

        model = self._torch_model()
        # ("stage", output) per call of the stage, ("pass", None) at the end of each pass.
        events = []

        def on_stage(_module, _inputs, output):
            output = output[0] if isinstance(output, tuple) else output
            events.append(("stage", output.detach().to("cpu", torch.float32, copy=True)))

        handles = [
            model.get_submodule(name).register_forward_hook(on_stage),
            model.register_forward_hook(lambda *_: events.append(("pass", None))),
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
        chunks = [value for kind, value in events[start : ends[-1]] if kind == "stage"]
        if not chunks:
            raise RuntimeError(f"The hook on {name} never fired during the last forward pass")

        # (tables, rows, ...): the row chunks are put back in order along the rows.
        activations = torch.cat(chunks, dim=1)
        if self.verbose:
            print(f"  captured {name}: {tuple(activations.shape)}")
            self.verbose = False  # only report the first capture of a run

        if name == self.STAGES["tf_col"]:
            activations = self._reduce_cells_without_target(activations, "tf_col")
        elif name == self.STAGES["tf_row"]:
            activations = activations.flatten(start_dim=-2)  # the CLS tokens, concatenated
        return self._reduce_activations(activations, len(X_test))
