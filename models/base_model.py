from abc import ABC, abstractmethod

import numpy as np


class BaseTabularPFM(ABC):
    """Common interface of the tabular foundation models used in the experiments.

    A model wraps a scikit-learn-like classifier (fit / predict_proba) and adds
    `get_embeddings`, which returns the representation of the test rows at the
    requested layer position.

    Attributes:
        LAYER_CONTAINERS (tuple): Dotted paths tried, in order, to locate the stack
            of layers of the underlying torch model. The children of the first one
            found are the layers addressed by an integer `layer`.
        DEFAULT_LAYER (str): The layer used when `layer` is "last".
        LAYER_ALIASES (dict): Short names accepted as `layer`, mapped to the dotted name
            of the submodule they stand for.
        LAYER_FRACTIONS (dict): Named depths accepted as `layer`, mapped to the fraction
            of the layer stack they stand for, for the models made of one flat stack of
            identical layers, e.g. {"early": 0.25, "middle": 0.5}. "early" reads the
            output of the layer that closes the first quarter of the stack.
        TAKES_SEED (bool): Whether the wrapper takes a `seed`, the random state of the
            classifier.
    """

    LAYER_CONTAINERS = ()
    DEFAULT_LAYER = "last"
    LAYER_ALIASES = {}
    LAYER_FRACTIONS = {}
    TAKES_SEED = False  # whether the wrapper takes the `seed` of the classifier

    def __init__(self, device="cuda", layer="last", feature_slot="target", verbose=True):
        """
        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
            layer (str or int, optional): The layer position the embeddings are read at,
                either "last" (the native embeddings of the model), an integer index in
                the layer stack, or the dotted name of any submodule. Defaults to "last".
            feature_slot (str, optional): How per-cell representations are reduced to one
                vector per row, in [target, mean, flatten]. Only used by models whose
                internal activations are shaped (batch, rows, features, dim).
                Defaults to "target".
            verbose (bool, optional): Whether to print the captured shapes. Defaults to True.
        """
        self.device = device
        self.layer = layer
        self.feature_slot = feature_slot
        self.verbose = verbose
        self.clf = self._build_classifier()

    @abstractmethod
    def _build_classifier(self):
        """Build the underlying scikit-learn-like classifier."""

    @abstractmethod
    def _torch_model(self):
        """Return the underlying torch module of the fitted classifier."""

    def _native_embeddings(self, X_test):
        """Return the embeddings the model exposes itself, without any hook."""
        raise NotImplementedError(
            f"{type(self).__name__} has no native embeddings, pass an explicit --layer"
        )

    def fit(self, X_train, y_train):
        """Fit the classifier on a training set.

        Args:
            X_train (np.ndarray): The train features.
            y_train (np.ndarray): The train labels.

        Returns:
            BaseTabularPFM: self.
        """
        self.clf.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def predict_proba(self, X_test):
        return self.clf.predict_proba(X_test)

    def get_embeddings(self, X_test):
        """Representations of the test rows at the layer position of the model.

        Args:
            X_test (np.ndarray): The test features.

        Returns:
            np.ndarray: The embeddings, of shape (n_test, dim) when read with a hook.
        """
        layer = self.DEFAULT_LAYER if self.layer in ("last", None) else self.layer
        if layer == "last":
            return self._native_embeddings(X_test)
        return self._hooked_embeddings(X_test, layer)

    def list_layers(self):
        """Names of the submodules the embeddings can be read at.

        The model has to be fitted first, since the torch module is only built then.

        Returns:
            list: The (index, name) pairs of the layer stack, and all submodule names.
        """
        model = self._torch_model()
        try:
            stack = self._layer_stack(model)
        except AttributeError:
            stack = []  # the submodule names below are still enough to pick a layer
        return {
            "layer_stack": [(i, name) for i, (name, _) in enumerate(stack)],
            "depths": {
                depth: stack[self._depth_index(len(stack), fraction)][0]
                for depth, fraction in self.LAYER_FRACTIONS.items()
            }
            if stack
            else {},
            "all_modules": [name for name, _ in model.named_modules() if name],
        }

    @staticmethod
    def _depth_index(n_layers, fraction):
        """Index of the layer closing the first `fraction` of a stack of `n_layers`."""
        return min(n_layers - 1, max(0, int(n_layers * fraction) - 1))

    def _layer_stack(self, model):
        """Locate the stack of layers, i.e. the children of the first container found."""
        for path in self.LAYER_CONTAINERS:
            container = _resolve_path(model, path)
            if container is not None:
                return [(f"{path}.{name}", module) for name, module in container.named_children()]
        raise AttributeError(
            f"None of the layer containers {self.LAYER_CONTAINERS} were found on "
            f"{type(model).__name__}; run with --list_layers to see the available submodules"
        )

    def _resolve_layer(self, layer):
        """Turn a layer spec into the (name, module) it addresses."""
        model = self._torch_model()
        if isinstance(layer, str) and layer in self.LAYER_FRACTIONS:
            stack = self._layer_stack(model)
            return stack[self._depth_index(len(stack), self.LAYER_FRACTIONS[layer])]
        layer = self.LAYER_ALIASES.get(layer, layer) if isinstance(layer, str) else layer
        if isinstance(layer, int) or (isinstance(layer, str) and layer.lstrip("-").isdigit()):
            stack = self._layer_stack(model)
            index = int(layer)
            if not -len(stack) <= index < len(stack):
                raise IndexError(f"Layer {index} out of range, the stack has {len(stack)} layers")
            return stack[index]

        module = _resolve_path(model, str(layer))
        if module is None:
            raise AttributeError(
                f"Submodule {layer} not found; run with --list_layers to see the available ones"
            )
        return str(layer), module

    def _hooked_embeddings(self, X_test, layer):
        """Capture the output of one submodule during a forward pass on `X_test`."""
        name, module = self._resolve_layer(layer)
        captured = []

        def hook(_module, _inputs, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        handle = module.register_forward_hook(hook)
        try:
            self.clf.predict_proba(X_test)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(f"The hook on {name} never fired during predict_proba")

        activations = captured[-1].detach().float().cpu()
        if self.verbose:
            print(f"  captured {name}: {tuple(activations.shape)}")
            self.verbose = False  # only report the first capture of a run
        return self._reduce_activations(activations, len(X_test), name)

    def _reduce_cells_without_target(self, activations, layer_name):
        """Reduce the feature axis of stage activations that have no target cell.

        Args:
            activations (torch.Tensor): The captured activations, shaped
                (batch, rows, n_features, dim).
            layer_name (str): The stage the activations come from, for the notice.

        Returns:
            torch.Tensor: Shaped (batch, rows, dim), or (batch, rows, n_features * dim)
                with feature_slot "flatten". "target" has no cell to pick here, it is
                read as "mean".
        """
        if self.feature_slot == "flatten":
            return activations.flatten(start_dim=-2)
        if self.feature_slot == "target" and not getattr(self, "_slot_note_shown", False):
            print(f"  {layer_name} has no target cell, reading feature_slot 'target' as 'mean'")
            self._slot_note_shown = True
        return activations.mean(dim=-2)

    def _reduce_activations(self, activations, n_test, layer_name=None):
        """Reduce captured activations to one embedding vector per test row.

        Args:
            activations (torch.Tensor): The captured activations, shaped
                (..., n_rows, dim) or (..., n_rows, n_features, dim).
            n_test (int): The number of test rows.
            layer_name (str, optional): The submodule the activations were captured at,
                for the models that reduce some of them differently. Defaults to None.

        Returns:
            np.ndarray: The embeddings, of shape (n_test, dim).
        """
        if activations.ndim == 4:
            # (batch, rows, cells, dim): one representation per table cell.
            if self.feature_slot == "target":
                activations = activations[..., -1, :]
            elif self.feature_slot == "mean":
                activations = activations.mean(dim=-2)
            elif self.feature_slot == "flatten":
                activations = activations.flatten(start_dim=-2)
            else:
                raise ValueError(f"Unknown feature_slot {self.feature_slot}")

        activations = activations.reshape(-1, activations.shape[-2], activations.shape[-1])
        # The sequence axis holds the train rows followed by the test rows.
        activations = activations[:, -n_test:, :].mean(dim=0)
        return activations.numpy()


def _resolve_path(module, path):
    """Follow a dotted attribute/submodule path, returning None if it does not exist."""
    current = module
    for attribute in path.split("."):
        if hasattr(current, attribute):
            current = getattr(current, attribute)
        elif attribute.isdigit() and hasattr(current, "__getitem__"):
            try:
                current = current[int(attribute)]
            except (IndexError, KeyError, TypeError):
                return None
        else:
            return None
    return current
