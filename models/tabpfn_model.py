import glob
import os

import numpy as np

from models.base_model import BaseTabularPFM


def find_hf_checkpoint(repo_id, filename, env_var):
    """Locate a checkpoint in the HF cache, whatever the snapshot hash is.

    The checkpoints are downloaded by down_weight.py, the compute nodes have no internet
    access. The cache is resolved from $HF_HOME (else $WORK/hf_cache) so it follows the
    default project, and `env_var` overrides the whole path.

    Args:
        repo_id (str): The HuggingFace repo of the checkpoint.
        filename (str): The name of the checkpoint file in the repo.
        env_var (str): The environment variable that overrides the path.

    Returns:
        str: The path of the checkpoint, or the path it is expected at when not found.
    """
    if os.environ.get(env_var):
        return os.environ[env_var]
    snapshots = os.path.join(
        os.environ.get("HF_HOME") or os.path.join(os.environ.get("WORK", ""), "hf_cache"),
        "hub",
        "models--" + repo_id.replace("/", "--"),
        "snapshots",
    )
    found = sorted(glob.glob(os.path.join(snapshots, "*", filename)))
    return found[-1] if found else os.path.join(snapshots, "<hash>", filename)


class TabPFN(BaseTabularPFM):
    """Shared TabPFN wrapper, whose final embeddings are exposed by `get_embeddings`.

    The versions live in tabpfn_v2_model.py and tabpfn_v3_model.py.
    """

    # Tried in order, the first one found gives the layer stack of the transformer.
    LAYER_CONTAINERS = (
        "transformer_encoder.layers",
        "transformer_encoder",
        "encoder.layers",
        "model.transformer_encoder.layers",
    )
    DEFAULT_LAYER = "last"

    def __init__(self, model_path, n_estimators=1, dict_params=None, **kwargs):
        """
        Args:
            model_path (str): The path of the checkpoint, its file name tells the library which
                TabPFN version to load.
            n_estimators (int, optional): The size of the ensemble. 1 keeps a single
                deterministic forward pass. Defaults to 1.
            dict_params (dict, optional): Extra parameters forwarded to TabPFNClassifier.
                Defaults to None.
            kwargs (dict): The parameters of BaseTabularPFM (device, layer, feature_slot).
        """
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.dict_params = {} if dict_params is None else dict_params
        super().__init__(**kwargs)

    def _build_classifier(self):
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(
            model_path=self.model_path,
            device=self.device,
            n_estimators=self.n_estimators,
            **self.dict_params,
        )

    def _torch_model(self):
        model = getattr(self.clf, "model_", None) or getattr(self.clf, "model", None)
        if model is None:
            raise RuntimeError("The classifier has to be fitted before its layers can be read")
        return model

    def _native_embeddings(self, X_test):
        return np.asarray(self.clf.get_embeddings(X_test))

