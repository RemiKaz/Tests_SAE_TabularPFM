from experiments.perturbations import (
    ColRemove,
    ColShuffle,
    RandomLabels,
    RowRemove,
    RowShuffle,
    ValueNoise,
)
from experiments.target_column import AltTarget

EXPERIMENTS = {
    experiment.name: experiment
    for experiment in (
        RowShuffle,
        ColShuffle,
        RandomLabels,
        ColRemove,
        RowRemove,
        ValueNoise,
        AltTarget,
    )
}

# Groups of experiments plotted together on a single figure.
EXPERIMENT_GROUPS = {
    # The 2x3 grid of test.py, in its original panel order.
    "perturbations": [
        "row_shuffle",
        "col_shuffle",
        "random_labels",
        "col_remove",
        "row_remove",
        "value_noise",
    ],
    "all": list(EXPERIMENTS),
}


def experiment_importer(exp_name, dict_params=None):
    """Import the experiments to run under the given name.

    Args:
        exp_name (str): The name of an experiment or of a group of experiments,
            in EXPERIMENTS or EXPERIMENT_GROUPS.
        dict_params (dict, optional): Parameters forwarded to the experiments,
            to do ablations. Defaults to None.

    Returns:
        list: The instantiated experiments.

    Raises:
        ValueError: If the experiment is not implemented.
    """
    print(f"Importing experiment {exp_name} ...")
    dict_params = {} if dict_params is None else dict_params

    if exp_name in EXPERIMENT_GROUPS:
        names = EXPERIMENT_GROUPS[exp_name]
    elif exp_name in EXPERIMENTS:
        names = [exp_name]
    else:
        raise ValueError(
            f"Experiment {exp_name} not implemented ! Available: "
            f"{list(EXPERIMENTS) + list(EXPERIMENT_GROUPS)}"
        )

    return [EXPERIMENTS[name](**dict_params) for name in names]
