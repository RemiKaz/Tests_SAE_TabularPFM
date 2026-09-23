from sklearn import datasets as sklearn_datasets

from datasets.tabular_dataset import TabularDataset

# sklearn loaders usable as-is, i.e. returning a (X, y) classification problem.
SKLEARN_LOADERS = {
    "breast_cancer": sklearn_datasets.load_breast_cancer,
    "iris": sklearn_datasets.load_iris,
    "wine": sklearn_datasets.load_wine,
    "digits": sklearn_datasets.load_digits,
}


def dataset_importer(dataset_name, test_size=0.33, seed=42, dict_params=None):
    """Import the specified dataset and split it in a train and a test set.

    Args:
        dataset_name (str): The name of the dataset to import.
        test_size (float, optional): The proportion of the test set. Defaults to 0.33.
        seed (int, optional): The seed of the train/test split. Defaults to 42.
        dict_params (dict, optional): Extra parameters forwarded to the loader. Defaults to None.

    Returns:
        TabularDataset: The imported dataset.

    Raises:
        ValueError: If the dataset is not implemented.
    """
    print(f"Importing dataset {dataset_name} ...")
    dict_params = {} if dict_params is None else dict_params

    if dataset_name in SKLEARN_LOADERS:
        X, y = SKLEARN_LOADERS[dataset_name](return_X_y=True, **dict_params)
    else:
        raise ValueError(
            f"Dataset {dataset_name} not implemented ! Available: {list(SKLEARN_LOADERS)}"
        )

    dataset = TabularDataset.from_xy(dataset_name, X, y, test_size=test_size, seed=seed)
    print(
        f"  {dataset.n_train} train rows, {dataset.n_test} test rows, "
        f"{dataset.n_features} features"
    )
    return dataset
