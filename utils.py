import os, sys

import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, average_precision_score

combinations = [
    ["A"], ["B"], ["C"], ["D"], 
    ["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"], ["C", "D"], 
    ["A", "B", "C"], ["A", "B", "D"], ["A", "C", "D"], ["B", "C", "D"], 
    ["A", "B", "C", "D"]
]


def set_seed(seed:int=42):
    """
    Set the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

def return_path(split:str="A", mode:str="train", data_type:str="x", csv:bool=True) -> str:
    """
    Return the path to the data file.
    Args:
        split: (str) [A, B, C, D]
        mode: (str) [train, test]
        data_type: (str) [x, y]
        csv: (bool) If True, return the path to the CSV file, otherwise to the Parquet file.
    Returns:
        path: str
    """
    data_path = "/cluster/courses/pmlr-24/data/team-6"
    if csv:
        data_path = os.path.join(data_path, "data")
    else:
        data_path = os.path.join(data_path, "data_parquet")
    assert split in ["A", "B", "C", "D"]
    assert mode in ["train", "test"]
    assert data_type in ["x", "y"]
    assert os.path.exists(data_path), f"Path does not exist: {data_path}"
    path = f"driams{split.lower()}_Staph_{mode}_{data_type}"
    path = f"{path}.csv" if csv else f"{path}.parquet"

    path = os.path.join(data_path, path)
    assert os.path.exists(path), f"Path does not exist: {path}"
    return path

def load_data(split:str="A", mode:str="train", csv:bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the data from the specified split and mode.
    Args:
        split: (str) [A, B, C, D]
        mode: (str) [train, test]
        csv: (bool) If True, load the data from CSV, otherwise from Parquet.
    Returns:
        x_np: (np.ndarray) [n_samples, n_features]
        y_np: (np.ndarray) [n_samples]
    """
    x_path = return_path(split=split, mode=mode, data_type="x", csv=csv)
    y_path = return_path(split=split, mode=mode, data_type="y", csv=csv)

    x_df = pd.read_csv(x_path) if csv else pd.read_parquet(x_path)
    y_df = pd.read_csv(y_path) if csv else pd.read_parquet(y_path)

    x_np = x_df.to_numpy()
    y_np = y_df.iloc[:, 0].to_numpy()
    return (x_np, y_np)



def load_all_data_to_memory(csv:bool=False, debug:bool=False) -> dict:
    """
    Load all the data to memory. This is useful when the data fits in memory.
    ! This function should not be used with CSV data as it takes too much memory.
    Args:
        csv: (bool) If True, load the data from CSV, otherwise from Parquet.
    Returns:
        cache: (dict) A dictionary containing the data for all splits and modes.
    """
    assert not csv, "Loading all data to memory with CSV takes too much memory."
    cache = {}
    for split in ["A", "B", "C", "D"]:
        if debug and split != "B": continue
        for mode in ["train", "test"]:
            cache[f"{split}_{mode}"] = load_data(split=split, mode=mode, csv=csv)
        
    return cache



def normalize_data(x_train: np.ndarray, x_test: np.ndarray, random_state:int=42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply PCA only to the first 6000 features of the data.
    But before PCA normalize the data using mean and variance across the features.
    Args:
        x_train: (np.ndarray)
            [n_samples, n_features]
            n_features = 6167: n_mdtof(6000) + n_other (167)
        x_test: (np.ndarray) [n_samples, n_features]
    Returns:
        x_train_new: (np.ndarray)
            [n_samples, n_new_features]
            n_new_features = 384: n_mdtof(217) + n_other (167)
        x_test_new: (np.ndarray) [n_samples, n_new_features]
    """ 
    set_seed(random_state)
    n_features = x_train.shape[1]
    assert n_features == 6167, f"Expected 6167 features, got {n_features}"
    n_mdtof_old = 6000 # PCA to 217 (384 - 167)
    n_other = 167
    n_new_features = 384

    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Fit the PCA on the first 6000 features
    n_mdtof_new = n_new_features - n_other
    # Select the first 6000 features
    x_train_ = x_train[:, :n_mdtof_old]
    x_test_ = x_test[:, :n_mdtof_old]

    # PCA on the first 6000 features
    pca = PCA(n_components=n_mdtof_new, random_state=random_state)
    x_train_ = pca.fit_transform(x_train_)
    x_test_ = pca.transform(x_test_)
    # Concatenate the PCA features with the other original features (167)
    # Train data
    x_train_new = np.zeros((x_train.shape[0], n_new_features))
    x_train_new[:, :n_mdtof_new] = x_train_
    x_train_new[:, -n_other:] = x_train[:, -n_other:]
    # Test data
    x_test_new = np.zeros((x_test.shape[0], n_new_features))
    x_test_new[:, :n_mdtof_new] = x_test_
    x_test_new[:, -n_other:] = x_test[:, -n_other:]

    return x_train_new, x_test_new

def run_sgd_classification(
        x_train:np.ndarray, x_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray,
        fhe_state:bool=False, n_runs:int=10, random_state:int=42, n_iterations:int=512, debug:bool=False):
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train.copy())
    x_test = scaler.transform(x_test.copy())

    accuracy_scores = []
    balanced_accuracy_scores = []
    f1_scores = []
    average_precision_scores = []

    for i in range(n_runs):
        if debug: print(f"Run: {i+1}/{n_runs}")
        set_seed(random_state + i)
        if fhe_state:
            from concrete.ml.sklearn import SGDClassifier
            sgd_clf = SGDClassifier(
                n_bits=8,
                random_state=random_state + i,
                max_iter=n_iterations,
                fit_encrypted=True,
                parameters_range=(-1.0, 1.0),
                verbose=False,
            )

            sgd_clf.fit(x_train, y_train, fhe="simulate")
            sgd_clf.compile(x_train)
            y_pred = sgd_clf.predict(x_test, fhe="simulate")
        else:
            from sklearn.linear_model import SGDClassifier
            sgd_clf = SGDClassifier(
                random_state=random_state + i,
                max_iter=n_iterations,
            )
            sgd_clf.fit(x_train, y_train)
            y_pred = sgd_clf.predict(x_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        average_precision_scores.append(average_precision_score(y_test, y_pred))

        del sgd_clf

    results = {
        "accuracy_mean": np.mean(accuracy_scores),
        "accuracy_std": np.std(accuracy_scores),
        "balanced_accuracy_mean": np.mean(balanced_accuracy_scores),
        "balanced_accuracy_std": np.std(balanced_accuracy_scores),
        "f1_score_mean": np.mean(f1_scores),
        "f1_score_std": np.std(f1_scores),
        "average_precision_score_mean": np.mean(average_precision_scores),
        "average_precision_score_std": np.std(average_precision_scores),
    }

    return results

if __name__ == "__main__":
    print("Utils")