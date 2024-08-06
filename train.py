
from utils import *

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, average_precision_score

import pandas as pd
import os
import random
import time
import gc

import argparse



def run_logistic_regression(
        x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
        fhe_state:bool=False, n_iterations:int=512,
        random_state:int=42, debug:bool=False, n_runs:int=10
    ) -> dict:
    """
    Run logistic regression on the data.
    Args:
        x_train: (np.ndarray) [n_samples, n_features]
        x_test: (np.ndarray) [n_samples, n_features]
        y_train: (np.ndarray) [n_samples]
        y_test: (np.ndarray) [n_samples]
        fhe_state: (bool)                                   
            if True, use the FHE state
        n_iterations: (int)                                 
            number of iterations for the logistic regression
        random_state: (int)                                 
            seed for the random number generator
        debug: (bool)                                       
            if True, only run on one split
        n_runs: (int) 
            number of runs to average the results
    Returns:
        results: (dict)
    """
    # Perform a random permutation of the data with a fixed seed
    set_seed(random_state)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(x_train.shape[0])
    x_train = x_train[perm, ::]
    y_train = y_train[perm]

    results = run_sgd_classification(
        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, 
        fhe_state=fhe_state, n_iterations=n_iterations, random_state=random_state, debug=debug, n_runs=n_runs, 
    )

    gc.collect()
    return results

def main(fhe_state = False, n_iterations=512, random_state=42, csv=False, debug=False, continue_from=None, n_runs=10):
    print(f"Running with FHE state: {fhe_state}, debug: {debug}")
    set_seed(random_state)
    splits = ["A", "B", "C", "D"]
    train_splits = combinations[:] # combinations is defined in utils.py
    test_splits = splits[:]
    # Save the data in the memory to avoid loading it multiple times. Only for parquet!
    if not csv: cached_data = load_all_data_to_memory(csv=csv, debug=debug)
    results_list = []
    uid = 0
    for train_split in train_splits:
        # Load the training data
        split_info = "".join(train_split)
        if debug and split_info != "B": continue
        print(f"Training on splits: {split_info}")
        train_x_list = []
        train_y_list = []
        for split in train_split:
            if debug and split != "B": continue
            # use cached data if it is already loaded (for parquet), otherwise load it
            x, y = load_data(split=split, mode="train", csv=csv) if csv else cached_data[f"{split}_train"]
            train_x_list.append(x)
            train_y_list.append(y)
        x_train_original = np.concatenate(train_x_list, axis=0)
        y_train_original = np.concatenate(train_y_list, axis=0)

        for test_split in test_splits:
            if continue_from is not None and uid < continue_from:
                # read the results from the file
                results = pd.read_csv(f"data/temp_data_fhe/{uid:03d}.csv") if fhe_state else pd.read_csv(f"temp_data/{uid:03d}.csv")
                # from pandas to dict, there is only one row
                results = results.to_dict(orient="records")[0]
                print(results)
                results_list.append(results)

                uid += 1
                continue
            # Load the test data
            split_info = "".join(test_split)
            if debug and split_info != "B": continue
            print(f"Testing on splits: {split_info}")

            # use cached data if it is already loaded (for parquet), otherwise load it
            x_test, y_test = load_data(split=test_split, mode="test", csv=csv) if csv else cached_data[f"{split_info}_test"]

            # Copy the data to avoid modifying the original data
            x_train = x_train_original.copy()
            x_test = x_test.copy()
            y_train = y_train_original.copy()
            y_test = y_test.copy()

            # Normalize the data
            x_train, x_test = normalize_data(x_train, x_test, random_state=random_state)

            print(f"Training data shape: {x_train.shape}", f"Testing data shape: {x_test.shape}")
            # Run the logistic regression
            start = time.time()
            results = run_logistic_regression(
                x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, 
                fhe_state=fhe_state,n_iterations=n_iterations, random_state=random_state, debug=debug)
            end = time.time()
            results["train_split"] = "".join(train_split)
            results["test_split"] = "".join(test_split)
            results["id"] = uid

            # save the results to .csv file
            if not debug:
                df = pd.DataFrame([results])
                if fhe_state:
                    df.to_csv(f"data/temp_data_fhe/{uid:03d}.csv", index=False)
                else:
                    df.to_csv(f"data/temp_data/{uid:03d}.csv", index=False)

            print(results)
            print(f"Time taken: {end-start}")
            print("------")

            results_list.append(results)
            uid += 1

            del x_train, x_test, y_train, y_test
            
            if debug: break
        if debug: break
    
    if not debug:
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f"data/results_fhe_{fhe_state}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fhe_state", type=bool, default=False, help="if True, use the FHE state")
    parser.add_argument("--debug", type=bool, default=False, help="if True, only run on one split")
    parser.add_argument("--csv", type=bool, default=False, help="if True, load the data from csv files, otherwise from parquet files. Default is False")
    parser.add_argument("--random_state", type=int, default=42, help="seed for the random number generator")
    parser.add_argument("--n_iterations", type=int, default=512, help="number of iterations for the logistic regression")
    parser.add_argument("--n_runs", type=int, default=10, help="number of runs to average the results")
    args = parser.parse_args()

    path = "data/temp_data_fhe" if args.fhe_state else "data/temp_data"
    files = sorted(os.listdir(path))
    if not files or args.debug:
        last_uid = None
    else:
        last_uid = int(files[-1].split(".")[0])
    main(
        fhe_state=args.fhe_state, csv=args.csv, 
        n_iterations=args.n_iterations, 
        random_state=args.random_state, debug=args.debug,
        continue_from=last_uid,
        n_runs=args.n_runs, # 
    )
    # sbatch --partition=long --time=16:00:00  -A pmlr-24 --wrap "python train.py" --output  "./out.txt" --error="./err.txt"