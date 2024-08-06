import pandas as pd
import os
import time

from train import return_path


if __name__ == "__main__":

    convert = False
    if convert:
        for split in ["A", "B", "C", "D"]:
            print(f"Processing split {split}")
            for mode in ["train", "test"]:
                for data_type in ["x", "y"]:
                    csv_path = return_path(split=split, mode=mode, data_type=data_type, csv=True)
                    parquet_path = return_path(split=split, mode=mode, data_type=data_type, csv=False)
                    df = pd.read_csv(csv_path)
                    df.to_parquet(parquet_path, compression=None)

    # check the efficiency of parquet
    compare = True
    
    if compare:
        csv_t = 0
        parquet_t = 0


        for split in ["A", "B", "C", "D"]:
            print(f"Processing split {split}")
            for mode in ["train", "test"]:
                for data_type in ["x", "y"]:
                    csv_path = return_path(split=split, mode=mode, data_type=data_type, csv=True)
                    parquet_path = return_path(split=split, mode=mode, data_type=data_type, csv=False)

                    start = time.time()
                    pd.read_csv(csv_path)
                    csv_time = time.time() - start

                    csv_t += csv_time

                    start = time.time()
                    pd.read_parquet(parquet_path)
                    parquet_time = time.time() - start

                    parquet_t += parquet_time
        print(f"CSV time: {csv_t}")
        print(f"Parquet time: {parquet_t}")
        print(f"Parquet is {csv_t/parquet_t} times faster than CSV")

