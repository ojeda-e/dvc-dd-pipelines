import os
import sys
import pandas as pd
import numpy as np
import yaml

params = yaml.safe_load(open("params.yaml"))["process"]

threshold = params["threshold"]
filter = params["filter_by"]

input = sys.argv[1]
output = sys.argv[2]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython process.py data-file-in  data-file-out \n")
    sys.exit(1)

def remove_zeros(df, threshold: int):
    """
    Remove features that matches the threshold provided.
    """
    null_features=[]
    for x in range(0, len(df.columns)):
        col=df.columns[x]
        N_zeroes=len(df.loc[[i == 0.0 for i in df[col]]])
        if N_zeroes > threshold:
            null_features.append(col)

    print(f"Removing {len(null_features)} features.")
    df=df.drop(null_features, axis=1)
    return df


def filter_correlation(df, filter_by: float = 0.9):
    """
    Filters correlated features in a DataFrame.

    Parameters:
    ------------
    df: pd.DataFrame
        Dataframe to filter correlated features.
    filter_by: float
        Threshold to filter correlation. Range [0,1].

    Returns:
    -----------
    df: pd.DataFrame
        Filtered pandas DataFrame
    """
    corr_mx=df.corr()  
    newColumns=[df.columns[0]]  
    for colx in df.columns[1:]:
        if (np.abs(corr_mx.loc[colx, newColumns]) < filter_by).all():
            newColumns.append(colx)
        else:
            print(f"Removing column {colx}")
            df=df.drop(colx, axis=1)
    return df

os.makedirs(os.path.join("data", "processed"), exist_ok=True)

df_input = pd.read_csv(input)
descriptors_df = remove_zeros(df_input, threshold)
prepared_df = filter_correlation(descriptors_df, filter_by=filter)
prepared_df.to_csv(output)
print('The shape of the filtered dataframe is: ', prepared_df.shape, '\n')