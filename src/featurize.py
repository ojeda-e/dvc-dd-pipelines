import sys
import os

# scientific
import pandas as pd
import numpy as np
import re
# ML
from rdkit import Chem
from rdkit.Chem import Descriptors as Descript
from molvs import Standardizer as Std

import yaml

params = yaml.safe_load(open("params.yaml"))["featurize"]
max_number = params["max_number"]

input = sys.argv[1]
output = sys.argv[2]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython featurize.py data-file-in  data-file-out \n")
    sys.exit(1)


def get_descriptors(smiles: pd.core.series.Series) -> pd.DataFrame:
    """
    This function generates molecule objects from SMILES and calculates
    descriptors from standardized molecules.

    Parameters:
    ------------
    smiles: pd.core.series.Series
        Series of SMILES to calculate descriptor from.

    Returns:
    ----------
    descriptor_df: pd.DataFrame
        Pandas DataFrame with descriptors from standarized molecules.
    """
    for k, smi in enumerate(smiles):  # Explain what is this loop for
        mol = Chem.MolFromSmiles(smi)
        mol = Std().standardize(mol)
        descriptor_list = []
        desc_values = []
        blacklist = ['Ipc']
        for i in dir(Descript):  # Explain what is this loop for
            # Explain what is this for
            i_alphanumeric = re.sub(r'[^A-Za-z0-9]', '0', i)
            i_alphanumeric = i_alphanumeric  # To fulfill flake8 conventions
            if i.startswith("_"):
                continue
            elif i in blacklist:
                continue
            else:
                try:
                    descriptor_list.append(i)
                    exec("desc_values.append(Descript.%(i)s(mol))" % vars())
                except:
                    pass
                if len(desc_values) < len(descriptor_list):
                    descriptor_list.remove(i)

        if k == 0:
            descriptors_df = pd.DataFrame(desc_values).T
        if k > 0:
            descriptors_df = pd.concat([descriptors_df, pd.DataFrame(desc_values).T])

    descriptors_df.columns = descriptor_list
    index_new = np.arange(0, len(descriptors_df))
    descriptors_df.index = index_new

    return descriptors_df



os.makedirs(os.path.join("data", "featurized"), exist_ok=True)

data = pd.read_csv(input)
smiles = data['smiles'][:max_number]
df_featurize = get_descriptors(smiles)
df_featurize.to_csv(f"{output}/featurized_data.csv")
