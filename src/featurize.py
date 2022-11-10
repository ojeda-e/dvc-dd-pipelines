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
    for k, smi in enumerate(smiles):  
        mol = Chem.MolFromSmiles(smi)
        mol = Std().standardize(mol)
        descriptor_list = []
        desc_values = []
        blacklist = ['Ipc']
        for element in dir(Descript):  
            i_alphanumeric = re.sub(r'[^A-Za-z0-9]', '0', element)
            i_alphanumeric = i_alphanumeric 
            if element.startswith("_"):
                continue
            elif element in blacklist:
                continue
            else:
                try:
                    descriptor_list.append(element)
                    exec("desc_values.append(Descript.%(element)s(mol))" % vars())
                except:
                    pass
                if len(desc_values) < len(descriptor_list):
                    descriptor_list.remove(element)

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
smiles = data.iloc[0:max_number]['smiles']
print(smiles.shape)
df_featurize = get_descriptors(smiles)
df_featurize.to_csv(f"{output}", index=False)
print(df_featurize.shape)
print("Featurizing process completed!")
print('The shape of the featurized dataframe is: ', df_featurize.shape)