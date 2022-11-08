# scientific
import pandas as pd
import numpy as np
import re
# ML
from rdkit import Chem
from rdkit.Chem import Descriptors as D
from rdkit.Chem import AllChem
from rdkit import DataStructs
from molvs import Standardizer as Std


def get_descriptors(smiles: str) -> pd.DataFrame:
    """
    This function generates molecule objects from SMILES and calculates
    descriptors from standardized molecules.

    Parameters:
    ------------
    smiles:  str
        SMILES to calculate descriptor from.

    Returns:
    ----------
    descriptor_df: pd.DataFrame
        Pandas DataFrame with descriptors from standarized molecules.
    """
    for k, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        mol = Std().standardize(mol)
        desc_list = []
        desc_values = []
        blacklist = ['Ipc']
        for i in dir(D):
            i_alphanumeric = re.sub(r'[^A-Za-z0-9]', '0', i)
            i_alphanumeric = i_alphanumeric
            if i.startswith("_"):
                continue
            elif i in blacklist:
                continue

            else:
                try:
                    desc_list.append(i)
                    exec("desc_values.append(D.%(i)s(mol))" % vars())

                except:
                    pass

                if len(desc_values) < len(desc_list):
                    desc_list.remove(i)
        if k == 0:
            descriptor_df = pd.DataFrame(desc_values).T

        if k > 0:
            descriptor_df = pd.concat([descriptor_df, pd.DataFrame(desc_values).T])

    descriptor_df.columns = desc_list
    index_new = np.arange(0, len(descriptor_df))
    descriptor_df.index=index_new

    return descriptor_df


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
        if (np.abs(corr_mx.loc[colx, newColumns]) < filter).all():
            newColumns.append(colx)
        else:
            print(f"Removing column {colx}")
            df=df.drop(colx, axis=1)
    return df

def get_smiles(subset, dataset):
    smiles_list=[]
    for i in range(0, len(subset)):
        smiles_list.append(dataset['smiles'][subset.index[i]])
    df_smiles=pd.DataFrame(smiles_list)[0]
    return df_smiles

def get_morganfps(smiles, radius: int, nBits):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    mfps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits) for mol in mols]
    return mfps

def get_tanimoto(mfps1, mfps2):
    tanimoto_list = []
    tanimoto_mx = np.zeros([len(mfps1), len(mfps2)])
    for i, j in [[i, j] for i in range(len(mfps1)) for j in range(len(mfps2))]:
        tanimoto_mx[i][j] = DataStructs.TanimotoSimilarity(mfps1[i], mfps2[j])
        tanimoto_list.append(tanimoto_mx[i][j])    
    return tanimoto_mx, tanimoto_list

def normalize(x):
    return (x - x.min(0)) / x.ptp(0)