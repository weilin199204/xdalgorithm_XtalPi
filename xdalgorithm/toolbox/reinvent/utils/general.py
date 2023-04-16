import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def get_indices_of_unique_smiles(smiles: [str]) -> np.array:
    """Returns an np.array of indices corresponding to the first entries in a list of smiles strings"""
    _, idxs = np.unique(smiles, return_index=True)
    sorted_indices = np.sort(idxs)
    return sorted_indices


def set_default_device_cuda():
    """Sets the default device (cpu or cuda) used for all tensors."""
    if torch.cuda.is_available() == False:
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        return False
    else:  # device_name == "cuda":
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        torch.set_default_tensor_type(tensor)
        return True


def fraction_valid_smiles(smiles):
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    fraction = 100 * i / len(smiles)
    return fraction


def scaffold_num_in_smiles(smiles):
    scaffold_set = set()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold_smi = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            scaffold_set.add(scaffold_smi)
        except:
            print("scaffold_error:" + str(smi))
            scaffold_set.add('')
    return len(scaffold_set)


def unique_smiles_num(smiles):
    smiles_set = set()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            formal_smi = Chem.MolToSmiles(mol)
        except:
            continue
        smiles_set.add(formal_smi)
    return len(smiles_set)

def scaffold_smarts_matched_num(smiles, smarts):
    smarts_mol = Chem.MolFromSmarts(smarts)
    generated_smiles_num = len(smiles)
    matched_unique_smiles_set = set()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        # try:
        if mol.HasSubstructMatch(smarts_mol):
            inchs = Chem.MolToInchi(mol)
            matched_unique_smiles_set.add(inchs)
        # except:
        #    continue
    unique_matched_num = len(matched_unique_smiles_set)
    matched_ratio = unique_matched_num * 1.0 / generated_smiles_num
    return matched_ratio, unique_matched_num
