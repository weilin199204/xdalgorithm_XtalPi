#coding:utf-8
from rdkit import Chem
import os
import numpy as np
import re
import pickle
import os
import random
import pandas as pd
from tqdm import tqdm
import rdkit.Chem as rkc
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from rdkit.Chem import SaltRemover
from rdkit.Chem import rdmolops
from functools import partial
import argparse
import rdkit.RDLogger as rkl
logger = rkl.logger()
logger.setLevel(rkl.ERROR)

import rdkit.rdBase as rkrb
rkrb.DisableLog('rdApp.error')

REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]
REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
legal_token_lst = ['$','^','#','%10','(',')','-','1','2','3','4','5','6','7','8','9','=','Br','C','Cl','F',
                   'N','O','S','[N+]','[N-]','[O-]','[S+]','[n+]','[nH]','c','n','o','s']


def split_by(data, regexps):
    if not regexps:
        return list(data)
    regexp = REGEXPS[regexps[0]]
    splitted = regexp.split(data)
    tokens = []
    for i, split in enumerate(splitted):
        if i % 2 == 0:
            tokens += split_by(split, regexps[1:])
        else:
            tokens.append(split)
    return tokens

def token_check(smi):
    tokens = split_by(smi,REGEXP_ORDER)
    for token in tokens:
        if token not in legal_token_lst:
            return False,"illegal token found: "+token
    return True,None

def _initialiseNeutralisationReactions():
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(rkc.MolFromSmarts(x), rkc.MolFromSmiles(y, False)) for x, y in patts]


def _get_largest_fragment(mol):
    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    maxmol = None
    for mol in frags:
        if mol is None:
            continue
        if maxmol is None:
            maxmol = mol
        if maxmol.GetNumHeavyAtoms() < mol.GetNumHeavyAtoms():
            maxmol = mol
    return maxmol


def _remove_salts(mol):
    return SaltRemover.SaltRemover().StripMol(mol, dontRemoveEverything=True)


def _neutralise_charges(mol, reactions=None):
    if reactions is None:
        reactions = _initialiseNeutralisationReactions()
    replaced = False
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return mol, True
    else:
        return mol, False

def _valid_size(mol,element_list):
    valid_elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
    return valid_elements

def _standardize_smiles(smiles):
    element_list = [6, 7, 8, 9, 16, 17, 35]  # C,N,O,F,S,Cl,Br
    mol = rkc.MolFromSmiles(smiles)
    if mol:
        mol = _get_largest_fragment(mol)
    if mol:
        mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    if mol:
        mol = _remove_salts(mol)
    if mol:
        mol, _ = _neutralise_charges(mol)
    if mol:
        rdmolops.Cleanup(mol)
        rdmolops.SanitizeMol(mol)
        mol = rdmolops.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    if not mol or not _valid_size(mol,element_list):
        return None
    if mol:
        smi = rkc.MolToSmiles(mol, isomericSmiles=False)
        return smi
    return None

def _smiles_check(input_smis):
    std_smis = [];error_smis = []
    for smi in tqdm(input_smis):
        raw_smi = smi
        std_smi = _standardize_smiles(smi)
        if std_smi is None:
            error_smis.append(raw_smi+","+"std is None")
            continue
        check_reuslt,check_msg = token_check(std_smi)
        if check_reuslt:
            std_smis.append(std_smi)
        else:
            error_smis.append(raw_smi+","+check_msg)
    return std_smis,error_smis

def preprocessing(input_file_name,output_file_name,error_file_name):
    with open(input_file_name) as file_handler:
        input_smis = [line.strip() for line in file_handler.readlines()]

    final_smis,error_smis = _smiles_check(input_smis)

    with open(output_file_name,"a") as output_file_handler:
        correct_content = '\n'.join(final_smis)
        output_file_handler.write(correct_content)

    with open(error_file_name,"a") as error_file_handler:
        error_content = '\n'.join(error_smis)
        error_file_handler.write(error_content)

