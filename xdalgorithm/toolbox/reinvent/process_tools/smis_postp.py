from rdkit import Chem
from rdkit import RDLogger
from xdalgorithm.toolbox.norm_mol import Normalizer
from tqdm.contrib.concurrent import process_map

RDLogger.DisableLog('rdApp.*')


def check_lone_pair_electrons(mol):
    for atom in mol.GetAtoms():
        # if atom.GetAtomicNum() == 6:
        #    if atom.GetTotalValence() < 4:
        atom.SetNumRadicalElectrons(0)
    cano_smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(cano_smiles)
    return mol


def clean_one_smiles(smi):
    # is valid
    norm = Normalizer()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = norm(mol)
        mol = check_lone_pair_electrons(mol)
    except:
        # print("Error smiles string {0} is skipped.".format(smi))
        return None
    try:
        canonical_smiles = Chem.MolToSmiles(mol)
    except:
        return None
    return canonical_smiles


def postprocessing(input_files, output_file):
    # load smiles from all files
    smiles_set_to_clean = set()
    for file_name in input_files:
        with open(file_name) as file1:
            smis = set([i.strip() for i in file1.readlines()])
        smiles_set_to_clean = smiles_set_to_clean | smis
    print("load unique {0} generated smiles from {1} files.".format(len(smiles_set_to_clean), len(input_files)))
    print("start to clean...")
    output_results = process_map(clean_one_smiles, smiles_set_to_clean, chunksize=1000)
    smiles_unique_list = list(set([i for i in output_results if i]))
    with open(output_file, 'w') as file_writer:
        content = '\n'.join(smiles_unique_list)
        file_writer.write(content)
    print("write success")
