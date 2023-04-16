from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from functools import partial
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm as tqdm_fun


def label_query_mol(query_smi):
    mol = Chem.MolFromSmiles(query_smi)
    dummy_atom_indices = []
    query_atom_indices = []
    if mol is None:
        raise ValueError("Invalid query_smiles")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_atom_indices.append(atom.GetIdx())
            neighbor_atoms = [neighbor_atom for neighbor_atom in atom.GetNeighbors()]
            assert len(neighbor_atoms) == 1
            query_atom_indices.append(neighbor_atoms[0].GetIdx())
    mol.SetProp("dummy_atom_indices", ",".join([str(i) for i in dummy_atom_indices]))
    mol.SetProp("query_atom_indices", ",".join([str(i) for i in query_atom_indices]))
    return mol


def label_atom(mol):
    dummy_atom_indices = [int(i) for i in mol.GetProp("dummy_atom_indices").split(',')]
    query_atom_indices = [int(i) for i in mol.GetProp("query_atom_indices").split(',')]
    for atom_idx in dummy_atom_indices:
        mol.GetAtomWithIdx(atom_idx).SetProp("dummy", "1")
    for atom_idx in query_atom_indices:
        mol.GetAtomWithIdx(atom_idx).SetProp("query", "1")
    return mol


def label_fragment(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetProp("dummy", "1")
            neighbor_atoms = [neighbor_atom for neighbor_atom in atom.GetNeighbors()]
            assert len(neighbor_atoms) == 1
            neighbor_atoms[0].SetProp("frag", "1")
    return mol


def link_fragment(frag_smi, query_mol):
    query_mol = label_atom(query_mol)
    frag_mol = label_fragment(frag_smi)

    # find labeled atom indices in new molecule `combo`
    combo = Chem.CombineMols(query_mol, frag_mol)
    frag_indices = []
    query_indices = []
    for a in combo.GetAtoms():
        if a.HasProp("frag"):
            frag_indices.append(a.GetIdx())
        if a.HasProp("query"):
            query_indices.append(a.GetIdx())
    # add a single bond to link two atoms
    edcomb = Chem.EditableMol(combo)
    for frag_idx, query_idx in zip(frag_indices, query_indices):
        edcomb.AddBond(frag_idx, query_idx, order=Chem.rdchem.BondType.SINGLE)
    # remove boundary dummy atoms
    dummy_atom_to_remove = []
    for atom in combo.GetAtoms():
        if atom.HasProp("dummy"):
            dummy_atom_to_remove.append(atom.GetIdx())
    dummy_atom_to_remove.sort(reverse=True)
    for atom_idx in dummy_atom_to_remove:
        edcomb.RemoveAtom(atom_idx)
    linked_mol = edcomb.GetMol()
    try:
        Chem.SanitizeMol(linked_mol)
        return (True, Chem.MolToSmiles(linked_mol))
    except:
        return (False, None)


def load_fragment_list(fragment_path):
    with open(fragment_path) as reader:
        fragment_smiles_list = [line.strip() for line in reader]
    return fragment_smiles_list


class FragmentConnection:
    def __init__(self, bb_from, query_smiles, output_path, parallel=False, n_cpu=8):
        self.bb_from = bb_from
        self.output_path = output_path
        self.query_mol = label_query_mol(query_smiles)
        Chem.SanitizeMol(self.query_mol)
        self.query_mol = PropertyMol(self.query_mol)
        self.n_cpu = n_cpu
        if parallel:
            self.generate_smiles = self.generate_smiles_parallel

    def run(self):
        generated_smiles_list = self.generate_smiles()
        print("generate new compounds successfully")
        smiles_set = set(generated_smiles_list)
        print("start to save in {0}".format(self.output_path))
        with open(self.output_path, 'a') as f:
            for i in tqdm_fun(smiles_set):
                f.write(i + "\n")
        print("save successfully")

    def generate_smiles_parallel(self):
        fragments = load_fragment_list(self.bb_from)
        partial_fun = partial(link_fragment, query_mol=self.query_mol)
        jobs = process_map(partial_fun, fragments,
                           max_workers=self.n_cpu,
                           chunksize=10000
                           )
        generated_smiles_list = [job[1] for job in jobs if job[0]]
        return generated_smiles_list

    def generate_smiles(self):
        fragments = load_fragment_list(self.bb_from)
        generated_smiles_list = []
        for each_fragment in tqdm_fun(fragments):
            result, new_smiles = link_fragment(each_fragment, self.query_mol)
            if result:
                generated_smiles_list.append(new_smiles)
        return generated_smiles_list
