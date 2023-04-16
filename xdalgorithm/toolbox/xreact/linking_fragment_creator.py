from glob import glob
import os
import re
from dataclasses import dataclass
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.AllChem import ReactionFromSmarts
from tqdm.contrib.concurrent import process_map
from xdalgorithm.toolbox.reinvent.utils import disable_rdkit_logging

disable_rdkit_logging()


@dataclass
class Config:
    reference_db: str
    reaction_dir: str = '/data/corp/lijun.yang/miniconda3/envs/xdalgo-env/lib/python3.7/site-packages/xdalgorithm/data/xreact_reactions'
    db_path: str = 'sorted_bbs_db'


def load_smiles_from_file(smi_path):
    """
    read smiles from a smi file
    :param smi_path:
    :return:
    """
    smiles_list = []
    with open(smi_path) as reader:
        for line in reader.readlines():
            smiles_list.append(line.strip())
    return smiles_list


def save_smiles_in_file(smiles_list, file_name):
    content = '\n'.join(smiles_list) + '\n'
    with open(file_name, 'w') as writer:
        writer.write(content)


def load_smiles_in_db(db_path):
    """
    read all smiles from a source xreact fragment db
    :param db_path:
    :return:
    """
    smiles_file_names = glob('{0}/*/*.smi'.format(db_path))
    smiles_dict = dict()
    for each_name in smiles_file_names:
        substance_type, reaction_type = each_name.split('/')[-2:]
        smiles_dict[(substance_type, reaction_type.replace('.smi', ''))] = load_smiles_from_file(each_name)
    return smiles_dict


def load_aid_substance(reference_db_path):
    """
    each substance type and reaction type map to an aid smiles for reaction
    :param reference_db_path:
    :return:
    """
    smiles_file_names = glob('{0}/*/*.smi'.format(reference_db_path))
    aid_dict = dict()
    for each_name in tqdm(smiles_file_names):
        with open(each_name) as reader:
            example = reader.readline().strip()
        substance_type, reaction_type = each_name.split('/')[-2:]
        aid_dict[(substance_type, reaction_type.replace('.smi', ''))] = example
    return aid_dict


def load_rxn_substance_map(reactant_path):
    """
    get a dict ,key is reaction type, value is a list of reactant types
    :param reactant_path:
    :return:
    """
    rxn_template_map = dict()
    with open(reactant_path) as reader:
        for line in reader.readlines():
            splited_list = line.strip().split()
            if len(splited_list) == 1:
                continue
            rxn_template_map[splited_list[0]] = splited_list[1:]
    return rxn_template_map


def load_rxn_template(template_path):
    """
    get a dict, key is name of reaction,value is SMARTS template
    :param template_path:
    :return:
    """
    template_dict = dict()
    with open(template_path) as reader:
        for line in reader.readlines():
            rxn_name, template = line.strip().split()
            template_dict[rxn_name] = template
    return template_dict


def get_aid_mols(aid_substance_dict):
    """
    A reactant type map to an aid mol
    :param aid_substance_dict:
    :return:
    """
    aid_mol_dict = dict()
    for each_key in aid_substance_dict:
        if len(aid_substance_dict[each_key]) > 0:
            aid_mol_dict[each_key] = Chem.MolFromSmiles(aid_substance_dict[each_key])
        else:
            aid_mol_dict[each_key] = None
    return aid_mol_dict


def clean_output_mols(mol):
    cut_bond_indices = []
    for bond in mol.GetBonds():
        begin_atom_num = bond.GetBeginAtom().GetAtomMapNum()
        end_atom_num = bond.GetEndAtom().GetAtomMapNum()
        if (begin_atom_num == 1000 and end_atom_num == 0) or (begin_atom_num == 0 and end_atom_num == 1000):
            cut_bond_indices.append(bond.GetIdx())
    if len(cut_bond_indices) == 0:
        return []
    fragments = Chem.FragmentOnBonds(mol, cut_bond_indices)
    remained_fragments = [smi for smi in Chem.MolToSmiles(fragments).split('.') if not re.search(':1000', smi)]
    remained_fragments = [re.sub('\[[0-9]+\*\]', '[*]', smi) for smi in remained_fragments if smi.count('*') == 1]
    return remained_fragments


def clean_hydrogens(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return (smi, True)
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        for atom in mol.GetAtoms():
            if atom.IsInRing() and atom.GetIsAromatic():
                if atom.GetAtomicNum() in [6, 7]:
                    if len(atom.GetNeighbors()) == 3:
                        atom.SetNumExplicitHs(0)
        cleaned_smi = Chem.MolToSmiles(mol)
        if Chem.MolFromSmiles(cleaned_smi) is not None:
            return (cleaned_smi, True)
        else:
            return (smi, False)
    except:
        return (smi, False)


def unit(mol):
    fragments = clean_output_mols(mol)
    checked_results = [clean_hydrogens(frag_smi) for frag_smi in fragments]
    good_smiles_list = [i for i, j in checked_results if j]
    bad_smiles_list = [i for i, j in checked_results if not j]
    return (good_smiles_list, bad_smiles_list)


class LinkingFragmentCreator:
    def __init__(self, reference_db, reaction_dir, db_path, n_cpu, result_dir):
        self.reference_db = reference_db
        self.reaction_dir = reaction_dir
        self.db_path = db_path
        self.n_cpu = n_cpu
        self.result_dir = result_dir
        assert not os.path.exists(self.result_dir), "Directory {0} is existed!".format(self.result_dir)
        os.mkdir(self.result_dir)

    def create_fragments(self):
        reactants_path = os.path.join(self.reaction_dir, 'reactants.txt')
        rxn_substance_type_map = load_rxn_substance_map(reactants_path)
        template_dict = load_rxn_template(os.path.join(self.reaction_dir, 'synthesis.txt'))
        aid_substance_dict = load_aid_substance(self.reference_db)
        aid_mols_dict = get_aid_mols(aid_substance_dict)
        smiles_dict = load_smiles_in_db(self.db_path)
        smiles_keys = list(smiles_dict.keys())
        generated_product_mols = []
        print('step1/3 enumerate all reaction to ger products.')
        for i, test_key in enumerate(smiles_keys):
            print('{0}/{1} start...'.format(i, len(smiles_keys)), end='\r')
            smiles_list = smiles_dict[test_key]
            main_reactant_type, rxn_name = test_key
            rxn_template = template_dict[rxn_name]
            main_index = rxn_substance_type_map[rxn_name].index(main_reactant_type)
            rxn_template_instance = ReactionFromSmarts(rxn_template)
            aid_list = []
            is_skipped = False
            for i in range(len(rxn_substance_type_map[rxn_name])):
                if i == main_index:
                    aid_list.append(None)
                else:
                    try:
                        aid_list.append(aid_mols_dict[rxn_substance_type_map[rxn_name][i], rxn_name])
                    except:
                        is_skipped = True
            if is_skipped:
                continue
            for each_mol in aid_list:
                if each_mol is not None:
                    for atom in each_mol.GetAtoms():
                        atom.SetAtomMapNum(1000)

            for each_smi in smiles_list:
                mol = Chem.MolFromSmiles(each_smi)
                aid_list[main_index] = mol
                if any([aid is None for aid in aid_list]):
                    break
                results = rxn_template_instance.RunReactants(aid_list)
                candidates = [j for i in results for j in i]
                generated_product_mols.extend(candidates)
        print('step 2/3 clean and check valid smiles')
        chunk_size = max(len(generated_product_mols) // (self.n_cpu * 3), 1)
        clean_result = process_map(unit, generated_product_mols,
                                   chunksize=chunk_size, max_workers=self.n_cpu)
        print('step 3/3 merge in file')
        good_smiles_list = []
        bad_smiles_list = []
        print()
        for good_sub_list,bad_sub_list in clean_result:
            good_smiles_list.extend(good_sub_list)
            bad_smiles_list.extend(bad_sub_list)
        good_smiles_list = list(set(good_smiles_list))
        bad_smiles_list = list(set(bad_smiles_list))

        # good_smiles_list = [i for i, j in clean_result if j]
        # bad_smiles_list = [i for i, j in clean_result if not j]
        save_smiles_in_file(good_smiles_list, os.path.join(self.result_dir, 'good_smiles.smi'))
        save_smiles_in_file(bad_smiles_list, os.path.join(self.result_dir, 'skipped_bad_smiles.smi'))
