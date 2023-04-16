from tqdm.contrib.concurrent import process_map
from functools import partial
from itertools import product
from rdkit import Chem
import os
from itertools import chain
from xdalgorithm.toolbox.xreact.utils import countTags, generatorProduct, \
    load_reactant_dataset, load_synthesis_dataset
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem.PropertyMol import PropertyMol
from tqdm import tqdm

def get_rt_prd(rt, reaction_smarts):
    # print('reactions---{0}###'.format(rt))
    reaction = AllChem.ReactionFromSmarts(reaction_smarts)
    # print("rt type is {0},reaction type is {1},rt={2}".format(type(rt),
    rt_mols = []
    for m in rt:
        if isinstance(m, str):
            rt_mols.append(Chem.MolFromSmiles(m))
        else:
            tag_indices = [int(i) for i in m.GetProp('indices').split(',')]
            for tag_index in tag_indices:
                atom = m.GetAtomWithIdx(tag_index)
                atom.SetProp("tag", "%s" % atom.GetIdx())
            rt_mols.append(m)

    # make sure the reactive group on the core is doing the reaction, not the one from the building block
    n_old_tag = np.sum([countTags(m, 'tag') for m in rt_mols])
    products = reaction.RunReactants(rt_mols)
    product_list = []
    for prd in chain.from_iterable(products):
        n_current_tag = countTags(prd, 'tag')
        if n_old_tag != n_current_tag:
            Chem.SanitizeMol(prd)
            prd = PropertyMol(prd)
            new_indices = [str(atom.GetIdx()) for atom in prd.GetAtoms() if atom.HasProp('tag')]
            new_indices_string = ','.join(new_indices)
            prd.SetProp('indices', new_indices_string)
            product_list.append(prd)
    return product_list


class Rebuild:
    def __init__(self, route, core, output_path, bb_dict, db_path, n_cpu):
        self.route = route
        self.core = Chem.MolFromSmiles(core)
        self.output_path = output_path
        self.bb_dict = bb_dict
        self.reactants = load_reactant_dataset(db_path)
        self.synthesis_reactions = load_synthesis_dataset(db_path)
        self.n_cpu = n_cpu

    def rebuildByRoute(self):
        new_mols = self._runRoute(self.route)
        # return generated_smiles_set
        print("start to drop duplicate compounds and write to file.")
        smiles_set = set()
        with open(self.output_path, 'a') as f:
            for i in tqdm(new_mols):
                smi = Chem.MolToSmiles(i)
                if smi is None:
                    continue
                if smi not in smiles_set:
                    f.write("{0}\n".format(smi))
                    smiles_set.add(smi)

    def _runRoute(self, route):
        reaction = self.synthesis_reactions[route[0]]
        reactants = []

        for i, r in enumerate(route[2:]):
            if type(r) is list:
                reactants.append(self._runRoute(r))
            else:
                reactants.append(self._bbGenerator(r, route[0], i))

        reaction_smarts = AllChem.ReactionToSmarts(reaction)
        parallel_fun = partial(get_rt_prd, reaction_smarts=reaction_smarts)
        possible_length = 1
        for i in reactants:
            possible_length = possible_length * len(i)
        print("{0} cpu are used in multiprocessing.".format(self.n_cpu))
        jobs = process_map(parallel_fun,
                           product(*reactants),
                           max_workers=self.n_cpu,
                           total=possible_length)
        print("multiprocessing completed.")
        return list(chain(*jobs))

    def _bbGenerator(self, bb_def, rname, i):
        _mol = Chem.MolFromSmiles(bb_def)
        if _mol.HasSubstructMatch(self.core):
            mol = PropertyMol(_mol)
            indices_string = ",".join([str(i) for i in range(mol.GetNumAtoms())])
            mol.SetProp('indices', indices_string)  # save the atom indices to add tag properties
            return [mol]
        else:
            bb_file = self.bb_dict['%s/%s.smi' % (self.reactants[rname][i], rname)]
            if not os.path.exists(bb_file):
                print('Warning: ', bb_file, " not found")
                return []
            else:
                return [line.split()[0] for line in open(bb_file)]
