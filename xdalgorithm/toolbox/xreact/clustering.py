from rdkit import Chem
from rdkit.Chem import AllChem
from xdalgorithm.toolbox.xreact.utils import load_reactant_dataset, clustering_by_pairwise_similarity
from collections import defaultdict
import os
from functools import partial
from tqdm.contrib.concurrent import process_map


def query_synthesis_reactions(db_path, queried_names):
    synthesis_reactions = {}
    with open(os.path.join(db_path, 'synthesis.txt')) as f:
        for rd in f:
            name, smarts = rd.split()
            if name in queried_names:
                synthesis_reactions[name] = AllChem.ReactionFromSmarts(smarts)
    return synthesis_reactions


def _get_scaffold(mol):
    """
        turn elements to C except *
        iteratively delete outer CH3
    """
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != '*':
            atom.SetAtomicNum(6)

    n = mol.GetNumAtoms()
    query = Chem.MolFromSmarts('[CH3]')

    last_n = 0
    while n != last_n:
        last_n = n
        mol = Chem.DeleteSubstructs(mol, query)
        n = mol.GetNumAtoms()

    return Chem.MolToSmiles(mol)


def unit_fun(line, reaction, reactant_type_i):
    index, line = line
    smiles = line.split()[0]
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        a.SetProp('tag', '1')
    partial_prod = reaction.RunReactant(mol, reactant_type_i)[0][0]
    for a in partial_prod.GetAtoms():
        if not a.HasProp('tag'):
            a.SetAtomicNum(0)
    p_sm = Chem.MolToSmiles(partial_prod)
    scaffold = _get_scaffold(partial_prod)
    return (scaffold, smiles, p_sm, index, partial_prod.GetNumAtoms())


class Clustering:
    def __init__(self, reactant_def, bb_from, rep, cutoff, db_path, n_cpu, logging_path):
        self.route_name = reactant_def.split('/')[1].split('.')[0]
        self.bb_def = reactant_def.split('/')[0]
        self.bb_from = bb_from
        self.rep = rep
        self.cutoff = cutoff
        self.reactants = load_reactant_dataset(db_path)
        self.synthesis_reactions = query_synthesis_reactions(db_path, self.route_name)
        self.reaction = self.synthesis_reactions[self.route_name]
        # print(self.reaction)
        self.n_cpu = n_cpu
        self.logging_path = logging_path

    def run(self):
        if (not os.path.exists(self.logging_path)) or os.path.isdir(self.logging_path):
            os.mkdir(self.logging_path)
        reactant_type_i = self.reactants[self.route_name].index(self.bb_def)
        # print("reactant_type_i:{0}".format(reactant_type_i))
        scaffolds = defaultdict(list)
        unique_partial_prod = set()

        partial_fun = partial(unit_fun,
                              reaction=self.reaction,
                              reactant_type_i=reactant_type_i)
        with open(self.bb_from) as f:
            lines = [(i, line) for i, line in enumerate(f)]

        print("{0} cpu are used in multiprocessing.".format(self.n_cpu))
        jobs = process_map(partial_fun, lines, max_workers=self.n_cpu, chunksize=1000)
        # jobs = [partial_fun(index=i,line=line) for i, line in enumerate(f)]
        for result in jobs:
            if result is None:
                continue
            if result[2] not in unique_partial_prod:
                unique_partial_prod.add(result[2])
                scaffolds[result[0]].append((result[1:]))
        print("Find representative compounds...")
        output_rep_txt = ''
        output_index = ''
        if self.rep == "least_num_atoms":
            for ci, key in enumerate(scaffolds):
                output_index += 'cluster_%d\n' % ci
                output_index += '\n'.join([str(x[2]) for x in scaffolds[key]]) + '\n'
                min_result = min(scaffolds[key], key=lambda x: x[3])
                output_rep_txt += min_result[0] + '\n'
        elif self.rep == 'fp_clustering':
            for ci, key in enumerate(scaffolds):
                _local_clusters = clustering_by_pairwise_similarity([Chem.MolFromSmiles(x[1]) for x in scaffolds[key]],
                                                                    self.cutoff)
                for cj, _c in enumerate(_local_clusters):
                    output_index += 'cluster_%d_%d\n' % (ci, cj)
                    output_index += '\n'.join([str(scaffolds[key][_i][2]) for _i in _c]) + '\n'
                    output_rep_txt += scaffolds[key][_c[0]][0] + '\n'
        else:
            raise ValueError("Representative type -- %s -- is not supported yet!" % self.rep)

        rep_path = os.path.join(self.logging_path, "rep_mols.smi")
        with open(rep_path, 'w') as f:
            f.write(output_rep_txt)

        index_path = os.path.join(self.logging_path, "index.dat")
        with open(index_path, 'w') as f:
            f.write(output_index)
