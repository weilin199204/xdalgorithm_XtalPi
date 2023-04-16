import numpy as np
from itertools import count
import os
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from xdalgorithm.utils import get_data_file_path


class GenCacher:
    def __init__(self, generator):
        self._g = generator
        self._cache = []

    def __getitem__(self, idx):
        while len(self._cache) <= idx:
            self._cache.append(next(self._g))
        return self._cache[idx]


def summations(sumTo, n=2):
    if n == 1:
        yield (sumTo,)
    else:
        for head in range(sumTo + 1):
            for tail in summations(sumTo - head, n - 1):
                yield (head,) + tail


def labelMol(m):
    for atom in m.GetAtoms():
        atom.SetProp("tag", "%s" % atom.GetIdx())
    return m


def generatorProduct(*gens):
    n = len(gens)
    gens = list(map(GenCacher, gens))

    try:
        for dist in count(0):
            n_idxs = 0
            n_fails = 0
            for idxs in summations(dist, n):
                n_idxs += 1
                try:
                    yield tuple([gen[idx] for gen, idx in zip(gens, idxs)])
                except StopIteration:
                    n_fails += 1
                    continue
            if n_fails == n_idxs: return

    except GeneratorExit:
        return


def countTags(m, tag_name):
    return np.sum([atom.HasProp(tag_name) for atom in m.GetAtoms()])


def findTags(m, tag_name):
    return [atom.GetProp(tag_name) for atom in m.GetAtoms() if atom.HasProp(tag_name)]


def load_reactant_dataset(db_path):
    reactants = {}
    with open(os.path.join(db_path, 'reactants.txt')) as f:
        for rd in f:
            items = rd.split()
            if len(items) > 1:
                reactants[items[0]] = items[1:]
            else:
                reactants[items[0]] = []
    return reactants


def load_synthesis_dataset(db_path):
    synthesis_reactions = {}
    with open(os.path.join(db_path, 'synthesis.txt')) as f:
        for rd in f:
            name, smarts = rd.split()
            reaction = AllChem.ReactionFromSmarts(smarts)
            synthesis_reactions[name] = reaction
    return synthesis_reactions


def clustering_by_pairwise_similarity(ms, cutoff):
    """
        cluster mols by sorted similarity coverage
        mols a, b cover each other if sim(a, b) > cutoff
    """
    if len(ms) == 1:
        return [[0]]

    similarity = pairwise_similarity(ms)
    covered = set()
    mask = np.ones(len(ms)).astype(bool)
    chosen = []

    while len(covered) < len(ms):
        smat = similarity[mask, :][:, mask]
        n_covered = np.sum(smat > cutoff, axis=1)

        best_i = np.argmax(n_covered)
        j = 0
        while j <= best_i:
            if mask[j] == False:
                best_i += 1
            j += 1

        chosen.append([best_i])
        mask[np.where(similarity[best_i, :] > cutoff)[0]] = False

        for i in np.where(similarity[best_i, :] > cutoff)[0]:
            if not i in covered:
                covered.add(i)
                if i != chosen[-1][0]:
                    chosen[-1].append(i)

    return chosen


def pairwise_similarity(ms):
    similarity = []
    for m in ms:
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=128)
        fps = [AllChem.GetMorganFingerprintAsBitVect(om, 2, nBits=128) for om in ms]

        similarity.append(BulkTanimotoSimilarity(fp, fps))

    return np.array(similarity)


def get_xreact_reaction_path():
    reactant_path = get_data_file_path("data", "xreact_reactions/reactants.txt")
    return os.path.dirname(reactant_path)


# get the path of building blocks dir from ENV VAR
def get_building_blocks_path(xreact_db_path=os.environ.get('XREACT_HOME', "/home/jovyan/team-share/aidd/Building_blocks_sorted_by_reactions")):
    idc_building_blocks_path = '/data/aidd-server/Building_blocks_sorted_by_reactions'
    if os.path.exists(xreact_db_path):
        return xreact_db_path
    elif os.path.exists(idc_building_blocks_path):
        return idc_building_blocks_path
    else:
        raise Exception("input file {} does not exist".format(xreact_db_path))
