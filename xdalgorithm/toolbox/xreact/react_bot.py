# coding: utf-8

from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import chain, product
import os
import numpy as np
from .utils import findTags
from xdalgorithm.toolbox.xreact.utils import get_xreact_reaction_path

# Core-based compound decomposition
# ======
#
# * try to break the bonds between the core and the rest part
# * route maintain the core
# * route deduplication
# * reactant specificity

def getMatchedPosition(reaction, mol):
    matched_position = []
    unmatched_position = []
    n_reactants = reaction.GetNumReactantTemplates()

    for i in range(n_reactants):
        template = reaction.GetReactantTemplate(i)
        if mol.HasSubstructMatch(template):
            matched_position.append(i)
        else:
            unmatched_position.append(i)
    return matched_position, unmatched_position

def labelTagAndCore(m, core):
    match = m.GetSubstructMatch(core)
    for i, idx in enumerate(match):
        m.GetAtomWithIdx(idx).SetProp("core", "%d" % i)

    for atom in m.GetAtoms():
        atom.SetProp("tag", "%s" % atom.GetIdx())

    return m

def compareType(type1, type2):
    """
    0: no overlap
    1: type1 >= type2
    -1: type1 < type2
    2: type1 != type2
    """
    if type1.split('_')[-1] == type2.split('_')[-1]:
        s1 = set([item for item in type1.split('_')[:-1]])
        s2 = set([item for item in type1.split('_')[:-1]])
        if s1.intersection(s2) == s1:
            return 1
        elif s2.intersection(s1) == s2:
            return -1
        elif len(s1.intersection(s2)) > 0:
            return 2
        else:
            return 0
    else:
        return 0


def compareTypeSet(ts1, ts2):
    if len(ts1) != len(ts2): return 0
    ts1 = sorted(ts1)
    ts2 = sorted(ts2)

    result = set([compareType(s1, s2) for s1, s2 in zip(ts1, ts2)])
    if len(result) == 1:
        if 1 in result:
            return 1
        elif -1 in result:
            return -1
        else:
            return 0
    else:
        return 0


def compareRoute(route1, route2):
    if set(route1.keys()) != set(route2.keys()): return 0
    result = set([compareTypeSet(route1[k], route2[k]) for k in route1.keys()])
    if len(result) == 1:
        if 1 in result:
            return 1
        elif -1 in result:
            return -1
        else:
            return 0
    else:
        return 0


class reactant_node:
    def __init__(self, compound_smiles, compound_type, depth):  # smiles,'',0
        self.smiles = compound_smiles
        self.type = compound_type
        self.child = []
        self.depth = depth

    def addChild(self, reaction_tuple):
        rn, rt = reaction_tuple
        new_list = []
        for s in rt:
            new_list.append(reaction_node((rn, *s), self.depth + 1))

        self.child += new_list
        return new_list

    def hasChild(self):
        return len(self.child) > 0

    def getTuple(self):
        if self.hasChild():
            return [child.getTuple() for child in self.child]
        else:
            return self.smiles

    def getList(self):
        if self.hasChild():
            return list(chain(*[child.getList(self.smiles) for child in self.child]))
        else:
            return [self.smiles]

    def getType(self):
        if self.hasChild():
            return list(chain(*[child.getTypeDict() for child in self.child]))
        else:
            return self.type

    def getWidgetsInfo(self):
        if self.hasChild():
            result = {}
            for i, child in enumerate(self.child):
                new_key = child.name + "_reaction" + str(i)
                result[new_key] = child.getWidgetsInfo()
            return result
        else:
            return {}


class reaction_node:
    def __init__(self, reaction_tuple, depth):
        self.name = reaction_tuple[0]
        self.core_reacted = reaction_tuple[1]
        self.core_pos = reaction_tuple[2]
        self.nodes = [reactant_node(sm, tp, depth) for sm, tp in reaction_tuple[3]]

    def getTuple(self):
        self.tuple = (self.name, [node.getTuple() for node in self.nodes])
        return self.tuple

    def getList(self,parent_reactant):
        self.list = [[self.name,parent_reactant] + list(p) for p in product(*[node.getList() for node in self.nodes])]
        return self.list

    def getTypeDict(self):
        _dict = {self.core_reacted: set([node.getType() for i, node in enumerate(self.nodes) if i != self.core_pos])}
        _dict_list = self.nodes[self.core_pos].getType()

        if type(_dict_list) is list:
            [item.update(_dict) for item in _dict_list]
        else:
            _dict_list = [_dict]

        return _dict_list

    def getWidgetsInfo(self):
        result = {}
        for i, node in enumerate(self.nodes):
            new_key = node.smiles + "_reactant" + str(i)
            result[new_key] = node.getWidgetsInfo()
        return result


class ReactBot:
    def __init__(self, start_smiles, core_smiles, db_path=None):
        """
             try to decompose the compound around the core
        """
        if db_path is None:
            db_path = get_xreact_reaction_path()

        self.start_smiles = start_smiles

        self.core = Chem.MolFromSmiles(core_smiles) if core_smiles else None

        self.root = reactant_node(start_smiles, '', 0)
        self.process_queue = [self.root]
        self._load_synthesis_dataset(db_path)
        self._load_reactant_dataset(db_path)
        self._load_retro_synthesis_dataset(db_path)

    def _load_reactant_dataset(self,db_path):
        self.reactants = {}
        with open(os.path.join(db_path, 'reactants.txt')) as f:
            for rd in f:
                items = rd.split()
                if len(items) > 1:
                    self.reactants[items[0]] = items[1:]
                else:
                    self.reactants[items[0]] = []

    def _load_retro_synthesis_dataset(self,db_path):
        self.retro_reactions = {}
        with open(os.path.join(db_path, 'retro_synthesis.txt')) as f:
            for rd in f:
                name, smarts = rd.split()
                reaction = AllChem.ReactionFromSmarts(smarts)
                if self.reactants[name]: self.retro_reactions[name] = reaction

    def _load_synthesis_dataset(self,db_path):
        self.synthesis_reactions = {}
        with open(os.path.join(db_path, 'synthesis.txt')) as f:
            for rd in f:
                name, smarts = rd.split()
                reaction = AllChem.ReactionFromSmarts(smarts)
                self.synthesis_reactions[name] = reaction

    def analysis(self, core_specific=True, core_single_reactive_center=True):
        while len(self.process_queue) > 0:
            active_node = self.process_queue.pop()

            reactant = active_node.smiles
            routes = self.decompose(reactant, core_specific, core_single_reactive_center)

            for rt in routes:
                new_nodes = active_node.addChild(rt)
                self.process_queue += [n for new_n in new_nodes for n in new_n.nodes]

        self.getRouteList()

    def decompose(self, compound_smiles, core_specific, core_single_reactive_center):
        mol = Chem.MolFromSmiles(compound_smiles)
        routes = []
        matches = set()
        if self.core is not None:
            matches = set(mol.GetSubstructMatch(self.core))

        for rn in self.retro_reactions:
            reaction = self.retro_reactions[rn]
            forward_reaction = self.synthesis_reactions[rn]
            if self.core is not None:
                # make sure that the reaction happens around the core
                [mol.GetAtomWithIdx(i).SetProp('_protected', '1') for i in range(mol.GetNumAtoms())]
                patt = reaction.GetReactantTemplate(0)
                reactive_matches = mol.GetSubstructMatches(patt)
                if not reactive_matches: continue

                for rm in reactive_matches:
                    if 0 < len(matches.intersection(set(rm))) < len(rm):
                        for atom_i in rm:
                            mol.GetAtomWithIdx(atom_i).ClearProp('_protected')
                        break

                mol = labelTagAndCore(mol, self.core)

            prods = reaction.RunReactants([mol])
            prod_set = set()
            prod_list = []

            for prd in prods:
                [Chem.SanitizeMol(p) for p in prd]
                ind = None

                if self.core is not None:
                    # must maintain the core
                    verify = [p.HasSubstructMatch(self.core) for p in prd]
                    if not True in verify: continue
                    ind = int(np.where(np.array(verify) == True)[0][0])
                    # print(rn, ind)

                    # core must be specific
                    if core_specific:
                        matched_p, _ = getMatchedPosition(forward_reaction, prd[ind])
                        reactant_types = self.reactants[rn]
                        n_core_types = 1
                        core_type = reactant_types[ind]

                        for i in range(len(reactant_types)):
                            if i != ind:
                                subtype = reactant_types[i]
                                # reactant types could be overlapping
                                if compareType(subtype, core_type) != 0:
                                    n_core_types += 1

                        if len(matched_p) != n_core_types: continue

                    # core must have only one reactive group
                    if core_single_reactive_center:
                        keep = True
                        old_atoms = set([atom.GetIdx() for atom in prd[ind].GetAtoms() if atom.HasProp('tag')])
                        patt = forward_reaction.GetReactantTemplate(ind)
                        matched_patts = prd[ind].GetSubstructMatches(patt)
                        for p in matched_patts:
                            # print(old_atoms, p)
                            if old_atoms.union(set(p)) == old_atoms:
                                keep = False
                                break

                        if not keep: continue

                    core_reacted = set(findTags(mol, 'core')).difference(set(findTags(prd[ind], 'core')))
                    # print(rn, core_reacted)
                else:
                    core_reacted = tuple()

                l = [Chem.MolToSmiles(m) for m in prd]
                t = self.reactants[rn]
                if not frozenset(l) in prod_set:
                    prod_set.add(frozenset(l))
                    prod_list.append((tuple(sorted(core_reacted)), ind, zip(l, t)))

            if prod_set: routes.append((rn, prod_list))

        return routes

    def getRouteTuple(self):
        return self.root.getTuple()

    def getRouteList(self):
        self.unique_routes = []
        l = self.root.getList()
        t = self.root.getType()
        _unique = []

        for i, item in enumerate(t):
            keep = True
            # print("Final: ", item)
            for j, _uitem in enumerate(_unique):
                c = compareRoute(item, _uitem)
                # print(item, _uitem, c)
                if c == -1:
                    keep = False
                    break
                elif c == 1:
                    _unique.remove(_uitem)
                    self.unique_routes.remove(self.unique_routes[j])
                    break

            if keep:
                _unique.append(item)
                self.unique_routes.append(l[i])

    def _get_bb_path(self, bb_def, rname, i):
        _mol = Chem.MolFromSmiles(bb_def)
        if _mol.HasSubstructMatch(self.core): return ''
        else:
            bb_file = '%s/%s.smi' %(self.reactants[rname][i], rname)
            return bb_file

    def _runRoute(self, route):
        path_collection = []

        for i, r in enumerate(route):
            if i < 2:
                continue
            else:
                if type(r) is list:
                    path_collection += self._runRoute(r)
                else:
                    p = self._get_bb_path(r, route[0], i-2)
                    if p:
                        path_collection.append(p)
        return path_collection