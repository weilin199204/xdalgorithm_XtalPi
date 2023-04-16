import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from collections import defaultdict
import os
from tqdm import tqdm

# configuration
# source_file_name
SMARTS_DICT = {
    'primary_secondary_halides': '[CX4;h][F,Cl,Br,I]',
    'monosubstituted_alkynes': '[CH]#[C]',
    'azides': '[N]=[N+]=[N-]',
    'aryl_nitriles': '[C&$(C*:*)]#[N]',
    'amidines': '[NX3;h][C&$(C[#6])]=[NH]',
    'nitriles': '[C&$(C[#6])]#[N]',
    'hydrazides': 'O=[C&$(C[#6])][NH][NH2]',
    'carboxylic_acids': '[C;H1,$(C[#6])](=O)[O;H1,X1]',
    'primary_secondary_amines': '[N;H2&$(N[#6]),H1&$(N([#6])[#6]);!$(NC=O)]',
    '2_halo_heterocycles': '[cr5&$(c([nr5])[nr5,or5,sr5])][F,Cl,Br,I]',
    '2_unsubstituted_heterocycles': '[ch;r5;$(c[nr5,or5,sr5])]',
    'aryl_halides': '[F,Cl,Br,I][*&$(*:*)]',
    'aryl_boronates': '[*&$(*:*)]B(O)O',
    'azoles': '[nH;r5;$(n:[nr5,or5,sr5]),$(n:[cr5]:[nr5,or5,sr5]),$(n1ccc2ccccc21)]',
    'acid_chlorides': '[C;H1,$(C[#6])](=[O])Cl',
    'heterocycles': '[nH]',
    'primary_amides': '[C;H1,$(C[#6])](=O)[N;H2,H1&$(N[#6])]',
    'aldehydes_and_ketones': '[C;H1&$(C[#6]),H2,H0&$(C([#6])[#6])](=O)',
    '2-unsubstituted_ketones': '[CH2][C;H1&$(C[#6]),$(C([#6])[#6])](=O)',
    '2-cyano_esters': 'N#C[CH2]C(=O)O[#6]',
    'ortho-amino_anilines': '[NH2]c1ccccc1[Nh]',
    'aldehydes': '[C;H1&$(C[#6]),H2]=O',
    '2-iodo_phenols': 'Ic1ccccc1[OH]',
    '2-mercapto_anilines': '[SH]c1ccccc1[NH2]',
    '2-iodo_thioanisoles': 'Ic1ccccc1S[CH3]',
    '2-amino_phenols': '[NH2]c1ccccc1[OH]',
    'aryl_aldehydes': '[CH&$(C*:*)](=O)',
    'isocyanates': '[N;H1,$(N[#6])]=[C]=[O]',
    'primary_secondary_alcohols': '[CX4;h][OH]',
    '2-iodo_anilines': 'Ic1ccccc1[Nh]',
    'aryl_vinyl_halides': '[#6;$([#6]:*),H1&$([#6](=[#6])),$([#6](=C)[#6])][F,Cl,Br,I]',
    'enolizable_ketones': '[CX4;h][C](=[O])[#6]',
    'non-enolizable_esters': '[#6]O[C](=[O])[#6;H0]',
    'enolizable_esters': '[#6][O][C](=[O])[CX4;h]',
    '3-ketoesters': '[#6][C](=[O])[CX4;h][C](=[O])O[#6]',
    'anilines': 'c1ccccc1[NH2]',
    'unsaturated_ketones': '[C&$(C([#6])[#6])](=O)C=[CX3]',
    'phenols': 'c1ccccc1[OH]',
    'aryl_carboxylic_acids': '[O;H1,X1]C(=O)[*&$(*:*)]',
    'n-acyl_glycines': '[C;H1,$(C[#6])](=O)[NH][CH2][C](=[O])[O;H,X1]',
    'alcohols': '[CX4][OH]',
    'primary_secondary_aryl_alcohols': '[#6;X4,$([#6]a)][OH]',
    '2-keto_phenols': '[OH]c1ccccc1C(=O)[CX4;h]',
    'aryl_acid_chlorides': '[C&$(C*:*)](=O)Cl',
    'Friedel-Crafts_substrates': '[ch]',
    '2-halo_ketones': '[C&$(C([#6])C)](=O)[CX4;H1&$(C(C)[#6]),H2][F,Cl,Br,I]',
    'activated_alkenes': '[CH2]=[C;$(C*:*),$(CC(=O)[#6,O]),$(CC#N),$(CO[#6]),$(C[NH]*:*)]',
    '2-halo_aldehydes': '[CH](=O)C[F,Cl,Br,I]',
    '2-halo_aryl_ketones': '[C&$(C*:*)](=O)[CX4;h][F,Cl,Br,I]',
    'diaryl_diketones': '[C&$(C*:*)](=O)[C&$(C*:*)]=O',
    'aryl_hydrazines': '[cH]1ccccc1[NH][NH2]',
    'primary_amines': '[NH2;$(N[#6]);!$(NC=O)]',
    'primary_secondary_sulfonamides': '[N;H1&$(N([#6])S(=O)(=O)[#6]),H2;!$(NC=O)]',
    'tetrazoles-2H': '[nH&$(n1ncnn1)]',
    'tetrazoles-1H': '[nH&$(n1nnnc1)]',
    'aryl_alkynyl_vinyl_halides': '[#6;$([#6]:*),$([#6]#[C]),H1&$([#6](=[#6])),$([#6](=C)[#6])][F,Cl,Br,I]',
    'cyanohydrins': '[CX4;H1&$(C([#6])C),$(C([#6])([#6])C)]([OH])C#N',
    'acetonitrile_proprionitrile': '[C;$(C[CH3]),$(C[CH2][CH3])]#N',
    '2-acyl_benzoic_acids': 'C(=O)([O;H,X1])c1ccccc1[C;H1,$(C([#6])c)]=O',
    'hydrazines': '[NH2][NH1&$(N[#6])]',
    'arylethyl_amines': '[ch]:c[CX4][CX4][N;H1&$(N(C)[#6]),$(N([#6])(C)[#6]);!$(NC=O)]',
    # 'aryl_ketones_aldehydes': '[ch]:c[C;H1,$(C(c)[#6])]=O',
    '3-keto_ketones': '[C&$(C([#6])C)](=O)[CX4;h][C&$(C([#6])C)](=O)',
    'cyanoacetamides': '[N;H2,H1&$(N(C)[#6])]C(=O)[CH2&$(CC#N)]',
    'amino_acrylates': '[N;H2,H1&$(N[#6])][C;H1,$(C[#6])]=[C;H1,$(C[#6])]C(=O)[O;H1,X1,$(O(C)[#6])]',
    '2-aminobenzoates': '[N;H2,H1&$(N[#6])]c1ccccc1C(=O)[O;H1,X1]',
    '4-keto_ketones': '[C&$(C([#6])C)](=O)[CX4;h][CX4;h][C&$(C([#6])C)]=O',
    'quinazolinones': 'n1c2ccccc2c(=O)[nH]c1',
    '2-carboxy_anilines': '[NH2]c1ccccc1C(=O)[O;H1,X1]',
    '2-amino_benzaldehydes': '[NH2]c1ccccc1[C;H1,$(C(c)[#6])]=O',
    'o-hydroxyacetophenones': '[OH]c1ccccc1C(=O)[CH3]',
    'cyclohexanones': 'O=[C&$(C1CCCCC1)]',
    'organostannanes': '[#6;$([#6]:*),H1&$([#6](=[#6])),$([#6](=C)[#6])][Sn&$([Sn]([CX4])[CX4])]',
    'sulfonyl_chlorides': '[S&$(S(=O)(=O)[#6])]Cl',
    'aryl_vinyl_boronates': '[#6;$([#6]:*),H1&$([#6](=[#6])),$([#6](=C)[#6])]B(O)O',
    '2-hydroxy_ketones': '[C&$(C(C)[#6])](=O)[CX4;H2,H1&$(C(C)[#6])][OH]',
    'thioamides': '[SX1]=[C&H1,$(C[#6])][NH2]',
    'styrenes': '[CX3&$(Cc1ccccc1)]=[CX3]',
    'thiols': '[SH;X2;$(S[#6])]',
    'thioisocyanates': 'S=C=[N;H1,$(N[#6])]'
}


def get_matched_position(reaction, mol):
    reaction.Initialize()
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

class BuildingBlockesCreator:
    def __init__(self, configuration):
        self.source_file_name = configuration["source_file_name"]
        # self.prepared_smiles = configuration["prepared_smiles"]
        # self.bb_dir = configuration["building_blocks_path"]  # useless
        self.sorted_bb_dir = configuration["sorted_bb_dir"]
        self.reaction_dir = configuration["reaction_dir"]

        self.filters = {'MW': 350, 'RB': 10, 'HBA': 10, 'HBD': 6, 'PSA': 160}
        self.reactants = self.load_reactants()
        self.synthesis_reactions = self.load_synthesis()

    @staticmethod
    def _del_ion_component(_smiles):
        if '.' in _smiles:
            components = _smiles.split('.')
            i = np.argmax([len(x) for x in components])
            return components[i]
        else:
            return _smiles

    def _is_remained(self, smiles):
        smiles = self._del_ion_component(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            q = QED.properties(mol)
            if q.MW < self.filters['MW'] and q.ROTB < self.filters['RB'] and \
                    q.HBA < self.filters['HBA'] and q.HBD < self.filters['HBD'] and \
                    q.PSA < self.filters['PSA']:
                return True
        return False

    def _load_filter_building_blocks(self):
        collected_bb_smiles_list = []
        with open(self.source_file_name) as file_handler:
            while True:
                line = file_handler.readline().strip()
                if line == '':
                    break
                smiles = self._del_ion_component(line)
                if self._is_remained(smiles):
                    collected_bb_smiles_list.append(smiles)
        return list(set(collected_bb_smiles_list))

    def load_reactants(self):
        reactant_reaction_dict = dict()
        reactants_path = os.path.join(self.reaction_dir, 'reactants.txt')
        with open(reactants_path) as reaction_reader:
            for reaction_line in reaction_reader:
                items = reaction_line.split()
                reaction_name = items[0]
                if len(items) > 1:
                    for i, reactant_name in enumerate(items[1:]):
                        if reactant_name not in reactant_reaction_dict.keys():
                            reactant_reaction_dict[reactant_name] = dict()
                        if reaction_name not in reactant_reaction_dict[reactant_name].keys():
                            reactant_reaction_dict[reactant_name][reaction_name] = list()
                        reactant_reaction_dict[reactant_name][reaction_name].append(i)
        assert len(SMARTS_DICT.keys()) == len(reactant_reaction_dict.keys())
        assert len(set(SMARTS_DICT.keys()) - set(reactant_reaction_dict.keys())) == 0
        assert len(set(reactant_reaction_dict.keys()) - set(SMARTS_DICT.keys())) == 0
        return reactant_reaction_dict

    def load_synthesis(self):
        synthesis_reactions = {}
        with open(os.path.join(self.reaction_dir, 'synthesis.txt')) as f:
            for rd in f:
                name, smarts = rd.split()
                reaction = AllChem.ReactionFromSmarts(smarts)
                synthesis_reactions[name] = reaction
        return synthesis_reactions

    @staticmethod
    def sort(smiles_list):
        reactant_smiles_dict = defaultdict(list)
        mols_list = [(smi, Chem.MolFromSmiles(smi)) for smi in smiles_list]
        smt_mol_dict = {reactant_name: Chem.MolFromSmarts(SMARTS_DICT[reactant_name]) for reactant_name in SMARTS_DICT.keys()}
        for each_smi, each_mol in tqdm(mols_list):
            for reactant_name in smt_mol_dict.keys():
                patt = smt_mol_dict[reactant_name]
                n = len(each_mol.GetSubstructMatches(patt))
                if n == 1:
                    reactant_smiles_dict[reactant_name].append(each_smi)
        return reactant_smiles_dict

    def create_building_blocks(self):
        print('start to collect smiles:')
        collected_smiles_list = self._load_filter_building_blocks()
        print('sort smiles by reactant types:')
        reactant_smiles_dict = self.sort(collected_smiles_list)
        print('start to create building blocks:')
        for reactant in tqdm(self.reactants):
            self.sort_by_reactions(reactant, reactant_smiles_dict[reactant])

    def sort_by_reactions(self, reactant, smiles_list):
        if not os.path.exists(self.sorted_bb_dir):
            os.mkdir(self.sorted_bb_dir)
        sub_dir_path = os.path.join(self.sorted_bb_dir, reactant)
        if not os.path.exists(sub_dir_path):
            os.mkdir(sub_dir_path)

        total_dict = dict()
        reaction_list = self.reactants[reactant]
        for each_smiles in smiles_list:
            mol = Chem.MolFromSmiles(each_smiles)
            for each_reaction in reaction_list:
                reaction_obj = self.synthesis_reactions[each_reaction]
                if (reactant, each_reaction) not in total_dict.keys():
                    total_dict[(reactant, each_reaction)] = []
                matched_position, _ = get_matched_position(reaction_obj, mol)
                if set(matched_position) == set(self.reactants[reactant][each_reaction]):
                    total_dict[(reactant, each_reaction)].append(Chem.MolToSmiles(mol))
        for key in total_dict.keys():
            reactant_name, reaction_name = key
            smiles_file_path = os.path.join(self.sorted_bb_dir, reactant_name, '{0}.smi'.format(reaction_name))
            content = '\n'.join(total_dict[key]) + '\n'
            with open(smiles_file_path, 'a') as writer:
                writer.write(content)
