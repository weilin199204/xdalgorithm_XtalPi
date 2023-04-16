import os
import sys
import shutil
import multiprocessing as mp

import numpy as np
import pandas as pd
from rdkit import Chem

class ConstraintDock(object):
    def __init__(self, config):
        self.config = config

    def __get_results__(self):
        names = []
        docked_smiles_list = []
        scores = []
        le = []
        docked_sdf_file_names = os.listdir('docked')

        for docked_sdf_file_name in docked_sdf_file_names:
            if os.path.getsize(os.path.join('docked', docked_sdf_file_name)) == 0:
                continue

            names.append(docked_sdf_file_name)
            rdmol = next(Chem.SDMolSupplier(os.path.join('docked', docked_sdf_file_name), removeHs=False))

            docked_smiles_list.append(Chem.MolToSmiles(rdmol))
            attr_name = 'Docking Energy'
            if rdmol.HasProp(attr_name):
                score = float(rdmol.GetProp(attr_name))
            else:
                score = np.NaN

            scores.append(score)
            le.append(score / rdmol.GetNumHeavyAtoms())

        data_dict = {'Ligand_name': names, 'Smiles': docked_smiles_list, 'Score': scores, 'Ligand Efficiency': le}
        data_df = pd.DataFrame(data_dict)
        data_df.to_csv('core_constraint_docking_results.csv')

    def run(self):
        from xdalgorithm.utils import load_arguments_from_json
        from xdalgorithm.toolbox.xtaldock.core_constraint_docking import process_protein, core_based_docking, process_core
        from xdalgorithm.toolbox.ligand_preparation.utils import smi_file_reader, sdf_file_reader

        input_parameters = self.config['parameters']
        if not os.path.isfile(input_parameters['input_ligands']):
            print('%s is not in the workding directory!' % input_parameters['input_ligands'])
            sys.exit(0)

        if not os.path.isfile(input_parameters['core']):
            print('%s is not in the workding directory!' % input_parameters['core'])
            sys.exit(0)
        else:
            core = process_core(input_parameters['core'])

        npt = input_parameters['box_size'] * 2 / input_parameters['spacing']
        if abs(int(npt) - npt) > 1e-5:
            print('Cannot evenly create the grids!')
            sys.exit(0)

        npt = int(npt)
        N = npt + 1
        process_protein(input_parameters['input_protein'], npt, input_parameters['spacing'])

        if not os.path.isfile('pro.pdb') or not os.path.isfile('protein.e.map'):
            print('Failed to generate maps for protein. Exit!')
            sys.exit(0)

        if input_parameters['hb']:
            hb_atoms = set(input_parameters['hb'])
            os.system('mkdir -p tmp')  # save temporary complex pdb
        else:
            hb_atoms = set()

        pro_txt = ''.join([x for x in open('pro.pdb').readlines() if x[:4] == 'ATOM'])

        file_type = os.path.basename(input_parameters['input_ligands']).split('.')[-1]
        if file_type == 'smi' or file_type == 'smiles':
            mol_list = smi_file_reader(input_parameters['input_ligands'])
        elif file_type == 'sdf':
            mol_list = sdf_file_reader(input_parameters['input_ligands'])
        else:
            raise ValueError('Specified file type is not supported. Please provide with a smi or sdf file!')

        os.system('mkdir -p docked')
        num_molecules = len(mol_list)
        core_constraint_docking_process_list = [None] * num_molecules

        for mol_idx in range(num_molecules):
            mol = mol_list[mol_idx]
            molecule_name = 'ligand_' + str(mol_idx)
            core_constraint_docking_process = mp.Process(target=core_based_docking, args=(mol,
                                                                                          molecule_name,
                                                                                          core,
                                                                                          input_parameters['box_size'],
                                                                                          N,
                                                                                          input_parameters['save_best'],
                                                                                          input_parameters['n_poses'],
                                                                                          input_parameters['rmsd_cutoff'],
                                                                                          hb_atoms,
                                                                                          pro_txt,
                                                                                          input_parameters['protonation']))

            core_constraint_docking_process_list[mol_idx] = core_constraint_docking_process

        for core_constraint_docking_process in core_constraint_docking_process_list:
            core_constraint_docking_process.start()
        for core_constraint_docking_process in core_constraint_docking_process_list:
            core_constraint_docking_process.join()

        self.__get_results__()


class CLICommand:
    __doc__ = "A tool to predict 3D poses through core constraint docking.\n" + \
    """ Input json template:
    {
        "parameters":
        {
            "input_ligands": "ligands.smi",
            "input_protein": "prepared_protein.pdb",
            "work_dir": ".",
            "core": "ligand_core.sdf",
            "n_cpu": 1,
            "box_size": 9,
            "spacing": 0.6,
            "save_best": true,
            "rmsd_cutoff": 2.0,
            "n_poses": 10,
            "hb": [],
            "protonation": True
        }
    }
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f','--input-json',default='',type=str)

    @staticmethod
    def cp_file_to_workdir(configuration):
        def copy_file(old_path):
            if os.path.exists(old_path):
                file_name = os.path.basename(old_path)
                new_path = os.path.join(configuration.get('parameters', ValueError).get('work_dir', '.'), file_name)
                if os.path.abspath(new_path) != os.path.abspath(old_path):
                    shutil.copyfile(old_path, new_path)

        copy_file(configuration['parameters']['input_ligands'])
        copy_file(configuration['parameters']['input_protein'])
        copy_file(configuration['parameters']['core'])

    @staticmethod
    def run(args):
        from xdalgorithm.utils import load_arguments_from_json
        configuration = load_arguments_from_json(args.input_json)

        work_dir = configuration.get('parameters', ValueError).get('work_dir', '.')
        if not os.path.exists(work_dir) or not os.path.isdir(work_dir):
            os.mkdir(work_dir)

        os.chdir(work_dir)

        CLICommand.cp_file_to_workdir(configuration)

        constraint_dock = ConstraintDock(configuration)
        constraint_dock.run()
