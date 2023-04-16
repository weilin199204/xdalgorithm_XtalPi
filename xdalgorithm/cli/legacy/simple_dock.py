import os
import sys
import multiprocessing as mp

import numpy as np
import pandas as pd
from rdkit import Chem

class XtalDock(object):
    def __init__(self, config):
        self.config = config

    def __get_results__(self):
        protein_indices = []
        names = []
        docked_smiles_list = []
        scores = []
        le = []
        docked_sdf_file_names = os.listdir('docked')

        for docked_sdf_file_name in docked_sdf_file_names:
            protein_idx = docked_sdf_file_name.split('_')[-2]
            names.append(docked_sdf_file_name)
            protein_indices.append(protein_idx)

            rdmol = next(Chem.SDMolSupplier(os.path.join('docked', docked_sdf_file_name), removeHs=False))
            docked_smiles_list.append(Chem.MolToSmiles(rdmol))
            attr_name = 'Score'
            if rdmol.HasProp(attr_name):
                score = float(rdmol.GetProp(attr_name))
            else:
                score = np.NaN

            scores.append(score)
            le.append(score / rdmol.GetNumHeavyAtoms())

        data_dict = {'Ligand_name': names, 'Smiles': docked_smiles_list, 'Protein_idx': protein_indices, 'Score': scores, 'Ligand Efficiency': le}
        data_df = pd.DataFrame(data_dict)
        data_df.to_csv('ligand_docking_results.csv')

    def run(self):
        from xdalgorithm.toolbox.xtaldock.ligand_dock import generate_config, smi2pose
        input_parameters = self.config['parameters']

        if not os.path.isfile(input_parameters['input_ligands']):
            print('%s is not in the working directory!' % input_parameters['input_ligands'])
            sys.exit(0)

        hb_atoms = []
        for hb in input_parameters['hb']:
            hb_atoms.append(set(hb))

        for i, p in enumerate(input_parameters['input_protein_list']):
            if not os.path.isfile(p):
                print('%s is not in the working directory!' % p)
                sys.exit(0)

            os.system('lepro_linux_x86 %s' % p)
            os.system('mv pro.pdb pro.%d.pdb' % i)
            os.system('mv dock.in dock.%d.in' % i)

            if not os.path.isdir('raw_output.%d' % i):
                os.mkdir('raw_output.%d' % i)

        if not os.path.isdir('docked'):
            os.mkdir('docked')
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')

        n_pro = len(input_parameters['input_protein_list'])
        n_jobs = generate_config(input_parameters['center_x'], input_parameters['center_y'],
                                 input_parameters['center_z'], input_parameters['n_poses'],
                                 input_parameters['input_ligands'], input_parameters['n_cpu'], n_pro)

        jobs = []
        for i in range(n_jobs):
            p = mp.Process(target=smi2pose, args=(
                i, n_pro, input_parameters['skip_ligPrep'], input_parameters['ignore'],
                input_parameters['max_saved_poses'],
                input_parameters['rmsd_cutoff'], input_parameters['energy_cutoff'], hb_atoms, input_parameters['prot']))

            jobs.append(p)

        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

        self.__get_results__()


class CLICommand:
    """A tool to predict 3D poses through ensemble docking.
    Input json template:
    {
        "parameters": 
        {
            "input_ligands": "ligand_dock_path/ligands.smi",
            "input_protein_list": 
            [
                "/path/prepared_protein_1.pdb",
                "/path/prepared_protein_2.pdb"
            ],
            "work_dir": ".",
            "n_cpu": 1,
            "skip_ligPrep": false,
            "ignore": 3,
            "max_saved_poses": 3,
            "energy_cutoff": 2.0,
            "rmsd_cutoff": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "n_poses": 20,
            "hb": [],
            "prot": true,
        }
    }
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--input-json', default='', type=str)

    @staticmethod
    def run(args):
        from xdalgorithm.utils import load_arguments_from_json
        configuration = load_arguments_from_json(args.input_json)

        work_dir = configuration.get('parameters', ValueError).get('work_dir', '.')
        if not os.path.exists(work_dir) or not os.path.isdir(work_dir):
            os.mkdir(work_dir)

        os.chdir(work_dir)

        xtal_dock = XtalDock(configuration)
        xtal_dock.run()
