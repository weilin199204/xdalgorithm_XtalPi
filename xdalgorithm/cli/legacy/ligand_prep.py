import os
from collections import defaultdict
import traceback

import pandas as pd
import numpy as np
from tqdm import tqdm

# from copy import deepcopy
from xdalgorithm.utils import (
    load_arguments_from_json, 
)
from xdalgorithm.toolbox.ligand_preparation.ligand_processor import (
    LigandProcessor
)
from xdalgorithm.toolbox.ligand_preparation.ligand_conformation_generator import (
    LigandConformationGenerator
)


CONFIG_TEMPLATE = os.path.join(
    os.path.dirname(__file__),
    '..',
    '..',
    'data',
    'template',
    'ligprep',
    'config.json'
)


class Processor(object):
    def __init__(self, config):
        self.params = config['parameters']
        self.config_template = load_arguments_from_json(
            CONFIG_TEMPLATE
        )
        self.mode = int(config['run_type'])
        self.params2d = self.config_template['parameters']['params2d']
        for _key in self.params['params2d']:
            if _key in self.params2d:
                self.params2d[_key] = self.params['params2d'][_key]

        if self.mode > 2:
            self.params3d = self.config_template['parameters']['params3d']

            for _key in self.params['params3d']:
                if _key in self.params3d:
                    self.params3d[_key] = self.params['params3d'][_key]
         
        # for _key in self.params['params3d']:
        #     if _key in self.params3d:
        #         self.params3d[_key] = self.params['params3d'][_key]
        
        # self.processed2d_smiles_dict = defaultdict(list)
        self.processed2d_smiles_list = []

    @staticmethod
    def collect_2d_processor(
        ligand_smiles_string,
        ligand_molecule_name,
        core_filter_smiles_string,
        core_filter_smarts_string,
        process_isomer,
        max_num_unspecified_chiral_centers,
        protonation,
        # use_chemaxon,
        mode='single',
        output_csv=None,
        other_values=[]
    ):
        processor = LigandProcessor(
            ligand_smiles_string=ligand_smiles_string,
            ligand_molecule_name=ligand_molecule_name,
            core_filter_smiles_string=core_filter_smiles_string,
            core_filter_smarts_string=core_filter_smarts_string,
            process_isomer=process_isomer,
            max_num_unspecified_chiral_centers=max_num_unspecified_chiral_centers,
            protonation=protonation,
            # use_chemaxon=use_chemaxon
        )
        processor.run_processing()

        if mode == 'single':
            processed2d_smiles_list = []
            print('The processed SMILESs are:\n')
            for isomer_smiles, isomer_name in zip(
                processor.ligand_smiles_string_list,
                processor.ligand_molecule_name_list
            ):
                print(isomer_smiles + '\n')
                processed2d_smiles_list.append(
                    (isomer_name, isomer_smiles)
                )
            return processed2d_smiles_list
        elif mode == 'batch':
            for isomer_smiles, isomer_name in zip(
                processor.ligand_smiles_string_list,
                processor.ligand_molecule_name_list
            ):
                with open(output_csv, 'a') as writer:
                    writer.write(
                        f"{ligand_molecule_name},{isomer_name},{ligand_smiles_string},{isomer_smiles}," + \
                            ','.join(other_values)
                    )
                    writer.write("\n")
    @staticmethod
    def collect_3d_processor(
        ligand_smiles_string,
        ligand_molecule_name,
        n_cpu,
        max_num_conformations_per_isomer,
        max_attempts,
        use_random_coords,
        prune_rmsd_threshold,
        enforce_chirality,
        use_exp_torsion_angle_preferences,
        use_basic_knowledge,
        use_small_ring_torsions,
        remove_twisted_six_rings,
        mode='single',
        working_dir_name='.',
    ):
        processor = LigandConformationGenerator(
            ligand_smiles_string=ligand_smiles_string,
            ligand_molecule_name=ligand_molecule_name,
            n_cpu=n_cpu,
            max_num_conformations_per_isomer=max_num_conformations_per_isomer,
            max_attempts=max_attempts,
            use_random_coords=use_random_coords,
            prune_rmsd_threshold=prune_rmsd_threshold,
            enforce_chirality=enforce_chirality,
            use_exp_torsion_angle_preferences=use_exp_torsion_angle_preferences,
            use_basic_knowledge=use_basic_knowledge,
            use_small_ring_torsions=use_small_ring_torsions,
            remove_twisted_six_rings=remove_twisted_six_rings,
            working_dir_name=working_dir_name
        )
        try:
            processor.run_conformation_generation()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            return

        if mode == 'single':
            for conf_name, sdf in zip(
                processor.ligand_molecule_name_list,
                processor.ligand_sdf_file_name_list
            ):
                print(f"Mol name: {conf_name}")
                print(f"SDF file path: {sdf}\n")

            for conf_gen_key in processor.failed_conf_gen_key_list:
                print(conf_gen_key)

        elif mode == 'batch':
            output_csv = os.path.join(working_dir_name, '3d_prepared.csv')
            output_log = os.path.join(working_dir_name, 'failed_conf_gen_molecules.log')

            with open(output_csv, 'a') as csv_writer:
                for conf_name, _smiles, sdf in zip(processor.ligand_molecule_name_list,
                                                   processor.ligand_smiles_string_list,
                                                   processor.ligand_sdf_file_name_list):

                    csv_writer.write(f"{ligand_molecule_name},{_smiles},{conf_name},{sdf}")
                    csv_writer.write('\n')
                    csv_writer.flush()

            with open(output_log, 'a') as log_writer:
                for conf_gen_key in processor.failed_conf_gen_key_list:
                    log_writer.write(conf_gen_key)
                    log_writer.write('\n')
                    log_writer.flush()

    def process2d(self):

        print("Start 2D preparation...")
        
        if self.params2d['input_smiles'] is not None:
            self.processed2d_smiles_list = self.collect_2d_processor(
                ligand_smiles_string=self.params2d['input_smiles'],
                ligand_molecule_name='MOL0000',
                core_filter_smiles_string=None,
                core_filter_smarts_string=None,
                process_isomer=self.params2d['process_isomer'],
                max_num_unspecified_chiral_centers=3,
                protonation=self.params2d['protonation'],
                # use_chemaxon=False,
                mode='single'
            )
            
        if self.params2d['input_csv'] is not None:
            if self.params2d['output_csv'] is None:
                dirname = os.path.dirname(self.params2d['input_csv'])
                filename = os.path.basename(self.params2d['input_csv'])
                prefix = filename.split('.')[0]
                self.params2d['output_csv'] = os.path.join(dirname, f"{prefix}_2d_prep.csv")
    
            df = pd.read_csv(self.params2d['input_csv'])
            
            if self.params2d['smiles_col'] is None:
                self.params2d['smiles_col'] = 'SMILES'

            num_mols = df.shape[0]
            smiles_array = df.loc[
                :, self.params2d['smiles_col']
            ].values.astype('U')

            extra_columns = [x for x in df.columns if not x in (self.params2d['smiles_col'], self.params2d['molname_col'])]
            other_values = df.loc[:, extra_columns].values.astype('U')
            
            if self.params2d['molname_col'] is not None:
                molname_array = df.loc[
                    :, self.params2d['molname_col']
                ].values.astype('U')
            else:
                digit = int(np.log10(num_mols)) + 1
                molname_list = [('MOL{:0' + str(digit) + 'd}').format(mol_idx) for mol_idx in range(num_mols)]

                molname_array = np.array(molname_list, dtype='U')
            
            with open(self.params2d['output_csv'], 'w') as writer:
                writer.write("original_name,output_name,original_smiles,output_smiles,"+','.join(extra_columns))
                writer.write("\n")
            for smiles, molname, ovs in tqdm(
                zip(smiles_array, molname_array, other_values),
                total=num_mols
            ):
                _ = self.collect_2d_processor(
                    ligand_smiles_string=smiles,
                    ligand_molecule_name=molname,
                    core_filter_smiles_string=None,
                    core_filter_smarts_string=None,
                    process_isomer=self.params2d['process_isomer'],
                    max_num_unspecified_chiral_centers=3,
                    protonation=self.params2d['protonation'],
                    mode='batch',
                    output_csv=self.params2d['output_csv'],
                    other_values = ovs
                )
            print(f"Done! 2D processed file is {os.path.abspath(self.params2d['output_csv'])}")

    def process3d(self):
        print("Start 3D preparation...")

        if self.params3d['output_dir'] is None:
            self.params3d['output_dir'] = '.'
        self.params3d['output_dir'] = os.path.abspath(self.params3d['output_dir'])
        output_csv = os.path.join(self.params3d['output_dir'], '3d_prepared.csv')
        output_log = os.path.join(self.params3d['output_dir'], 'failed_conf_gen_molecules.log')

        os.makedirs(self.params3d['output_dir'], exist_ok=True)
            
        with open(output_csv, 'w') as csv_writer:
            csv_writer.write('mol_name,smiles,conf_name,ligand_sdf_file_name')
            csv_writer.write('\n')
        
        with open(output_log, 'w') as log_writer:
            log_writer.write('Identifiers for failed molecules in 3D conformation generation:')
            log_writer.write('\n')

        if self.mode == 3:
            if self.params3d['input_smiles'] is not None:
                self.collect_3d_processor(
                    ligand_smiles_string=self.params3d['input_smiles'],
                    ligand_molecule_name='input_mol',
                    n_cpu=self.params3d['n_cpu'],
                    max_num_conformations_per_isomer=self.params3d['max_num_conformations_per_isomer'],
                    max_attempts=self.params3d['max_attempts'],
                    use_random_coords=True,
                    prune_rmsd_threshold=self.params3d['prune_rmsd_threshold'],
                    enforce_chirality=True,
                    use_exp_torsion_angle_preferences=True,
                    use_basic_knowledge=True,
                    use_small_ring_torsions=True,
                    remove_twisted_six_rings=True,
                    working_dir_name=self.params3d['output_dir'],
                    mode='single',
                )

        if self.mode ==5:
            if self.params2d['input_smiles'] is not None:
                for molname, smiles in self.processed2d_smiles_list:
                    self.collect_3d_processor(
                        ligand_smiles_string=smiles,
                        ligand_molecule_name=molname,
                        n_cpu=self.params3d['n_cpu'],
                        max_num_conformations_per_isomer=self.params3d['max_num_conformations_per_isomer'],
                        max_attempts=self.params3d['max_attempts'],
                        use_random_coords=True,
                        prune_rmsd_threshold=self.params3d['prune_rmsd_threshold'],
                        enforce_chirality=True,
                        use_exp_torsion_angle_preferences=True,
                        use_basic_knowledge=True,
                        use_small_ring_torsions=True,
                        remove_twisted_six_rings=True,
                        working_dir_name=self.params3d['output_dir'],
                        mode='single',
                    )

            if self.params2d['input_csv'] is not None:
                self.params3d['molname_col'] = 'output_name'
                self.params3d['input_csv'] = self.params2d['output_csv']
                self.params3d['smiles_col'] = 'output_smiles'
        
        elif self.params3d['smiles_col'] is None:
            self.params3d['smiles_col'] = 'smiles'

        if self.params3d['input_csv'] is not None:
            df = pd.read_csv(self.params3d['input_csv'])
            # print(self.params3d['input_csv'])
            # print(df.head)
            num_mols = df.shape[0]
            smiles_array = df.loc[
                :, self.params3d['smiles_col']
            ].values.astype('U')

            if self.params3d['molname_col'] is not None:
                molname_array = df.loc[
                    :, self.params3d['molname_col']
                ].values.astype('U')
            else:
                digit = int(np.log10(num_mols)) + 1
                molname_list = [('MOL{:0' + str(digit) + 'd}').format(mol_idx) for mol_idx in range(num_mols)]

                molname_array = np.array(molname_list, dtype='U')

            for smiles, molname in tqdm(
                zip(smiles_array, molname_array),
                total=num_mols
            ): 
                self.collect_3d_processor(
                    ligand_smiles_string=smiles,
                    ligand_molecule_name=molname,
                    n_cpu=self.params3d['n_cpu'],
                    max_num_conformations_per_isomer=self.params3d['max_num_conformations_per_isomer'],
                    max_attempts=self.params3d['max_attempts'],
                    use_random_coords=True,
                    prune_rmsd_threshold=self.params3d['prune_rmsd_threshold'],
                    enforce_chirality=True,
                    use_exp_torsion_angle_preferences=True,
                    use_basic_knowledge=True,
                    use_small_ring_torsions=True,
                    remove_twisted_six_rings=True,
                    working_dir_name=self.params3d['output_dir'],
                    mode='batch',
                )

            print(
                f"Done! Summary table saved at {output_csv}"
            ) 

    def run(self):
        # print(self.mode)
        if self.mode != 3:
            self.process2d()
        if self.mode != 2:
            self.process3d()


class CLICommand:
    __doc__ = "A ligand preparation tool.\n"
    
    with open(CONFIG_TEMPLATE) as f:
        json_raw = f.read()
    
    __doc__ += "The input json file template is: \n" 
    __doc__ += json_raw

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--input-json', default='', type=str)

    @staticmethod
    def run(args):

        config = load_arguments_from_json(args.input_json)
        processor = Processor(config)
        processor.run()