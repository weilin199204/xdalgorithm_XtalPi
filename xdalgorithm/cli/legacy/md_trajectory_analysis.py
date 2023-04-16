import os
import multiprocessing as mp

class MDAnalyzer(object):
    def __init__(self, config):
        self.config = config
        self.analysis_parameters = self.config['parameters']

        if 'work_dir' in self.analysis_parameters.keys():
            self.work_dir = os.path.abspath(self.analysis_parameters['work_dir'])
        else:
            self.work_dir = '.'

        os.chdir(self.work_dir)

        self.n_parallel_MD = self.analysis_parameters['n_parallel_MD']
        self.parallel_MD_dir = []
        for i in range(self.n_parallel_MD):
            self.parallel_MD_dir.append('./md_parallel_sampling_' + str(i))

        self.ligand_resnames = self.analysis_parameters['ligand_resnames']
        self.saved_selection_str = 'protein or resname ' + ' or resname '.join(self.ligand_resnames)

        unsolvated_prmtop_file_name_split = self.analysis_parameters['unsolvated_prmtop_file_name'].split('/')
        unsolvated_inpcrd_file_name_split = self.analysis_parameters['unsolvated_inpcrd_file_name'].split('/')
        solvated_prmtop_file_name_split = self.analysis_parameters['solvated_prmtop_file_name'].split('/')

        if len(unsolvated_prmtop_file_name_split) == 1:
            self.unsolvated_prmtop_file_name = self.work_dir + '/' + unsolvated_prmtop_file_name_split[0]
        else:
            self.unsolvated_prmtop_file_name = self.analysis_parameters['unsolvated_prmtop_file_name']

        if len(unsolvated_inpcrd_file_name_split) == 1:
            self.unsolvated_inpcrd_file_name = self.work_dir + '/' + unsolvated_inpcrd_file_name_split[0]
        else:
            self.unsolvated_inpcrd_file_name = self.analysis_parameters['unsolvated_inpcrd_file_name']

        if len(solvated_prmtop_file_name_split) == 1:
            self.solvated_prmtop_file_name = self.work_dir + '/' + solvated_prmtop_file_name_split[0]
        else:
            self.solvated_prmtop_file_name = self.analysis_parameters['solvated_prmtop_file_name']

    def __perform_processing_analysis__(self, sampling_dir):
        from xdalgorithm.toolbox.md.analysis.trajectory_processing import align_trajectory
        from xdalgorithm.toolbox.md.analysis.rmsd_hbonds_analysis import analyze_trajectory

        os.chdir(sampling_dir)
        align_trajectory(self.solvated_prmtop_file_name, self.unsolvated_prmtop_file_name, self.unsolvated_inpcrd_file_name, 'trajectory_prod.dcd', self.saved_selection_str, 'protein_ligand.dcd', 'protein_ligand_last_frame.pdb')
        analysis_summary_df = analyze_trajectory(self.unsolvated_prmtop_file_name, 'protein_ligand.dcd', rmsd_selection_str='not protein', mass_weighted=True, hbond_cutoff_frequency=0.3)
        analysis_summary_df.to_csv('md_analysis_summary.csv')

    def run(self):
        md_analysis_processes = []
        for i in range(self.n_parallel_MD):
            current_parallel_MD_dir = self.parallel_MD_dir[i]
            print(current_parallel_MD_dir)
            md_analysis_process = mp.Process(target=self.__perform_processing_analysis__, args=(current_parallel_MD_dir,))
            md_analysis_processes.append(md_analysis_process)

        for md_analysis_process in md_analysis_processes:
            md_analysis_process.start()
        for md_analysis_process in md_analysis_processes:
            md_analysis_process.join()


class CLICommand:
    __doc__ = "Analyze results and trajectories from parallel regular MD simulations.\n" + \
    """ Input json template:
    {
        "parameters":
        {
            "work_dir": ".",
            "n_parallel_MD": 1,
            "ligand_resnames": ["MOL"],
            "unsolvated_prmtop_file_name": "/the/prmtop/file.prmtop",
            "unsolvated_inpcrd_file_name": "/the/inpcrd/file.prmtop",
            "solvated_prmtop_file_name": "/the/prmtop/file.prmtop"
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

        md_analyzer = MDAnalyzer(configuration)
        md_analyzer.run()
