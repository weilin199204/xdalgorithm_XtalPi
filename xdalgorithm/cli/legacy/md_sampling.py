import os
from copy import deepcopy
import multiprocessing as mp

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *
import parmed as pmd

class MDSampler(object):
    def __init__(self, config):
        self.config = config
        self.simulation_parameters = self.config['parameters']
        self.unsolvated_prmtop_file_name = self.simulation_parameters['unsolvated_prmtop_file_name']
        self.unsolvated_inpcrd_file_name = self.simulation_parameters['unsolvated_inpcrd_file_name']
        self.solvated_prmtop_file_name = self.simulation_parameters['solvated_prmtop_file_name']
        self.solvated_inpcrd_file_name = self.simulation_parameters['solvated_inpcrd_file_name']
        self.ligand_resnames = self.simulation_parameters['ligand_resnames']

        self.n_parallel_MD = self.simulation_parameters['n_parallel_MD']
        for i in range(self.n_parallel_MD):
            os.mkdir('./md_parallel_sampling_' + str(i))

        self.n_gpu = int(self.simulation_parameters['n_gpu'])

        self.platform_name = self.simulation_parameters['platform_name']

        if (self.simulation_parameters['platform_property'] is not None) and (not isinstance(self.simulation_parameters['platform_property'], dict)):
            raise ValueError('platform_property should be a dict or None type')
        else:
            self.platform_property = self.simulation_parameters['platform_property']

        self.gen_temperature = float(self.simulation_parameters['gen_temperature']) * kelvin
        self.temperature = float(self.simulation_parameters['temperature']) * kelvin

        self.pcouple = self.simulation_parameters['pcouple']
        self.p_ref = float(self.simulation_parameters['p_ref']) * bar
        self.p_type = self.simulation_parameters['p_type']
        self.p_scale = self.simulation_parameters['p_scale']

        if self.simulation_parameters['p_xy_mode'] == 'constant':
            self.p_xy_mode = MonteCarloMembraneBarostat.ConstantVolume
        elif self.simulation_parameters['p_xy_mode'] == 'anisotropic':
            self.p_xy_mode = MonteCarloMembraneBarostat.XYAnisotropic
        elif self.simulation_parameters['p_xy_mode'] == 'isotropic':
            self.p_xy_mode = MonteCarloMembraneBarostat.XYIsotropic
        else:
            self.p_xy_mode = self.simulation_parameters['p_xy_mode']

        if self.simulation_parameters['p_z_mode'] == 'constant':
            self.p_z_mode = MonteCarloMembraneBarostat.ConstantVolume
        elif self.simulation_parameters['p_z_mode'] == 'fixed':
            self.p_z_mode = MonteCarloMembraneBarostat.ZFixed
        elif self.simulation_parameters['p_z_mode'] == 'free':
            self.p_z_mode = MonteCarloMembraneBarostat.ZFree
        else:
            self.p_z_mode = self.simulation_parameters['p_z_mode']

        self.p_tens = float(self.simulation_parameters['p_tens']) * bar * nanometers
        self.p_freq = int(self.simulation_parameters['p_freq'])

        self.mini_tol = float(self.simulation_parameters['mini_tol']) * kilojoule / mole / nanometers
        self.mini_nstep = int(self.simulation_parameters['mini_nstep'])

        self.fric_coeff = float(self.simulation_parameters['fric_coeff']) / picoseconds
        self.dt = float(self.simulation_parameters['dt']) * picoseconds

        self.n_step_heating = int(self.simulation_parameters['n_step_heating'])
        self.n_step_NVT_eq = int(self.simulation_parameters['n_step_NVT_eq'])
        self.n_step_NPT_eq = int(self.simulation_parameters['n_step_NPT_eq'])
        self.n_step_prod = int(self.simulation_parameters['n_step_prod'])

        if int(self.simulation_parameters['n_step_dcd_heating']) == 0:
            self.n_step_dcd_heating = self.n_step_heating
        else:
            self.n_step_dcd_heating = int(self.simulation_parameters['n_step_dcd_heating'])

        if int(self.simulation_parameters['n_step_dcd_NVT_eq']) == 0:
            self.n_step_dcd_NVT_eq = self.n_step_NVT_eq
        else:
            self.n_step_dcd_NVT_eq = int(self.simulation_parameters['n_step_dcd_NVT_eq'])

        if int(self.simulation_parameters['n_step_dcd_NPT_eq']) == 0:
            self.n_step_dcd_NPT_eq = self.n_step_NPT_eq
        else:
            self.n_step_dcd_NPT_eq = int(self.simulation_parameters['n_step_dcd_NPT_eq'])

        if int(self.simulation_parameters['n_step_dcd_prod']) == 0:
            self.n_step_dcd_prod = self.n_step_prod
        else:
            self.n_step_dcd_prod = int(self.simulation_parameters['n_step_dcd_prod'])

        if self.simulation_parameters['nonbonded_method'] == 'LJPME':
            self.nonbonded_method = LJPME
        elif self.simulation_parameters['nonbonded_method'] == 'PME':
            self.nonbonded_method = PME
        elif self.simulation_parameters['nonbonded_method'] == 'Ewald':
            self.nonbonded_method = Ewald
        elif self.simulation_parameters['nonbonded_method'] == 'CutoffPeriodic':
            self.nonbonded_method = CutoffPeriodic
        elif self.simulation_parameters['nonbonded_method'] == 'CutoffNonPeriodic':
            self.nonbonded_method = CutoffNonPeriodic
        elif self.simulation_parameters['nonbonded_method'] == 'NoCutoff':
            self.nonbonded_method = NoCutoff

        self.nonbonded_cutoff = float(self.simulation_parameters['nonbonded_cutoff']) * nanometers
        self.switch_distance = float(self.simulation_parameters['switch_distance']) * nanometers

        if self.simulation_parameters['constraints'] == 'HBonds':
            self.constraints = HBonds
        elif self.simulation_parameters['constraints'] == 'HAngles':
            self.constraints = HAngles
        elif self.simulation_parameters['constraints'] == 'AllBonds':
            self.constraints = AllBonds
        else:
            self.constraints = self.simulation_parameters['constraints']

        self.restraints_force_constant = float(self.simulation_parameters['restraints_force_constant'])
        if self.simulation_parameters['restraints_selection_string'] == 'auto':
            self.restraints_selection_string = 'protein or resname ' + ' or resname '.join(self.ligand_resnames) + ' and not name H*'
        elif self.simulation_parameters['restraints_selection_string'] is None:
            self.restraints_selection_string = 'not all'
            self.restraints_force_constant = 0.0
        else:
            self.restraints_selection_string = self.simulation_parameters['restraints_selection_string']

        self.rigid_water = self.simulation_parameters['rigid_water']

        if self.simulation_parameters['implicit_solvent'] == 'HCT':
            self.implicit_solvent = HCT
        elif self.simulation_parameters['implicit_solvent'] == 'OBC1':
            self.implicit_solvent = OBC1
        elif self.simulation_parameters['implicit_solvent'] == 'OBC2':
            self.implicit_solvent = OBC2
        elif self.simulation_parameters['implicit_solvent'] == 'GBn':
            self.implicit_solvent = GBn
        else:
            self.implicit_solvent = self.simulation_parameters['implicit_solvent']

        if self.simulation_parameters['implicit_solvent_kappa'] is None:
            self.implicit_solvent_kappa = None
        elif isinstance(self.simulation_parameters['implicit_solvent_kappa'], (float, int)):
            self.implicit_solvent_kappa = float(self.simulation_parameters['implicit_solvent_kappa']) / nanometers

        self.implicit_solvent_salt_conc = float(self.simulation_parameters['implicit_solvent_salt_conc']) * moles / liter
        self.solute_dielectric = float(self.simulation_parameters['solute_dielectric'])
        self.solvent_dielectric = float(self.simulation_parameters['solvent_dielectric'])
        self.use_SASA = self.simulation_parameters['use_SASA']
        self.remove_CM_motion = self.simulation_parameters['remove_CM_motion']

        if self.simulation_parameters['use_HMR'] is True:
            if isinstance(self.simulation_parameters['hydrogen_mass'], (float, int)):
                self.hydrogen_mass = float(self.simulation_parameters['hydrogen_mass']) * amu
            else:
                raise ValueError('Hydrogen mass should be a float number')
        else:
            self.hydrogen_mass = None

        self.ewald_error_tolerance = float(self.simulation_parameters['ewald_error_tolerance'])
        self.flexible_constraints = self.simulation_parameters['flexible_constraints']
        self.verbose = self.simulation_parameters['verbose']
        self.split_dihedrals = self.simulation_parameters['split_dihedrals']

        self.structure = pmd.load_file(self.solvated_prmtop_file_name, xyz=self.solvated_inpcrd_file_name)

    def __allocate_openmm_GPU_devices__(self):
        max_num_of_cuda_devices = self.n_gpu
        allocated_cuda_properties = [None] * self.n_parallel_MD

        current_cuda_idx = 0
        for md_process_idx in range(self.n_parallel_MD):
            current_cuda_property = deepcopy(self.platform_property)
            current_cuda_property['DeviceIndex'] = str(current_cuda_idx)
            allocated_cuda_properties[md_process_idx] = current_cuda_property
            current_cuda_idx += 1

            if current_cuda_idx >= max_num_of_cuda_devices:
                current_cuda_idx = 0
            else:
                continue

        return allocated_cuda_properties

    def __perform_MD_sampling__(self, output_prefix, platform_property):
        from xdalgorithm.toolbox.md.sampling.md_sampler import MDSimulations
        md_sampler = MDSimulations(self.structure,
                                   platform_name=self.platform_name,
                                   platform_property=platform_property,
                                   gen_temperature=self.gen_temperature,
                                   temperature=self.temperature,
                                   p_couple=self.pcouple,
                                   p_ref=self.p_ref,
                                   p_type=self.p_type,
                                   p_scale=self.p_scale,
                                   p_xy_mode=self.p_xy_mode,
                                   p_z_mode=self.p_z_mode,
                                   p_tens=self.p_tens,
                                   p_freq=self.p_freq,
                                   fric_coeff=self.fric_coeff,
                                   mini_nstep=self.mini_nstep,
                                   mini_tol=self.mini_tol,
                                   dt=self.dt,
                                   n_step_heating=self.n_step_heating,
                                   n_step_NVT_eq=self.n_step_NVT_eq,
                                   n_step_NPT_eq=self.n_step_NPT_eq,
                                   n_step_prod=self.n_step_prod,
                                   n_step_dcd_heating=self.n_step_dcd_heating,
                                   n_step_dcd_NVT_eq=self.n_step_dcd_NVT_eq,
                                   n_step_dcd_NPT_eq=self.n_step_dcd_NPT_eq,
                                   n_step_dcd_prod=self.n_step_dcd_prod,
                                   nonbonded_method=self.nonbonded_method,
                                   nonbonded_cutoff=self.nonbonded_cutoff,
                                   switch_distance=self.switch_distance,
                                   constraints=self.constraints,
                                   restraints_selection_string=self.restraints_selection_string,
                                   restraints_force_constant=self.restraints_force_constant,
                                   rigid_water=self.rigid_water,
                                   implicit_solvent=self.implicit_solvent,
                                   implicit_solvent_kappa=self.implicit_solvent_kappa,
                                   implicit_solvent_salt_conc=self.implicit_solvent_salt_conc,
                                   solute_dielectric=self.solute_dielectric,
                                   solvent_dielectric=self.solvent_dielectric,
                                   use_SASA=self.use_SASA,
                                   remove_CM_motion=self.remove_CM_motion,
                                   hydrogen_mass=self.hydrogen_mass,
                                   ewald_error_tolerance=self.ewald_error_tolerance,
                                   flexible_constraints=self.flexible_constraints,
                                   verbose=self.verbose,
                                   split_dihedrals=self.split_dihedrals)

        md_sampler.create_openmm_system()
        md_sampler.perform_minimization()
        if md_sampler.n_step_heating > 0:
            md_sampler.perform_NVT_heating()
        if md_sampler.n_step_NVT_eq > 0:
            md_sampler.perform_NVT_equilibration()
        if md_sampler.n_step_NPT_eq > 0:
            md_sampler.perform_NPT_equilibration()
        if md_sampler.n_step_prod > 0:
            md_sampler.perform_production()

    def run(self):
        md_sampling_processes = []

        if self.platform_name == 'CUDA' or self.platform_name == 'OpenCL':
            allocated_cuda_properties = self.__allocate_openmm_GPU_devices__()
            for i in range(self.n_parallel_MD):
                current_output_prefix = './md_parallel_sampling_' + str(i)
                md_sampling_process = mp.Process(target=self.__perform_MD_sampling__, args=(current_output_prefix, allocated_cuda_properties[i]))
                md_sampling_processes.append(md_sampling_process)

        elif self.platform_name == 'OpenCL':
            raise NotImplementedError('MD parallel processing currently not supported for OpenCL platform')

        elif self.platform_name == 'CPU':
            for i in range(self.n_parallel_MD):
                current_output_prefix = './md_parallel_sampling_' + str(i)
                md_sampling_process = mp.Process(target=self.__perform_MD_sampling__, args=(current_output_prefix, self.platform_property))
                md_sampling_processes.append(md_sampling_process)

        else:
            raise ValueError('Platform name unsupported')

        for md_sampling_process in md_sampling_processes:
            md_sampling_process.start()
        for md_sampling_process in md_sampling_processes:
            md_sampling_process.join()


class CLICommand:
    __doc__ = "A regular MD simulation protocol to sample the dynamics of protein-ligand complex.\n" + \
    """ Input json template:
    {
        "parameters":
        {
            "unsolvated_prmtop_file_name": "/the/prmtop/file.prmtop",
            "unsolvated_inpcrd_file_name": "/the/inpcrd/file.inpcrd",
            "solvated_prmtop_file_name": "/the/prmtop/file.prmtop",
            "solvated_inpcrd_file_name": "/the/inpcrd/file.inpcrd",
            "ligand_resnames": ["MOL"],
            "n_parallel_MD": 1,
            "n_gpu": 1,
            "platform_name": "CUDA",
            "platform_property":
                {
                    "Precision": "mixed",
                    "DisablePmeStream": "true"
                },
            "gen_temperature": 0.15,
            "temperature": 298.15,
            "pcouple": true,
            "p_ref": 1.0,
            "p_type": "isotropic",
            "p_scale":
                [true, true, true],
            "p_xy_mode": null,
            "p_z_mode": null,
            "p_tens": 0.0,
            "p_freq": 25,
            "fric_coeff": 1.0,
            "mini_nstep": 1000000,
            "mini_tol": 1.0,
            "dt": 0.004,
            "n_step_heating": 50000,
            "n_step_NVT_eq": 250000,
            "n_step_NPT_eq": 5000000,
            "n_step_prod": 50000000,
            "n_step_dcd_heating": 2500,
            "n_step_dcd_NVT_eq": 2500,
            "n_step_dcd_NPT_eq": 2500,
            "n_step_dcd_prod": 2500,
            "nonbonded_method": "LJPME",
            "nonbonded_cutoff": 1.2,
            "switch_distance": 0.0,
            "constraints": "HBonds",
            "restraints_selection_string": "auto",
            "restraints_force_constant": 100.0,
            "rigid_water": true,
            "implicit_solvent": null,
            "implicit_solvent_kappa": null,
            "implicit_solvent_salt_conc": 0.0,
            "solute_dielectric": 1.0,
            "solvent_dielectric": 78.5,
            "use_SASA": false,
            "remove_CM_motion": true,
            "use_HMR": true,
            "hydrogen_mass": 3.024,
            "ewald_error_tolerance": 0.0005,
            "flexible_constraints": true,
            "verbose": false,
            "split_dihedrals": false
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

        md_sampler = MDSampler(configuration)
        md_sampler.run()
