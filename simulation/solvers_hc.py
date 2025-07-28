#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for implementing different wave equation solvers.}

{
    Copyright (C) [2023]  [Malte Schade]

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
}
"""

# -------- IMPORTS --------
# Built-in modules
from typing import Dict, List, Any
from time import sleep
from itertools import product
import json

# Other modules
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# Own modules
from utility.transform_hc import FDTransform1DA
from utility.processing import ThermalMediumHCProcessor, StateProcessor
from utility.backends import CloudBackend, LocalBackend, BackendService
from utility.circuits_hc import CircuitGen1DA
from utility.tomography import TomographyReal, parallel_transport

# -------- CLASSES --------
class Solver1D:
    """
    A class for solving 1D elastic wave forward problems using quantum computing methods.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.

    Attributes:
        logger (object): Logger for logging information.
        kwargs (dict): Dictionary of keyword arguments.
        data (dict): Dictionary to store various configurations and results.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        self.base_data = base_data
        self.logger = logger
        self.kwargs = kwargs
        self.idx = kwargs['idx']
        self.data = {}

        # Check parameters
        self.check_kwargs()
        self.logger.info('Parameters checked for validity.')

        # Save kwargs
        self.data['settings'] = kwargs

        # Set time steps
        self.times = self.get_times(kwargs['nt'], kwargs['dt'])
        self.data['times'] = self.times
        self.logger.info(f'Solving for {self.kwargs["nt"]} time steps.')
        self.logger.debug(f'Times: {self.data["times"]}')

        # Set transform
        self.logger.info('Calculating Transformation and Hamiltonian.')
        self.tf = self.get_transform(kwargs['alpha'],  kwargs['dx'],
                                        kwargs['nx'], kwargs['order'], kwargs['bcs'])
        self.logger.info('---S-REPORT---'*3)
        self.logger.info(f'norm of D: {np.linalg.norm(self.tf.d):.2e}')
        self.logger.info(f'norm of T: {np.linalg.norm(self.tf.t):.2e}')
        self.logger.info(f'norm of T^{-1}: {np.linalg.norm(self.tf.inv_t):.2e}')
        self.logger.info(f'norm of H_tilde: {np.linalg.norm(self.tf.h_tilde):.2e}')
        self.logger.info(f'norm of H_herm: {np.linalg.norm(self.tf.h_herm):.2e}')
        self.logger.info(f'norm of H_non_herm: {np.linalg.norm(self.tf.h_non_herm):.2e}')
        self.logger.info(f'norm of H_embed: {np.linalg.norm(self.tf.h_embed):.2e}')
        self.logger.info('---E-REPORT---'*3)
        self.data['transform'] = self.tf.get_dict()
        self.logger.info('Calculation completed.')

        # Set medium
        self.md = self.get_medium_processor(kwargs['alpha'])
        self.medium = self.md.get_medium()
        self.data['medium'] = self.md.get_dict()
        self.logger.info('Medium initialized.')

    def check_kwargs(self):
        """
        Validates the keyword arguments provided to the solver.

        Raises:
            AssertionError: If any of the conditions for the arguments are not met.
        """

        assert self.kwargs['nx'] > 0, 'nx must be greater than zero'
        assert np.log2(self.kwargs['nx']+1) % 1 == 0, 'nx must be a power of two minus one'
        assert self.kwargs['nx'] == len(self.kwargs['alpha'])-1,'length of alpha \
            must be one more than nx'
        assert np.all(np.array(self.kwargs['alpha']) > 0), 'alpha must be positive'
        assert self.kwargs['nx'] == len(self.kwargs['u']), 'length of u must be equal to nx'
        assert self.kwargs['nx'] == len(self.kwargs['v']), 'length of v must be equal to nx'
        assert self.kwargs['nt'] > 0, 'nt must be greater than zero'
        assert self.kwargs['dt'] > 0, 'dt must be greater than zero'
        assert self.kwargs['dx'] > 0, 'dx must be greater than zero'
        assert self.kwargs['order'] in [1,2,3,4], "Order must be in [1,2,3,4]"
        assert self.kwargs['bcs']['left'] in ['DBC', 'NBC'], "Left boundary condition \
            must be DBC or NBC"
        assert self.kwargs['bcs']['right'] in ['DBC', 'NBC'], "Right boundary condition \
            must be DBC or NBC"

    def get_times(self, nt, dt) -> np.ndarray:
        """
        Sets up the time steps for the solver based on the 'nt' and 'dt' arguments.
        
        Args:
            nt (int): The number of time steps.
            dt (float): The time step size.
            
        Returns:
            np.ndarray: An array of time steps.
        """

        return np.arange(nt)*dt

    def get_transform(self, alpha: np.ndarray, dx: float, nx: int,
                      order: int, bcs: Dict[str, str]) -> FDTransform1DA:
        """
        Initializes and returns a finite difference transformation object for 1D analysis.

        This method creates an instance of the FDTransform1DA class, which is used for
        transforming the state space based on the provided medium parameters.

        Args:
            alpha (np.ndarray): An array of alpha values representing the medium thermal diffusivity.
            dx (float): The spatial step size.
            nx (int): The number of spatial steps.
            order (int): The order of the finite difference scheme.
            bcs (Dict[str, str]): A dictionary specifying the boundary conditions
                                  with keys 'left' and 'right'.

        Returns:
            FDTransform1DA: An instance of FDTransform1DA initialized with the given parameters.
        """
        # Initialize transform
        return FDTransform1DA(alpha, dx, nx, order, bcs)

    def get_medium_processor(self, alpha: np.ndarray) -> ThermalMediumHCProcessor:
        """
        Initializes and configures a medium processor for the simulation.

        This method creates an instance of ThermalMediumProcessor, setting it up with the 
        specified thermal diffusivity (alpha) representing the medium's properties. 

        Args:
            alpha (np.ndarray): An array of µ values, representing the medium shear modulus.

        Returns:
            ThermalMediumHCProcessor: An initialized and configured medium processor object.
        """

        # Initialize medium processor
        md =  ThermalMediumHCProcessor(len(alpha))

        # Set medium parameters
        md.set_alpha(alpha)

        return md

class Solver1DODE(Solver1D):
    """
    A subclass of Solver1D for solving with a classical
        Ordinary Differential Equations (ODEs) solver.

    Inherits from Solver1D and adds specific methods for handling ODEs.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=0)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.i)
        self.logger.info('Initial state forward-transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the ODE solver and processes the results.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """
        self.logger.info('Solving ODE.')
        self.st.states = solve_ivp(lambda t, y: self.tf.h_emb @ y, (0, self.times[-1]),
                self.st.get_state(0), 
                t_eval=self.times,
                method='Radau').y.T
        self.logger.info('ODE solved.')

        _ = [self.st.inverse_state(i, self.tf.i)
         for i in range(len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data


class Solver1DEXP(Solver1D):
    """
    A subclass of Solver1D for solving with a classical
        Matrix exponential time evolution solver.

    Inherits from Solver1D.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.i)
        self.logger.info('Initial state forward-transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the matrix exponential solver and processes the results.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """
        self.logger.info('Solving matrix exponential.')
        initial_state = self.st.get_state(0)
        self.st.states = np.array([
            scipy.linalg.expm(time * -1j * self.tf.h_emb) @ initial_state
            for time in self.times])
        self.logger.info(f'Shape of st.states: {self.st.states.shape}')
        self.logger.info('Matrix exponential solved.')

        _ = [self.st.inverse_state(i, self.tf.i)
         for i in range(len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DImaginary(Solver1D):
    """
    A subclass of Solver1D for solving with a classical
        Imaginary time evolution solver.

    Inherits from Solver1D.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.i)
        self.logger.info('Initial state forward-transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the matrix exponential solver and processes the results.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """
        self.logger.info('Imaginary Solving.')
        initial_state = self.st.get_state(0)
        self.st.states = np.array([
            np.real(scipy.linalg.expm(time * -1j * self.tf.m) @ initial_state)
            for time in self.times])
        self.logger.info(f'Shape of st.states: {self.st.states.shape}')
        self.logger.info('Imaginary solved.')

        _ = [self.st.inverse_state(i, self.tf.i)
         for i in range(len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DImaginaryQ(Solver1D):
    """
    A subclass of Solver1D for solving with a classical
        Matrix exponential time evolution solver.

    Inherits from Solver1D.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.factor = 1.0
        self.st.forward_state(0, self.tf.t @ self.tf.sqrt_m, factor=self.factor)
        self.logger.info('Initial state forward-transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the matrix exponential solver and processes the results.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """
        self.logger.info('Solving matrix exponential.')
        initial_state = self.st.get_state(0,factor = self.factor)
        N = len(initial_state)
        Psi_0 = np.zeros(2 * N, dtype=complex)
        Psi_0[:N] = initial_state  # Upper half is physical state
        self.st.states = np.array([
            np.real(scipy.linalg.expm(time * -1j * self.tf.h_embed) @ Psi_0)
            for time in self.times])[:,:N]
        self.logger.info(f'Shape of st.states: {self.st.states.shape}')
        self.logger.info('Matrix exponential solved.')

        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t, factor=self.factor)
         for i in range(len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DEmb(Solver1D):
    """
    A subclass of Solver1D for solving with a classical
        Matrix exponential time evolution solver.

    Inherits from Solver1D.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.i)
        self.logger.info('Initial state forward-transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the Embedding solver and processes the results.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """
        self.logger.info('Solving Embedding.')
        initial_state = self.st.get_state(0)
        self.st.states = np.array([
            np.real(scipy.linalg.expm(time * -1j * self.tf.h_emb) @ Psi_0)
            for time in self.times])
        self.logger.info(f'Shape of st.states: {self.st.states.shape}')
        self.logger.info('Embedding solved.')

        _ = [self.st.inverse_state(i, self.tf.i)
         for i in range(len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DEmbQ(Solver1D):
    """
    A subclass of Solver1D for local quantum computing simulations.

    Inherits from Solver1D and adds specific methods for handling local simulations.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.t @ self.tf.sqrt_m)
        self.circuit_groups = []
        self.logger.info('Initial state transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the local solver, including quantum circuit generation, execution, and tomography.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """

        self.logger.info('Initializing backend.')
        backend = LocalBackend(self.logger,
                               backend=None,
                               fake=self.kwargs['backend']['fake'],
                               method=self.kwargs['backend']['method'],
                               seed=self.kwargs['backend']['seed'],
                               shots=self.kwargs['backend']['shots'],
                               optimization=self.kwargs['backend']['optimization'],
                               resilience=self.kwargs['backend']['resilience'],
                               local_transpilation = self.kwargs['backend']['local_transpilation'],
                               max_parallel_experiments=0)
        sampler, _ = backend.get_sampler()
        self.logger.info('Backend initialized.')

        self.logger.info('Generating circuits.')
        circuit_gen = CircuitGen1DA(self.logger, backend.fake_backend)
        self.circuit_groups = circuit_gen.tomography_circuits(
            self.st.get_state(0),
            self.tf.h_emb,
            self.times[1:],
            self.kwargs['backend']['synthesis'],
            self.kwargs['backend']['batch_size'],
            self.kwargs['backend']['optimization'],
            self.kwargs['backend']['seed'],
            self.kwargs['backend']['local_transpilation'])

        self.logger.info('Submitting jobs to backend.')
        jobs = [sampler.run(circuits) for circuits in self.circuit_groups]
        self.logger.info('Jobs submitted.')
        _wait_for_completion(jobs, self.logger)
        result_groups = [job.result() for job in jobs]
        self.logger.info('Jobs completed.')

        self.logger.info('Running tomography.')
        tomo = TomographyReal(self.logger, self.kwargs['backend']['fitter'])
        observables = list(product("ZX", repeat=int(np.log2(self.tf.h_emb.shape[0]))))
        self.logger.debug(f'Observables: {observables}')
        states_raw = tomo.run_tomography(result_groups, observables, self.times[1:])
        self.logger.info('Tomography completed.')

        self.st.states = np.real(parallel_transport(states_raw, self.st.get_state(0)))
        self.logger.info('State polarization corrected.')
        _ = [self.st.inverse_state(i, self.tf.i)
         for i in range(1, len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DSplit(Solver1D):
    """
    A subclass of Solver1D for solving with a
        Split time evolution solver.

    Inherits from Solver1D.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.i)
        self.logger.info('Initial state forward-transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the Split solver and processes the results.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """
        self.logger.info('Solving Hermitian Part.')
        initial_state = self.st.get_state(0)
        self.st.states = np.array([
            scipy.linalg.expm(time * -1j * self.tf.h_herm) @ initial_state
            for time in self.times])
        #self.st.states = np.array([v / np.linalg.norm(v) for v in self.st.states])
        self.logger.info(f'Shape of st.states: {self.st.states.shape}')
        self.logger.info('Hermian Part solved.')

        for i, time in enumerate(self.times):
            self.st.states[i] = scipy.linalg.expm(time*-1j*self.tf.h_non_herm) @ self.st.states[i]

        _ = [self.st.inverse_state(i, self.tf.i)
         for i in range(len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DSplitQ(Solver1D):
    """
    A subclass of Solver1D for local quantum computing simulations.

    Inherits from Solver1D and adds specific methods for handling local simulations.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, base_data: object, logger: object, **kwargs) -> None:
        super().__init__(base_data, logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.t @ self.tf.sqrt_m)
        self.circuit_groups = []
        self.logger.info('Initial state transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the local solver, including quantum circuit generation, execution, and tomography.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """

        self.logger.info('Initializing backend.')
        backend = LocalBackend(self.logger,
                               backend=None,
                               fake=self.kwargs['backend']['fake'],
                               method=self.kwargs['backend']['method'],
                               seed=self.kwargs['backend']['seed'],
                               shots=self.kwargs['backend']['shots'],
                               optimization=self.kwargs['backend']['optimization'],
                               resilience=self.kwargs['backend']['resilience'],
                               local_transpilation = self.kwargs['backend']['local_transpilation'],
                               max_parallel_experiments=0)
        sampler, _ = backend.get_sampler()
        self.logger.info('Backend initialized.')

        initial_state = self.st.get_state(0)

        self.logger.info(f'initial_state Norm: {np.linalg.norm(initial_state):.2e}')

        self.logger.info('Generating circuits.')
        circuit_gen = CircuitGen1DA(self.logger, backend.fake_backend)
        self.circuit_groups = circuit_gen.tomography_circuits(
            initial_state,
            self.tf.h_herm,
            self.times[1:],
            self.kwargs['backend']['synthesis'],
            self.kwargs['backend']['batch_size'],
            self.kwargs['backend']['optimization'],
            self.kwargs['backend']['seed'],
            self.kwargs['backend']['local_transpilation'])

        self.logger.info('Submitting jobs to backend.')
        jobs = [sampler.run(circuits) for circuits in self.circuit_groups]
        self.logger.info('Jobs submitted.')
        _wait_for_completion(jobs, self.logger)
        result_groups = [job.result() for job in jobs]
        self.logger.info('Jobs completed.')

        self.logger.info('Running tomography.')
        tomo = TomographyReal(self.logger, self.kwargs['backend']['fitter'])
        observables = list(product("ZX", repeat=int(np.log2(self.tf.h_herm.shape[0]))))
        self.logger.debug(f'Observables: {observables}')
        states_raw = tomo.run_tomography(result_groups, observables, self.times[1:])
        self.logger.info('Tomography completed.')

        self.st.states = np.real(parallel_transport(states_raw, initial_state))

        for i, time in enumerate(self.times):
            self.st.states[i] = scipy.linalg.expm(time*-1j*self.tf.h_non_herm) @ self.st.states[i]

        self.logger.info('State polarization corrected.')
        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t)
         for i in range(1, len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

# -------- FUNCTIONS --------
def _wait_for_completion(jobs: List[object], logger: object, sleep_time: float = 10) -> None:
    """
    Waits for a list of jobs to complete.

    Args:
        jobs (List[object]): A list of jobs to wait for.
        logger (object): A logger to record the status of the jobs.
    """

    all_completed = False
    while not all_completed:
        sleep(sleep_time)
        status = [job.status().name for job in jobs]
        logger.debug(f"Jobs status: {status}")
        if 'ERROR' in status:
            logger.debug(f"Fatal error occurred: {[job.status() for job in jobs]}")
            raise RuntimeError('Runtime error in simulating the quantum circuit.\
                This might be a problem of your qiskit installation.')
        completed = [job.status().name == 'DONE' for job in jobs]
        logger.info(f"Jobs completed: {sum(completed)} | {len(jobs)}")
        all_completed = all(completed)

def _save_jobids(job_ids: List[str], path: str, indent: int = 4,
                 encoding: str = 'utf8') -> None:
    """
    Saves a list of job IDs to a JSON file.

    Args:
        job_ids (List[str]): A list of job IDs to save.
        path (str): The path to save the job IDs to.
    """

    with open(path, 'w', encoding=encoding) as f:
        json.dump(job_ids, f, indent=indent)
