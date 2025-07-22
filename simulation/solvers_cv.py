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

# Own modules
from utility.transform import FDTransform1DA
from utility.processing import ThermalMediumProcessor, StateProcessor
from utility.backends import CloudBackend, LocalBackend, BackendService
from utility.circuits import CircuitGen1DA
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
        self.tf = self.get_transform(kwargs['alpha'], kwargs['tau'], kwargs['dx'],
                                        kwargs['nx'], kwargs['order'], kwargs['bcs'])
        self.data['transform'] = self.tf.get_dict()
        self.logger.info('Calculation completed.')

        # Set medium
        self.md = self.get_medium_processor(kwargs['alpha'], kwargs['tau'])
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
        assert self.kwargs['nx'] == len(self.kwargs['tau']), 'length of tau must be equal to nx'
        assert np.all(np.array(self.kwargs['alpha']) > 0), 'alpha must be positive'
        assert np.all(np.array(self.kwargs['tau']) > 0), 'tau must be positive'
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

    def get_transform(self, alpha: np.ndarray, tau: np.ndarray, dx: float, nx: int,
                      order: int, bcs: Dict[str, str]) -> FDTransform1DA:
        """
        Initializes and returns a finite difference transformation object for 1D analysis.

        This method creates an instance of the FDTransform1DA class, which is used for
        transforming the state space based on the provided medium parameters.

        Args:
            alpha (np.ndarray): An array of µ values representing the medium shear modulus.
            tau (np.ndarray): An array of ρ values representing the medium density.
            dx (float): The spatial step size.
            nx (int): The number of spatial steps.
            order (int): The order of the finite difference scheme.
            bcs (Dict[str, str]): A dictionary specifying the boundary conditions
                                  with keys 'left' and 'right'.

        Returns:
            FDTransform1DA: An instance of FDTransform1DA initialized with the given parameters.
        """
        # Initialize transform
        return FDTransform1DA(alpha, tau, dx, nx, order, bcs)

    def get_medium_processor(self, alpha: np.ndarray, tau: np.ndarray) -> ThermalMediumProcessor:
        """
        Initializes and configures a medium processor for the simulation.

        This method creates an instance of ThermalMediumProcessor, setting it up with the 
        specified thermal diffusivity (alpha) and ρ (tau) values representing the medium's properties. 

        Args:
            alpha (np.ndarray): An array of µ values, representing the medium shear modulus.
            tau (np.ndarray): An array of ρ values, representing the medium density.

        Returns:
            ThermalMediumProcessor: An initialized and configured medium processor object.
        """

        # Initialize medium processor
        md =  ThermalMediumProcessor(len(alpha), len(tau))

        # Set medium parameters
        md.set_alpha(alpha)
        md.set_tau(tau)

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
        self.st.forward_state(0, self.tf.sqrt_m)
        self.logger.info('Initial state forward-transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the ODE solver and processes the results.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """
        self.logger.info('Solving ODE.')
        self.st.states = solve_ivp(lambda t, y: self.tf.q @ y, (0, self.times[-1]),
                                     self.st.get_state(0), t_eval=self.times,
                                     method='Radau').y.T
        self.logger.info('ODE solved.')

        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m)
         for i in range(len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DLocal(Solver1D):
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
            self.tf.h,
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
        observables = list(product("ZX", repeat=int(np.log2(self.tf.h.shape[0]))))
        self.logger.debug(f'Observables: {observables}')
        states_raw = tomo.run_tomography(result_groups, observables, self.times[1:])
        self.logger.info('Tomography completed.')

        self.st.states = np.real(parallel_transport(states_raw, self.st.get_state(0)))
        self.logger.info('State polarization corrected.')
        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t)
         for i in range(1, len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

