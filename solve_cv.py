#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Main module for running the quantum 1D elastic wave equation solver.}

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
# Own modules
from simulation.experiment_cv import ForwardExperiment1D
from utility.distributions import (spike, ricker, gaussian, raised_cosine,
                                    sinc, homogeneous, exponential, polynomial)

# -------- FUNCTIONS --------
def main() -> None:
    """
    Runs the quantum 1D CV equation solver.
    """

    # Create experiment
    experiment = ForwardExperiment1D(verbose=2)

    # Set Experiment Parameters
    n = 3
    nx = 2**n - 1
    alpha = 1.11e-4
    tau = 1e-12
    L = 7e-9
    dx = L/nx
    dt = 1e-10
    nt = 199

    t0 = L*L/alpha
    eps = tau*alpha/L

    experiment.logger.info(f'dimensionless t0 = {t0:.2e}')
    experiment.logger.info(f'dimensionless eps = {eps:.2e}')

    alpha = 1
    tau = eps
    dx = 1
    dt = 0.001
    parameters = {
        'dx': dx,                                             # Grid spacing
        'nx': nx,                                               # Number of grid points
        'dt': dt,                                           # Time stepping
        'nt': nt,                                               # Number of time steps
        'order': 1,                                             # Finite-difference order
        'bcs': {'left': 'DBC', 'right': 'DBC'},                 # Boundary conditions
        'alpha': raised_cosine(alpha, nx+1, nx, 6, alpha),  # thermal diffusivity distribution
        'tau': raised_cosine(tau, nx, nx-1, 6, tau),          # lagging distribution
        'u': spike(1, nx, nx//2+1),                             # Initial temperature
        'v': homogeneous(0, nx),                                # Initial temperature change rate
        'backend': {
            'synthesis': 'MatrixExponential',           # Time Evolution Synthesis Method
            'batch_size': 100,                                  # Circuit Batch Size
            'fitter': 'cvxpy_gaussian',                         # State Tomography fitter
            'backend': 'ibmq_qasm_simulator',                   # Cloud backend name
            'shots': 1000,                                      # Number of circuit samples
            'optimization': 3,                                  # Circuit optimization level
            'resilience': 1,                                    # Circuit resilience level
            'seed': 0,                                          # Transpilation seed
            'local_transpilation': False,                       # Local transpilation
            'method': 'statevector',                            # Classical simulation method
            'fake': None,                                       # Fake backend model (Currently not supported)
            }
        }

    # Define solvers
    solvers=['ode','exp','local']
    solvers_idx=[0,1,2]

    #solvers=['ode','exp']
    #solvers_idx=[0,1]
    for s in solvers:
        experiment.add_solver(s, **parameters)

    # Run experiment
    _ = experiment.run()

    dstep = nt//5

    plotparameters = {
            'idx': [
                nt-1-4*dstep,
                nt-1-3*dstep,
                nt-1-2*dstep,
                nt-1-dstep,
                nt-1]
            }
    
    experiment.plot(mode='multi',solvers=solvers_idx,**plotparameters)

# -------- SCRIPT --------
if __name__ == '__main__':
    main()
