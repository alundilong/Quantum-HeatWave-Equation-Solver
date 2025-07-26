#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for plotting functions.}

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
# Other modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# -------- SETTINGS --------
#plt.rcParams['font.family'] = 'Times New Roman'

# -------- CONSTANTS --------
ENUMS = ['a.)', 'b.)', 'c.)', 'd.)', 'e.)', 'f.)']
PATH_MULTIPLOT = './figures/forward_sim.png'
PATH_CIRCUIT = './figures/circuit.png'

# -------- FUNCTIONS --------
def plot_multi(data, idx, shots=None, colors=['black', 'red', 'blue', 'green']):
    """
    Plotting function for plotting multiple solvers at different time steps.
    
    Args:
        data (list): List of data dictionaries.
        idx (list): List of time indices to plot.
    
    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
    """
    assert len(idx) == 5, "Please provide 5 time indices as idx for plotting"
    assert max(idx) < len(data[0]['times']), "idx out of range"

    # Read data (Assuming local = noise free, cloud = quantum computer !) TODO: Fix this
    settings = data[0]['settings']
    times, tau, alpha = data[0]['times'], settings['tau'], settings['alpha']
    bcs, nx = settings['bcs'], settings['nx']
    tau_lim = (0, 6)
    alpha_lim = (0, 3)
    field_lim = (-1.2, 1.2)
    digits = 4

    solv_list = []
    for i, d in enumerate(data):
        apx = f'({shots[i]} Shots)' if shots and shots[i] else ''
        if d['settings']['solver'] == 'ode':
            solv_list.append(f'ODE Solver {apx}')
        elif d['settings']['solver'] == 'exp':
            solv_list.append(f'Matrix Exponential Solver {apx}')
        elif d['settings']['solver'] == 'exp_emb':
            solv_list.append(f'Emb Matrix Exponential Solver {apx}')
        elif d['settings']['solver'] == 'lindblad':
            solv_list.append(f'Lindblad Solver {apx}')
        elif d['settings']['solver'] == 'local' and not d['settings']['backend']['fake']:
            solv_list.append(f'Quantum Simulator {apx}')
        elif d['settings']['solver'] == 'local' and d['settings']['backend']['fake']:
            solv_list.append(f'Quantum Simulator {apx}')
        elif (d['settings']['solver'] == 'cloud' and
              d['settings']['backend']['backend'] == 'ibmq_qasm_simulator'):
            solv_list.append(f'IBM QASM Simulator {apx}')
        elif (d['settings']['solver'] == 'cloud' and
              d['settings']['backend']['backend'] != 'ibmq_qasm_simulator'):
            solv_list.append(f'Quantum Computer {apx}')

    # Prepare data with boundary conditions
    data_fields = []
    for d in data:
        field = np.zeros((len(times), nx + 2))
        field[:, 1:-1] = d['field']['u']
        if bcs['left'] == 'NBC':
            field[:, 0] = d['field']['u'][:, 0]
        if bcs['right'] == 'NBC':
            field[:, -1] = d['field']['u'][:, -1]
        data_fields.append(field)

    # Prepare medium
    medium_fields = []
    for d in [alpha, tau]:
        field = np.zeros(nx+2)
        field[1:-1] = d if len(d) == nx else d[:-1]
        field[0] = d[0]
        field[-1] = d[-1]
        medium_fields.append(field)

    # Plot multiplot
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    ax_tau_alpha = axes[0, 0]
    ax_alpha = ax_tau_alpha.twinx()

    ax_tau_alpha.text(0.05, 0.9, ENUMS[0], transform=ax_tau_alpha.transAxes,
                   fontsize=14)

    ax_tau_alpha.plot(np.arange(nx+2), medium_fields[1], color='blue', label='$\\tau$')
    ax_tau_alpha.set_ylabel('$\\tau$ [s]', color='blue')
    ax_tau_alpha.tick_params(axis='y', labelcolor='blue')
    ax_tau_alpha.set_ylim(*tau_lim)

    ax_alpha.plot(np.arange(nx+2), medium_fields[0], color='red', label='$\\alpha$')
    ax_alpha.set_ylabel('$\\alpha$ [$m^2$/s]', color='red')
    ax_alpha.tick_params(axis='y', labelcolor='red')
    ax_alpha.set_ylim(*alpha_lim)

    lines, labels = ax_tau_alpha.get_legend_handles_labels()
    lines2, labels2 = ax_alpha.get_legend_handles_labels()
    ax_alpha.legend(lines + lines2, labels + labels2, loc='lower right')
    ax_alpha.set_xlabel('x [m]')
    ax_tau_alpha.set_xlabel('x [m]')

    for i, t in enumerate(idx):
        ax = axes[(i+1) // 3, (i+1) % 3]
        ax.text(0.05, 0.9, ENUMS[i+1], transform=ax.transAxes,
                fontsize=14)
        for j, field in enumerate(data_fields):
            style = 'scatterplot' if j == 0 else 'lineplot'
            getattr(sns, style)(x=np.arange(nx+2), y=field[t], ax=ax,
                                label=solv_list[j],
                                color=colors[j])
        ax.set_title(f"t = {times[t]:.{digits}e} s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("u [K]")
        ax.set_ylim(*field_lim)
        if i == 0:
            ax.legend(loc='lower right')
        else:
            ax.legend([],[], frameon=False)

    plt.tight_layout()
    plt.savefig(PATH_MULTIPLOT, dpi=300)
    plt.close(fig)
    return fig

def plot_medium(mu, tau, **kwargs):
    """
    Plotting function for plotting the medium.
    
    Args:
        alpha (list): List of alpha values.
        tau (list): List of tau values.
        
    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
    """
    _ = kwargs
    nx = len(tau)
    tau_lim = (0, 6)
    alpha_lim = (0, 3)

    # Prepare medium
    medium_fields = []
    for d in [alpha, tau]:
        field = np.zeros(nx+2)
        field[1:-1] = d if len(d) == nx else d[:-1]
        field[0] = d[0]
        field[-1] = d[-1]
        medium_fields.append(field)

    fig, axes = plt.subplots(1, figsize=(6, 3))
    ax_tau_alpha = axes
    ax_alpha = ax_tau_alpha.twinx()

    ax_tau_alpha.plot(np.arange(nx+2), medium_fields[1], color='blue', label='$\\tau$')
    ax_tau_alpha.set_ylabel('$\\tau$ [s]', color='blue')
    ax_tau_alpha.tick_params(axis='y', labelcolor='blue')
    ax_tau_alpha.set_ylim(*tau_lim)

    ax_alpha.plot(np.arange(nx+2), medium_fields[0], color='red', label='$\\mu$')
    ax_alpha.set_ylabel('$\\alpha$ [$m^2$/s]', color='red')
    ax_alpha.tick_params(axis='y', labelcolor='red')
    ax_alpha.set_ylim(*alpha_lim)

    lines, labels = ax_tau_alpha.get_legend_handles_labels()
    lines2, labels2 = ax_alpha.get_legend_handles_labels()
    ax_alpha.legend(lines + lines2, labels + labels2, loc='lower right')
    ax_alpha.set_xlabel('x [m]')
    ax_tau_alpha.set_xlabel('x [m]')

    plt.tight_layout()
    return fig

def plot_initial(u, v, bcs, **kwargs):
    """
    Plotting function for plotting the initial state.
    
    Args:
        u (list): List of u values.
        v (list): List of v values.
        bcs (dict): Boundary conditions.
    
    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
    """
    _ = kwargs
    nx = len(u)
    u_lim = (-1, 1)
    v_lim = (-1000, 1000)

    # Prepare initial state
    state_fields = []
    for d in [u, v]:
        field = np.zeros(nx+2)
        field[1:-1] = d if len(d) == nx else d[:-1]
        if bcs['left'] == 'NBC':
            field[0] = d[0]
        if bcs['right'] == 'NBC':
            field[-1] = d[-1]
        state_fields.append(field)

    fig, axes = plt.subplots(1, figsize=(6, 3))
    ax_u_v = axes
    ax_v = ax_u_v.twinx()

    ax_u_v.plot(np.arange(nx+2), state_fields[0], color='blue', label='u')
    ax_u_v.set_ylabel('u [$\\mu$m]', color='black')
    ax_u_v.tick_params(axis='y', labelcolor='black')
    ax_u_v.set_ylim(*u_lim)

    ax_v.plot(np.arange(nx+2), state_fields[1], color='red', label='v')
    ax_v.set_ylabel('v [$\\mu$m / s$^2$]', color='black')
    ax_v.tick_params(axis='y', labelcolor='black')
    ax_v.set_ylim(*v_lim)

    lines, labels = ax_u_v.get_legend_handles_labels()
    lines2, labels2 = ax_v.get_legend_handles_labels()
    ax_v.legend(lines + lines2, labels + labels2, loc='lower right')
    ax_v.set_xlabel('x [m]')
    ax_u_v.set_xlabel('x [m]')

    plt.tight_layout()
    return fig

def plot_error(data1, data2, **kwargs):
    """
    Plotting function for plotting the error between two data sets.
    
    Args:
        data1 (list): List of data values.
        data2 (list): List of data values.
        
    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
    """
    _ = kwargs
    e_lim = (1e-4, 1e+1)
    data1 = data1['field']['u']
    data2 = data2['field']['u']

    l2_errors = np.linalg.norm(data1 - data2, axis=1) / np.linalg.norm(data2, axis=1)
    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.plot(l2_errors, color='blue')
    ax.axhline(np.mean(l2_errors), color='black', linestyle='--')
    ax.text(0.15, 0.9, f"Mean error: {np.mean(l2_errors):.2e}",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Relative L2 error')
    ax.set_yscale('log')
    ax.set_ylim(*e_lim)
    ax.set_title('Relative L2 error over time')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_circuit(solver, **kwargs):
    """
    Plotting function for plotting the circuit.
    
    Args:
        solver (Solver1D): Solver object.
        
    Returns:
        fig (matplotlib.pyplot.figure): Figure of the Quantum Circuit.
    """
    g, i = kwargs.get("group", 0), kwargs.get("idx", 0)
    circuit = solver.circuit_groups[g][i]
    fig = circuit.draw(output='mpl')
    fig.suptitle(f'Time Evolution Quantum Circuit (Group {g}, Index {i})')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(PATH_CIRCUIT, dpi=300)
    return fig

# TODO: Fix plot returns/plottings
