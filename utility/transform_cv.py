#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for calculating the transforms and hamiltonian
for the 1D elastic wave equation solver.}

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
from typing import Dict, List

# Other modules
import numpy as np
from scipy.linalg import expm
from qiskit_dynamics.solvers import Solver
from qiskit.quantum_info import Operator

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli, SparsePauliOp

# -------- CONSTANTS --------
FORWARD_FD_COEFF: Dict[int, List[float]] = {
        1: [-1, 1],
        2: [-3/2, 2, -1/2],
        3: [-11/6, 3, -3/2, 1/3],
        4: [-25/12, 4, -3, 4/3, -1/4]
    }

class LindbladFromNonHermitian:
    def __init__(self, H_tilde: np.ndarray):
        """
        Parameters:
            H_tilde : complex (n, n) ndarray, non-Hermitian generator
        """
        if not np.iscomplexobj(H_tilde):
            raise ValueError("H_tilde must be complex-valued")
        if H_tilde.shape[0] != H_tilde.shape[1]:
            raise ValueError("H_tilde must be square")

        self.n = H_tilde.shape[0]
        self.H_tilde = H_tilde

        # Split into Hermitian and anti-Hermitian parts
        self.H = 0.5 * (H_tilde + H_tilde.conj().T)
        self.A = 0.5j * (H_tilde - H_tilde.conj().T)

        # Validate and construct collapse operators
        self.C_ops = self._construct_collapse_operators(self.A)

        # Create Solver
        self.solver = Solver(
            static_hamiltonian=self.H,
            static_dissipators=self.C_ops,
        )

    def _construct_collapse_operators(self, A: np.ndarray):
        """
        Try to compute collapse operators such that A = 0.5 * sum Cj† Cj
        """
        # Check Hermiticity
        if not np.allclose(A, A.conj().T, atol=1e-12):
            raise ValueError("A is not Hermitian. Cannot construct collapse operators.")

        # Ensure A is positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(A)
        if np.any(eigvals < -1e-12):
            raise ValueError("Anti-Hermitian part is not positive semi-definite. Cannot simulate via Lindblad.")

        # Form collapse operators: C_j = sqrt(2λ_j) v_j
        C_ops = []
        for lam, vec in zip(eigvals, eigvecs.T):
            if lam > 1e-12:
                Cj = np.sqrt(2 * lam) * vec[:, np.newaxis]  # shape (n, 1)
                C_ops.append(Operator(Cj @ Cj.conj().T))     # shape (n, n)

        return C_ops

    def simulate(self, psi0: np.ndarray, t_span: np.ndarray, return_density: bool = False,
                 check_purity: bool = False, check_fidelity: bool = False) -> list:
        """
        Simulate the Lindblad evolution and return quantum states at specified times.
        
        Parameters:
        psi0            : (n,) ndarray, initial state vector
        t_span          : array-like, list of time points
        return_density  : if True, return list of density matrices
                          if False, return list of dominant pure state vectors
        check_purity    : if True, print Tr(rho^2) at each time step
        check_fidelity  : if True, print fidelity with exp(-i H_eff t) @ psi0 at each step
        
        Returns:
        List of states (either density matrices or state vectors) at each time in t_span
        """
        if psi0.shape != (self.n,):
            raise ValueError(f"psi0 must be of shape ({self.n},)")
    
        result = self.solver.solve(
            t_span=[t_span[0], t_span[-1]],
            y0=np.outer(psi0, psi0.conj()),
            t_eval=t_span,
            method="RK45"  # Ensures t_eval is respected
        )
    
        rho_list = result.y  # Shape: (len(t_span), n, n)
    
        if return_density:
            if check_purity:
                print("Purity check:")
                for i, rho in enumerate(rho_list):
                    purity = np.trace(rho @ rho).real
                    print(f"t = {t_span[i]:.4f}: Tr(rho^2) = {purity:.6f}")
            return rho_list
    
        # Extract dominant eigenvector at each time point
        psi_list = []
        for i, rho in enumerate(rho_list):
            eigvals, eigvecs = np.linalg.eigh(rho)
            psi = eigvecs[:, -1]  # Eigenvector with largest eigenvalue
            psi_list.append(psi)
    
            if check_purity:
                purity = np.trace(rho @ rho).real
                print(f"t = {t_span[i]:.4f}: Tr(rho^2) = {purity:.6f}")
    
            if check_fidelity:
                # Use exp(-i H_eff t) @ psi0 to compare
                H_eff = self.H - 0.5j * sum(Cj.data.conj().T @ Cj.data for Cj in self.C_ops)
                exact_psi = expm(-1j * H_eff * t_span[i]) @ psi0
                fid = np.abs(np.vdot(exact_psi, psi))**2
                print(f"t = {t_span[i]:.4f}: Fidelity = {fid:.6f}")
    
        return psi_list

class LCU_NonHermitianSimulator:
    def __init__(self, H_tilde: np.ndarray):
        """
        Initialize the simulator with a non-Hermitian matrix.
        Decompose it as a linear combination of unitaries.
        """
        self.H_tilde = H_tilde
        self.dim = H_tilde.shape[0]
        self.num_qubits = int(np.ceil(np.log2(self.dim)))

        # Normalize H_tilde to avoid overflow
        norm = np.linalg.norm(H_tilde)
        self.H_norm = H_tilde / norm
        self.norm_factor = norm

        # H_tilde is Hermitian or not, does not matter here
        pauli_decomp = SparsePauliOp.from_operator(Operator(self.H_norm))
        
        # Get terms and coefficients
        self.terms = [Operator(p) for p in pauli_decomp.paulis]
        self.coeffs = pauli_decomp.coeffs
        print(f'Number of terms: {len(self.terms)}')

    def build_lcu_circuit(self, t: float) -> QuantumCircuit:
        """
        Build a quantum circuit that implements the LCU step using ancilla superposition and controlled unitaries.
        Assumes self.terms is a list of unitary matrices (e.g., numpy arrays or Qiskit Operator),
        and self.coeffs are real non-negative weights.
        """
        m = len(self.coeffs)
        ancilla = int(np.ceil(np.log2(m)))
        total_qubits = self.num_qubits + ancilla
        qc = QuantumCircuit(total_qubits)
    
        # Prepare ancilla superposition weighted by sqrt(coeff / total)
        norm = sum(self.coeffs)
        angles = [np.sqrt(c / norm) for c in self.coeffs]
    
        # Use initialize to encode weighted superposition: |ψ⟩ = ∑ sqrt(α_j/Σα) |j⟩
        from qiskit.quantum_info import Statevector
        amp_vector = np.zeros(2 ** ancilla, dtype=complex)
        amp_vector[:m] = angles
        amp_vector /= np.linalg.norm(amp_vector)
        qc.initialize(amp_vector, list(range(self.num_qubits, total_qubits)))
    
        # Apply each U_j controlled on ancilla state |j⟩
        for j, Uj in enumerate(self.terms):
            ctrl_state = format(j, f'0{ancilla}b')  # binary string of control state
            ctrl_qubits = list(range(self.num_qubits, total_qubits))
            target_qubits = list(range(self.num_qubits))
    
            # Convert Uj to a gate
            gate = UnitaryGate(Uj, label=f'U{j}')
    
            # Create multi-controlled gate
            from qiskit.circuit.library import MCMT
            controlled_gate = gate.control(num_ctrl_qubits=ancilla, ctrl_state=ctrl_state)
    
            # Append controlled gate
            qc.append(controlled_gate, ctrl_qubits + target_qubits)
    
        return qc


    def simulate(self, psi0: np.ndarray, times: list):
        """
        Simulate the time evolution under LCU for given times.
        """
        if len(psi0) != self.dim:
            raise ValueError("Initial state dimension does not match H_tilde")

        results = []
        for t in times:
            circuit = self.build_lcu_circuit(t)
            sv = Statevector.from_label('0' * self.num_qubits)
            full_state = sv.tensor(Statevector(psi0))
            evolved = full_state.evolve(circuit)
            reduced = evolved.data[-self.dim:]  # trace out ancilla
            results.append(reduced)

        return results

# -------- CLASSES --------
class FDTransform1DA:
    """
    Class that calculates the transformation matrices and hamiltonian
    for the 1D CV equation solver subject to boundary conditions.
    """

    def __init__(self, alpha: np.ndarray, tau: np.ndarray, dx: float, nx: int,
                 order: int, bcs: dict) -> None:
        self.alpha = alpha
        self.tau = tau
        self.dx = dx
        self.nx = nx
        self.order = order
        self.bcs = bcs

        # Define FD operator
        self.d = boundary(scale(self.get_d(self.order, self.nx, self.dx), rows=1), self.bcs)

        # Define mass-like matrices(lagging coefficient)
        self.sqrt_m = self.get_sqrt_m(self.tau, self.get_z(self.nx))
        self.inv_sqrt_m = self.get_inv_sqrt_m(self.tau, self.get_z(self.nx))
        self.inv_m = self.get_inv_m(self.tau, self.get_z(self.nx))

        # Define cholesky decomposition
        self.u = self.get_u(self.tau, self.alpha, self.d)

        # Define stiffness matrix
        self.k = self.get_k(self.u)

        # Define impedance matrix
        inv_m = np.diag(1/np.array(tau))
        self.q = self.get_q(self.k, self.get_z(self.nx), inv_m, self.get_i(self.nx))

        # Define transformation matrices
        self.t = self.get_t(self.u, scale(self.get_z(self.nx), rows=1),
                        scale(self.get_i(self.nx), rows=1))
        self.inv_t = self.get_inv_t(self.t)

        # Define hamiltonian
        self.h_tilde = self.get_h_tilde(scale(self.u, cols=1), self.get_z(self.nx+1), scale(inv_m, rows=1, cols=1))

        self.h_test = self.get_h_test(scale(self.u, cols=1), self.get_z(self.nx+1))

        self.h_embed = self.get_embed_hamiltonian(self.h_tilde)

        self.solver = LindbladFromNonHermitian(self.h_tilde)

        self.lcu = LCU_NonHermitianSimulator(self.h_tilde)

    def simulate_lindblad(self, psi0, time_list):
        """Simulate the Lindblad dynamics with optional initial state and time points."""
        return self.solver.simulate(psi0,t_span=time_list)

    def simulate_lcu(self, psi0, time_list):
        """Simulate the Lindblad dynamics with optional initial state and time points."""
        return self.lcu.simulate(psi0,time_list)

    def get_z(self, length: int) -> np.ndarray:
        """
        Returns a zero matrix with specified size.

        Args:
            length (int): The size of the matrix.


        Returns:
            np.ndarray: The zero matrix.
        """
        return np.zeros((length, length))

    def get_i(self, length: int) -> np.ndarray:
        """
        Returns an identity matrix with specified size.

        Args:
            length (int): The size of the matrix.


        Returns:
            np.ndarray: The identity matrix.
        """
        return np.identity(length)

    def get_sqrt_m(self, tau: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the square root matrix of the medium densities.

        Args:
            tau (np.ndarray): The medium densities.
            z (np.ndarray): The zero matrix.

        Returns:
            np.ndarray: The square root mass matrix.
        """
        return np.block([[np.diag(np.sqrt(tau)), z],
                         [z, np.diag(np.sqrt(tau))]])


    def get_inv_m(self, tau: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the inverse  matrix of the medium densities.
        
        Args:
            tau (np.ndarray): The medium densities.
            z (np.ndarray): The zero matrix.
            
        Returns:
            np.ndarray: The inverse mass matrix.
        """
        return np.block([[np.diag(1/np.array(tau)), z],
                         [z, np.diag(1/np.array(tau))]])

    def get_inv_sqrt_m(self, tau: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the inverse square root matrix of the medium densities.
        
        Args:
            tau (np.ndarray): The medium densities.
            z (np.ndarray): The zero matrix.
            
        Returns:
            np.ndarray: The inverse square root mass matrix.
        """
        return np.block([[np.diag(np.sqrt(1/np.array(tau))), z],
                         [z, np.diag(np.sqrt(1/np.array(tau)))]])

    def get_inv_m(self, tau: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the inverse matrix of the medium densities.
        
        Args:
            tau (np.ndarray): The medium densities.
            z (np.ndarray): The zero matrix.
            
        Returns:
            np.ndarray: The inverse mass matrix.
        """
        return np.block([[np.diag(1/np.array(tau)), z],
                         [z, np.diag(1/np.array(tau))]])

    def get_d(self, order: int, length: int, dx: float) -> np.ndarray:
        """
        Calculates the 1D forward Finite-Difference (FD) matrix of n-th order.

        Args:
            order (int): The order of the FD matrix.
            length (int): The length of the FD matrix.
            dx (float): The grid spacing.

        Returns:
            np.ndarray: The 1D FD matrix.
        """
        return (1/dx) * (1/order) * np.sum([np.diag(np.full(length-k, c), k=-k)
                       for k, c in enumerate(FORWARD_FD_COEFF[order])], axis=0)

    def get_u(self, tau: np.ndarray, alpha: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Calculates the analytical Cholesky decomposition of
        the FD operator with the medium parameters.
        
        Args:
            tau (np.ndarray): The medium densities.
            alpha (np.ndarray): The medium viscosities.
            d (np.ndarray): The 1D FD matrix.
        
        Returns:
            np.ndarray: The analytic Cholesky decomposition matrix.
        """

        return np.diag(np.sqrt(np.array(alpha))) @ d @ np.diag(np.sqrt(1/np.array(tau)))

    def get_k(self, u: np.ndarray)  -> np.ndarray:
        """
        Calculates the stiffness matrix.
        
        Args:
            u (np.ndarray): The Cholesky decomposition matrix.
            
        Returns:
            np.ndarray: The stiffness matrix.
        """
        return -u.T @ u

    def get_q(self, k: np.ndarray, z: np.ndarray, inv_m: np.ndarray, i: np.ndarray) -> np.ndarray:
        """
        Calculates the impedance matrix.
        
        Args:
            k (np.ndarray): The stiffness matrix.
            z (np.ndarray): The zero matrix.
            i (np.ndarray): The identity matrix.
        
        Returns:
            np.ndarray: The impedance matrix.
        """
        return np.block([[z, i],[k, -inv_m]])

    def get_t(self, u: np.ndarray, z: np.ndarray, i: np.ndarray) -> np.ndarray:
        """
        Calculates the transformation matrix.
        
        Args:
            u (np.ndarray): The Cholesky decomposition matrix.
            z (np.ndarray): The zero matrix.
            i (np.ndarray): The identity matrix.
            
        Returns:
            np.ndarray: The transformation matrix.
        """
        return np.block([[u, z],[z, i]])

    def get_inv_t(self, t: np.ndarray) -> np.ndarray:
        """
        Calculates the inverse transformation matrix with 
        least-squares (left inverse).
        
        Args:
            t (np.ndarray): The transformation matrix.
            
        Returns:
            np.ndarray: The inverse transformation matrix.
        """
        return np.linalg.inv(t.T @ t) @ t.T # Left inverse | Least squares (full rank)

    def get_h_tilde(self, u: np.ndarray, z: np.ndarray, inv_m: np.ndarray) -> np.ndarray:
        """
        Calculates the non-Hermitian hamiltonian matrix.
        
        Args:
            u (np.ndarray): The Cholesky decomposition matrix.
            z (np.ndarray): The zero matrix.
            
        Returns:
            np.ndarray: The hamiltonian matrix.
        """
        return np.block([[z, 1j*u],[-1j*u.T, -1j*inv_m]])

    def get_h_test(self, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the Hermitian hamiltonian matrix.
        
        Args:
            u (np.ndarray): The Cholesky decomposition matrix.
            z (np.ndarray): The zero matrix.
            
        Returns:
            np.ndarray: The hamiltonian matrix.
        """
        return np.block([[z, 1j*u],[-1j*u.T, z]])

    def get_embed_hamiltonian(self, h_tilde: np.ndarray) -> np.ndarray:
        """
        Embeds a non-Hermitian matrix tilde{H} into a Hermitian matrix (H_{embed}.
    
        Args:
            h_tilde (np.ndarray): Non-Hermitian input matrix of shape (N, N).
    
        Returns:
            np.ndarray: Hermitian block matrix of shape (2N, 2N) with structure:
                        \[
                        H_{\text{embed}} = \begin{bmatrix} 0 & \tilde{H}^\dagger \\ \tilde{H} & 0 \end{bmatrix}
                        \]
        """
        n = h_tilde.shape[0]  # Get dimension N of \(\tilde{H}\)
        z = np.zeros((n, n), dtype=h_tilde.dtype)  # Zero matrix of same dtype
    
        # Construct \(H_{\text{embed}}\) using block structure
        h_embed = np.block([
            [z,             h_tilde],   # Top row: 0 and \(\tilde{H}^\dagger\)
            [h_tilde.conj().T,       z] # Bottom row: \(\tilde{H}\) and 0
        ])
    
        return h_embed

    def get_dict(self) -> dict:
        """
        Returns a dictionary containing the transformation matrices.
        
        Returns:
            dict: The transformation matrices.
        """
        return {'h_tilde': self.h_tilde,
                'h_test': self.h_test,
                'h_embed': self.h_embed,
                't': self.t,
                'inv_t': self.inv_t,
                'k': self.k,
                'q': self.q,
                'u': self.u,
                'd': self.d,
                'sqrt_m': self.sqrt_m,
                'inv_sqrt_m': self.inv_sqrt_m}

# -------- FUNCTIONS --------
def scale(array: np.ndarray, rows: int = 0, cols: int = 0) -> np.ndarray:
    """
    Scales a matrix by adding rows and columns of zeros.
    
    Args:
        array (np.ndarray): The matrix to be scaled.
        rows (int, optional): The number of rows to be added. Defaults to 0.
        cols (int, optional): The number of columns to be added. Defaults to 0.
    
    Returns:
        np.ndarray: The scaled matrix.
    """
    l = np.zeros((array.shape[0]+rows, array.shape[1]+cols))
    l[:array.shape[0], :array.shape[1]] = array
    return l

def boundary(array: np.ndarray, bcs: dict) -> np.ndarray:
    """
    Applies boundary conditions to a matrix.
    
    Args:
        array (np.ndarray): The matrix to be modified.
        bcs (dict): The boundary conditions.
        
    Returns:
        np.ndarray: The modified matrix.
    """
    for side in ['left', 'right']:
        index = 0 if side == 'left' else -1
        if bcs[side] == 'DBC':
            array[index, index] = 1
        elif bcs[side] == 'NBC':
            array[index, index] = 0
        else:
            pass
    return array
