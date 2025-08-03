import numpy as np
import warnings
from scipy.linalg import sqrtm, expm, norm
from scipy.special import jv  # Bessel functions for Fourier coefficients
from scipy.linalg import block_diag

from qiskit_dynamics.solvers import Solver
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit import QuantumRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RZGate, XGate

from utility.matrix_tools import analyze_matrix

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
    def __init__(self, H_tilde: np.ndarray, cond_threshold: float = 1e10, regularization_eps: float = 1e-6):
        """
        Initialize the simulator with a non-Hermitian matrix.
        Automatically regularize if condition number is too large.
        """
        self.H_tilde_original = H_tilde
        self.dim = H_tilde.shape[0]
        self.num_qubits = int(np.ceil(np.log2(self.dim)))

        # --- SVD Decomposition ---
        U, S, Vh = np.linalg.svd(H_tilde, full_matrices=True)
        cond_number = S[0] / S[-1] if S[-1] > 0 else np.inf
        print(f"[INFO] Condition number of H_tilde: {cond_number:.2e}")

        if cond_number > cond_threshold:
            print(f"[WARN] Ill-conditioned matrix detected (cond > {cond_threshold}). Applying regularization.")
            S_reg = np.where(S < S[0] / cond_threshold, regularization_eps, S)
            H_tilde = U @ np.diag(S_reg) @ Vh

            residual = np.linalg.norm(H_tilde - self.H_tilde_original)
            print(f"[INFO] Regularized H_tilde residual ‖H_reg - H‖ = {residual:.2e}")

        # Normalize to avoid overflow
        norm = np.linalg.norm(H_tilde)
        self.H_norm = H_tilde / norm
        self.norm_factor = norm

        # Decompose into Pauli basis
        pauli_decomp = SparsePauliOp.from_operator(Operator(self.H_norm))
        self.terms = [Operator(p) for p in pauli_decomp.paulis]
        self.coeffs = pauli_decomp.coeffs

        print(f"[INFO] Number of terms in Pauli decomposition: {len(self.terms)}")

    def build_lcu_circuit(self, t: float) -> QuantumCircuit:
        """
        Build a quantum circuit that implements the LCU step using ancilla superposition and controlled unitaries.
        """
        m = len(self.coeffs)
        ancilla = int(np.ceil(np.log2(m)))
        total_qubits = self.num_qubits + ancilla
        qc = QuantumCircuit(total_qubits)

        # Prepare ancilla superposition weighted by sqrt(coeff / total)
        norm = sum(self.coeffs)
        angles = [np.sqrt(c / norm) for c in self.coeffs]

        amp_vector = np.zeros(2 ** ancilla, dtype=complex)
        amp_vector[:m] = angles
        amp_vector /= np.linalg.norm(amp_vector)
        qc.initialize(amp_vector, list(range(self.num_qubits, total_qubits)))

        # Apply each U_j controlled on ancilla state |j⟩
        for j, Uj in enumerate(self.terms):
            ctrl_state = format(j, f'0{ancilla}b')
            ctrl_qubits = list(range(self.num_qubits, total_qubits))
            target_qubits = list(range(self.num_qubits))

            gate = UnitaryGate(Uj, label=f'U{j}')
            controlled_gate = gate.control(num_ctrl_qubits=ancilla, ctrl_state=ctrl_state)
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
            print(f't = {t}')
            circuit = self.build_lcu_circuit(t)
            sv = Statevector.from_label('0' * self.num_qubits)
            full_state = sv.tensor(Statevector(psi0))
            evolved = full_state.evolve(circuit)
            reduced = evolved.data[-self.dim:]  # trace out ancilla
            results.append(reduced)

        return results

class HermitianDilationSimulator:
    def __init__(self, H_q: np.ndarray, m0: float = 2.0, evolution_time: float = 1.0e-3):
        """
        Initialize the Hermitian Dilation Simulator for a time-independent non-Hermitian Hamiltonian.

        Args:
            H_q: Non-Hermitian Hamiltonian matrix (n x n).
            m0: Scalar for initial M0 matrix (must be > 1).
            evolution_time: Virtual time used to generate a well-behaved M matrix.
        """
        self.H_q = H_q
        analyze_matrix(H_q)
        self.dim = H_q.shape[0]
        self.m0 = m0
        self.M0 = m0 * np.eye(self.dim)
        self.evolution_time = evolution_time
        self.Haq = self._construct_Hermitian_dilation()

    def _construct_Hermitian_dilation(self) -> np.ndarray:
        """
        Construct the Hermitian dilation Hamiltonian H_herm from time-independent H_q.
        """
        # Compute M using time-evolved formulation
        U_dagger = expm(-1j * self.H_q.conj().T * self.evolution_time)
        U = expm(1j * self.H_q * self.evolution_time)
        M_t = U_dagger @ self.M0 @ U

        # Ensure M is Hermitian
        M_t = 0.5 * (M_t + M_t.conj().T)

        # Compute eta and M_inv
        eta = sqrtm(M_t - np.eye(self.dim))
        M_inv = np.linalg.inv(M_t)
        eta_dot = np.zeros_like(eta)  # still zero since H_q is constant

        A = (self.H_q + 1j * eta_dot + eta @ self.H_q @ eta) @ M_inv
        B = 1j * (self.H_q @ eta - eta @ self.H_q - 1j * eta_dot) @ M_inv

        I = np.eye(2)
        sigma_y = np.array([[0, -1j], [1j, 0]])
        return np.kron(I, A) + np.kron(sigma_y, B)

    def simulate(self, psi0: np.ndarray, times: list) -> np.ndarray:
        """
        Simulate the non-Hermitian dynamics using Hermitian dilation.

        Args:
            psi0: Initial state vector of the system (dim, ).
            times: List of time values for simulation.

        Returns:
            psi_t: Array of evolved system states at each time step (len(times), dim).
        """
        dt = times[1] - times[0]

        eta0 = np.sqrt(self.m0 - 1)
        theta = 2 * np.arctan(eta0)
        ancilla0 = np.array([np.cos(theta / 2), -np.sin(theta / 2)])
        ancilla0 = ancilla0 / np.linalg.norm(ancilla0)
        Psi0 = np.kron(ancilla0, psi0)

        Psi_t = [Psi0]
        U = expm(-1j * self.Haq * dt)

        for _ in range(1, len(times)):
            Psi_next = U @ Psi_t[-1]
            Psi_t.append(Psi_next)

        project_ancilla_0 = np.kron(np.array([[1, 0], [0, 0]]), np.eye(self.dim))
        psi_t = [(project_ancilla_0 @ Psi).reshape(2, self.dim)[0] for Psi in Psi_t]
        psi_t = [v / np.linalg.norm(v) for v in psi_t]

        return np.array(psi_t)

class KrylovNonHermitianSimulator:
    def __init__(self, H_tilde: np.ndarray, m: int = 30):
        """
        Initialize the Krylov simulator for a non-Hermitian Hamiltonian.

        Args:
            H_tilde: Non-Hermitian matrix (n x n).
            m: Number of Krylov basis vectors.
        """
        self.H_tilde = H_tilde

        self.dim = H_tilde.shape[0]
        self.m = m

    def _arnoldi_iteration(self, v0: np.ndarray):
        """
        Perform Arnoldi iteration to generate an orthonormal Krylov basis.
        """
        n, m = self.dim, self.m
        V = np.zeros((n, m), dtype=complex)
        Hm = np.zeros((m, m), dtype=complex)

        V[:, 0] = v0 / norm(v0)
        for j in range(m - 1):
            w = self.H_tilde @ V[:, j]
            for i in range(j + 1):
                Hm[i, j] = np.vdot(V[:, i], w)
                w -= Hm[i, j] * V[:, i]
            Hm[j + 1, j] = norm(w)
            if Hm[j + 1, j] < 1e-12:
                return V[:, :j+1], Hm[:j+1, :j+1]
            V[:, j + 1] = w / Hm[j + 1, j]

        # Last column
        w = self.H_tilde @ V[:, m - 1]
        for i in range(m):
            Hm[i, m - 1] = np.vdot(V[:, i], w)
        return V, Hm

    def simulate(self, psi0: np.ndarray, time_list: list) -> np.ndarray:
        """
        Simulate the evolution over a list of time points.

        Args:
            psi0: Initial state vector (n, ).
            time_list: List of time values.

        Returns:
            psi_t_list: Array of evolved states at each time (len(time_list), n).
        """
        # Save the complex matrix
        V, Hm = self._arnoldi_iteration(psi0)
        e1 = np.zeros((Hm.shape[0],), dtype=complex)
        e1[0] = 1.0

        psi_t_list = []
        for t in time_list:
            psi_t_sub = expm(-1j * Hm * t) @ e1
            psi_t = V @ psi_t_sub
            psi_t_list.append(psi_t)
            #psi_t_list.append(psi_t / norm(psi_t))
        return np.array(psi_t_list)

import numpy as np
from scipy.linalg import norm, expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, RZGate, XGate
from qiskit.quantum_info import Statevector
import warnings

class BlockEncodingBuilder:
    def __init__(self, Hm: np.ndarray, n_qubits: int):
        """
        Construct a block-encoding unitary U such that <0|U|0> = Hm / alpha, padded to 2^n x 2^n.

        Args:
            Hm: Non-Hermitian square matrix (m x m).
            n_qubits: Number of qubits for the block-encoding (n_qubits >= ceil(log2(2m))).
        """
        assert Hm.shape[0] == Hm.shape[1], "Hm must be square"
        self.Hm = Hm
        self.m = Hm.shape[0]
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        assert self.dim >= 2 * self.m, "n_qubits too small for block-encoding"
        self.alpha = max(np.linalg.norm(Hm, ord=2), 1e-12)  # Spectral norm, avoid zero
        self.A = Hm / self.alpha
        self.U, self.unitary_error = self._construct_block_encoding()

    def _construct_block_encoding(self):
        """Construct the block-encoding unitary matrix, padded to 2^n x 2^n."""
        A = self.A
        I = np.eye(self.m, dtype=complex)

        # Check contractivity
        if np.linalg.norm(A, ord=2) > 1 + 1e-10:
            warnings.warn("Matrix A is not contractive, block-encoding may be inaccurate")

        try:
            AAdag = A @ A.conj().T
            AdagA = A.conj().T @ A
            eigvals_1 = np.linalg.eigvals(I - AAdag)
            eigvals_2 = np.linalg.eigvals(I - AdagA)

            if np.any(np.real(eigvals_1) < -1e-10) or np.any(np.real(eigvals_2) < -1e-10):
                raise ValueError("Matrix is not contractive - negative eigenvalues detected")

            # Compute matrix square roots
            upper_right = self._matrix_sqrt(I - AAdag)
            lower_left = self._matrix_sqrt(I - AdagA)
            lower_right = -A.conj().T

            # Construct 2m x 2m unitary
            top = np.hstack([A, upper_right])
            bottom = np.hstack([lower_left, lower_right])
            U_small = np.vstack([top, bottom])

            # Pad to 2^n x 2^n
            U = np.eye(self.dim, dtype=complex)
            U[:2*self.m, :2*self.m] = U_small

            # Verify unitarity
            identity = np.eye(self.dim)
            error = np.linalg.norm(U.conj().T @ U - identity, ord='fro')
            if error > 1e-10:
                warnings.warn(f"Block-encoding unitarity error: {error:.2e}")

            return U, error

        except Exception as e:
            raise ValueError(f"Block-encoding failed: {str(e)}") from e

    def _matrix_sqrt(self, M):
        """Compute matrix square root using eigenvalue decomposition."""
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T

def qsp_phases_for_expm(Hm: np.ndarray, t: float, degree: int, tolerance: float = 1e-6):
    """
    Compute QSP phases for approximating exp(-i Hm t) using Chebyshev polynomials.

    Args:
        Hm: Hamiltonian matrix (scaled to spectral norm <= 1)
        t: Evolution time
        degree: Polynomial degree
        tolerance: Approximation tolerance

    Returns:
        List of QSP phase angles
    """
    spectral_radius = max(np.max(np.abs(np.linalg.eigvals(Hm))), 1e-12)
    t_scaled = t * spectral_radius

    # Chebyshev nodes
    nodes = np.cos(np.pi * (2 * np.arange(degree + 1) + 1) / (2 * (degree + 1)))
    func_values = np.exp(-1j * t_scaled * nodes)

    # Simplified phase mapping with symmetry
    phases = [np.arctan2(np.imag(f), np.real(f)) / 2 for f in func_values]
    if degree > 0:
        phases[0] += np.pi / 4
        phases[-1] -= np.pi / 4

    return phases

class KrylovQuantumSimulator:
    def __init__(self, H_tilde: np.ndarray, m: int = 5, degree: int = 20):
        """
        Simulate non-Hermitian Hamiltonian dynamics using Krylov subspace and QSP.

        Args:
            H_tilde: Non-Hermitian Hamiltonian (n x n)
            m: Krylov subspace dimension
            degree: QSP polynomial degree
        """
        self.H_tilde = H_tilde
        self.dim = H_tilde.shape[0]
        self.m = min(m, self.dim)
        self.degree = max(degree, 1)
        self.n_block_qubits = int(np.ceil(np.log2(2 * self.m)))
        self.n_qubits = self.n_block_qubits + 1  # +1 for ancilla
        self.V = None
        self.Hm = None
        self.block_builder = None

    def _arnoldi_iteration(self, v0: np.ndarray):
        """
        Perform Arnoldi iteration to build Krylov subspace.

        Args:
            v0: Initial vector

        Returns:
            V: Orthonormal basis of Krylov subspace
            Hm: Upper Hessenberg matrix
        """
        n, m = self.dim, self.m
        V = np.zeros((n, m), dtype=complex)
        Hm = np.zeros((m, m), dtype=complex)

        v_norm = norm(v0)
        if v_norm < 1e-12:
            raise ValueError("Initial vector has zero norm")
        V[:, 0] = v0 / v_norm

        for j in range(m - 1):
            w = self.H_tilde @ V[:, j]
            for i in range(j + 1):
                Hm[i, j] = np.vdot(V[:, i], w)
                w -= Hm[i, j] * V[:, i]

            beta = norm(w)
            Hm[j + 1, j] = beta
            if beta < 1e-12:
                return V[:, :j + 1], Hm[:j + 1, :j + 1]

            V[:, j + 1] = w / beta

        w = self.H_tilde @ V[:, m - 1]
        for i in range(m):
            Hm[i, m - 1] = np.vdot(V[:, i], w)

        return V, Hm

    def simulate_classical(self, psi0: np.ndarray, time_list: list) -> np.ndarray:
        """
        Simulate time evolution classically (exact).

        Args:
            psi0: Initial state vector
            time_list: List of times to evaluate

        Returns:
            Array of state vectors at each time
        """
        self.V, self.Hm = self._arnoldi_iteration(psi0)
        psi0_krylov = self.V.conj().T @ psi0
        psi_t_list = []

        for t in time_list:
            exp_Hm_t = expm(-1j * self.Hm * t)
            psi_krylov_t = exp_Hm_t @ psi0_krylov
            psi_full_t = self.V @ psi_krylov_t
            psi_t_list.append(psi_full_t)

        return np.array(psi_t_list)

    def simulate_quantum(self, psi0: np.ndarray, time_list: list) -> np.ndarray:
        """
        Simulate time evolution using QSP and block-encoding.

        Args:
            psi0: Initial state vector
            time_list: List of times to evaluate

        Returns:
            Array of state vectors at each time
        """
        self.V, self.Hm = self._arnoldi_iteration(psi0)
        self.block_builder = BlockEncodingBuilder(self.Hm, self.n_block_qubits)
        U_block = self.block_builder.U
        alpha = self.block_builder.alpha

        psi0_krylov = self.V.conj().T @ psi0
        psi_t_list = []

        for t in time_list:
            try:
                # Compute QSP phases
                phases = qsp_phases_for_expm(self.Hm / alpha, t, self.degree)

                # Prepare initial quantum state
                initial_state = self._prepare_initial_quantum_state(psi0_krylov, self.n_qubits, self.m)

                # Build and simulate QSP circuit
                qc = QuantumCircuit(self.n_qubits)
                qc.initialize(initial_state, range(self.n_qubits))
                qsp_circuit = self._build_qsp_block_encoding_circuit(U_block, phases)
                qc.compose(qsp_circuit, inplace=True)

                # Simulate circuit
                statevector = Statevector(qc)
                psi_krylov_t = self._extract_result_from_quantum_state(statevector.data, self.m)

                # Project back to full space
                psi_full_t = self.V @ psi_krylov_t
                psi_t_list.append(psi_full_t)

            except Exception as e:
                warnings.warn(f"Quantum simulation failed at t={t}: {e}. Using classical.")
                exp_Hm_t = expm(-1j * self.Hm * t)
                psi_krylov_t = exp_Hm_t @ psi0_krylov
                psi_full_t = self.V @ psi_krylov_t
                psi_t_list.append(psi_full_t)

        return np.array(psi_t_list)

    def _prepare_initial_quantum_state(self, psi0_krylov: np.ndarray, n_qubits: int, krylov_dim: int):
        """
        Prepare initial quantum state for QSP.

        Args:
            psi0_krylov: Initial Krylov state
            n_qubits: Total number of qubits
            krylov_dim: Krylov subspace dimension

        Returns:
            Normalized quantum state vector
        """
        dim_full = 2 ** n_qubits
        quantum_state = np.zeros(dim_full, dtype=complex)
        block_dim = dim_full // 2

        psi0_krylov_normalized = psi0_krylov / norm(psi0_krylov)
        quantum_state[:min(krylov_dim, block_dim)] = psi0_krylov_normalized
        return quantum_state / norm(quantum_state)

    def _extract_result_from_quantum_state(self, final_state: np.ndarray, krylov_dim: int):
        """
        Extract evolved Krylov state from quantum state.

        Args:
            final_state: Final state vector
            krylov_dim: Krylov subspace dimension

        Returns:
            Evolved Krylov state
        """
        block_dim = len(final_state) // 2
        result_block = final_state[:block_dim]
        psi_krylov_result = result_block[:krylov_dim]

        norm_result = norm(psi_krylov_result)
        if norm_result > 1e-12:
            psi_krylov_result = psi_krylov_result / norm_result
        return psi_krylov_result

    def _build_qsp_block_encoding_circuit(self, U_block: np.ndarray, phases: list):
        """
        Build QSP circuit with block-encoded unitary.

        Args:
            U_block: Block-encoded unitary matrix (2^n x 2^n)
            phases: QSP phase angles

        Returns:
            QuantumCircuit for QSP
        """
        qc = QuantumCircuit(self.n_qubits, name='QSP_Block_Encoding')
        ancilla = self.n_qubits - 1
        block_qubits = list(range(self.n_block_qubits))

        # Initial phase rotation
        qc.rz(2 * phases[0], ancilla)

        # QSP sequence
        unitary_gate = UnitaryGate(U_block, label='U_block')
        controlled_unitary = unitary_gate.control(num_ctrl_qubits=1, label='c-U_block')

        for k in range(1, len(phases)):
            qc.x(ancilla)
            qc.rz(2 * phases[k], ancilla)
            qc.x(ancilla)
            qc.append(controlled_unitary, [ancilla] + block_qubits)

        return qc

if __name__ == "__main__":
    # Test with a small non-Hermitian Hamiltonian
    np.random.seed(42)
    n = 4
    H_tilde = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H_tilde = 0.1 * (H_tilde + H_tilde.conj().T) + 0.05j * (H_tilde - H_tilde.conj().T)

    # Initialize simulator with m=2 to ensure 2m=4 is a power of 2
    simulator = KrylovQuantumSimulator(H_tilde, m=2, degree=20)

    # Initial state
    psi0 = np.random.randn(n) + 1j * np.random.randn(n)
    psi0 = psi0 / norm(psi0)

    # Time points
    times = [0.1, 0.2, 0.5]

    print("Testing Krylov Quantum Simulator")
    print("=" * 50)

    # Classical simulation
    try:
        results_classical = simulator.simulate_classical(psi0, times)
        print("Classical simulation completed:")
        for i, t in enumerate(times):
            print(f"  t={t:.1f}: norm = {norm(results_classical[i]):.6f}")
    except Exception as e:
        print(f"Classical simulation failed: {e}")

    # Quantum simulation
    try:
        results_quantum = simulator.simulate_quantum(psi0, times)
        print("\nQuantum simulation completed:")
        for i, t in enumerate(times):
            print(f"  t={t:.1f}: norm = {norm(results_quantum[i]):.6f}")

        # Compare results
        print("\nComparison with classical simulation:")
        errors = [norm(results_quantum[i] - results_classical[i]) for i in range(len(times))]
        for i, (t, error) in enumerate(zip(times, errors)):
            fidelity = abs(np.vdot(results_quantum[i], results_classical[i]))**2
            print(f"  t={t:.1f}: error = {error:.2e}, fidelity = {fidelity:.6f}")
    except Exception as e:
        print(f"Quantum simulation failed: {e}")

    # Test block-encoding
    print("\nTesting block-encoding...")
    try:
        test_matrix = np.array([[0.1 + 0.05j, 0.2], [0.15, 0.1 - 0.05j]], dtype=complex)
        block_builder = BlockEncodingBuilder(test_matrix, n_qubits=2)
        print(f"Block-encoding α: {block_builder.alpha:.6f}")
        print(f"Unitarity error: {block_builder.unitary_error:.2e}")
        error = norm(block_builder.alpha * block_builder.U[:2, :2] - test_matrix)
        print(f"Block-encoding extraction error: {error:.2e}")
    except Exception as e:
        print(f"Block-encoding test failed: {e}")

    print("\nAll tests completed!")

class NaimarkDilationSimulator:
    def __init__(self, H_q: np.ndarray):
        """
        Simulate non-Hermitian dynamics using true Naimark dilation for normal matrices.

        Args:
            H_q: A normal (non-Hermitian) matrix.
        """
        self.H_q = H_q
        self.dim = H_q.shape[0]
        assert np.allclose(H_q @ H_q.conj().T, H_q.conj().T @ H_q), "H_q must be normal"
        self.U, self.Lambda = self._diagonalize(H_q)

    def _diagonalize(self, H):
        """Diagonalize a normal matrix."""
        eigvals, eigvecs = eig(H)
        return eigvecs, np.diag(eigvals)

    def simulate(self, psi0: np.ndarray, times: np.ndarray):
        """
        Simulate the evolution using spectral decomposition.

        Args:
            psi0: Initial state vector.
            times: Array of time points.

        Returns:
            psi_t: Array of evolved states at each time point.
        """
        U, Lambda = self.U, self.Lambda
        U_dag = U.conj().T
        psi_t = []
        for t in times:
            Ut = U @ np.diag(np.exp(-1j * np.diag(Lambda) * t)) @ U_dag
            psi_t.append(Ut @ psi0)
        return np.array(psi_t)

    def exact_solution(self, psi0: np.ndarray, times: np.ndarray):
        """
        Compute the exact solution using matrix exponential.

        Args:
            psi0: Initial state vector.
            times: Array of time points.

        Returns:
            psi_t_exact: Array of evolved states at each time point.
        """
        psi_t_exact = [expm(-1j * self.H_q * t) @ psi0 for t in times]
        return np.array(psi_t_exact)

class WarpingPhaseTransformerSimulator:
    def __init__(self, A: np.ndarray, N: int = 16):
        """
        Simulate non-Hermitian dynamics using true Naimark dilation for normal matrices.

        A : np.ndarray
            Input square matrix (real or complex), not assumed Hermitian.
        N : int
            Number of Fourier modes in the auxiliary variable domain (should be even).
        L : float
            Length of the p-domain for Fourier dual variable η.
        positive_only : bool
            Whether to project only onto the p > 0 subspace (default True).
        """
        self.A = A
        self.N = N
        self.L = 2 * np.pi
        self.positive_only = True
        self.H_block, self.eta = self.schrodingerise_operator_block_hamiltonian()

    def schrodingerise_operator_block_hamiltonian(self):
        """
        Construct the block-diagonal Hermitian Hamiltonian for Schrödingerisation
        from a possibly non-Hermitian matrix A.
    
        Returns:
            H_block : np.ndarray
                Block-diagonal Hermitian matrix of shape (N*d, N*d),
                where d is the dimension of A.
            eta_vals : np.ndarray
                Array of η_j values used to construct each block.
        """
        A = self.A
        N = self.N
        L = self.L
        if not isinstance(A, np.ndarray):
            raise TypeError("Input A must be a NumPy ndarray.")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")
        if N % 2 != 0:
            raise ValueError("N must be even for symmetric discretization.")
    
        d = A.shape[0]
        A_dag = A.conj().T
    
        # Hermitian and anti-Hermitian parts
        H = 0.5 * (A + A_dag)
        H_bar = 0.5j * (A_dag - A)
    
        # Fourier mode grid
        eta_vals = (2 * np.pi / L) * (np.arange(N) - N // 2)
    
        # Build block diagonal Hamiltonian
        H_blocks = [eta * H + H_bar for eta in eta_vals]
        H_block = block_diag(*H_blocks)
    
        return H_block, eta_vals
    
    def reconstruct_original_solution(self, w_t: np.ndarray):
        """
        Reconstruct the original solution u(t) from the Schrödingerised solution w(t)
        using inverse Fourier transform along the p-domain.
    
        Parameters:
            w_t : np.ndarray
                Flattened solution vector of shape (N*d,), where N is number of Fourier modes
                and d is the size of the original vector u(t).
    
        Returns:
            u_t : np.ndarray
                Reconstructed solution vector u(t) of shape (d,)
        """
        N = self.N
        positive_only = self.positive_only

        d = w_t.size // N
        w_t_matrix = w_t.reshape((N, d))  # shape: (N, d)
    
        # Inverse FFT over the Fourier (p) axis
        v_t_p = np.fft.ifft(w_t_matrix, axis=0)
    
        if positive_only:
            # Keep only p > 0 (second half of array, assuming symmetric FFT)
            p_positive_indices = np.arange(N // 2, N)
            v_t_p = v_t_p[p_positive_indices, :]
    
        # Integrate (sum) over the p-domain to recover u(t)
        u_t = np.sum(v_t_p, axis=0).real  # discard any numerical imaginary part
    
        return u_t

    def simulate(self, psi0: np.ndarray, times: np.ndarray):
        H_block = self.H_block
        eta = self.eta
        N = self.N
        w0 = np.tile(psi0, N).astype(complex)

        solutions = []
        for t in times:
            U_t = expm(-1j * H_block * t)
            w_t = U_t @ w0
            u_t = self.reconstruct_original_solution(w_t)
            solutions.append(u_t)

        result = np.array(solutions)
        return result
