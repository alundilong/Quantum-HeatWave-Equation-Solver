import numpy as np
from scipy.linalg import sqrtm, expm

from qiskit_dynamics.solvers import Solver
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli, SparsePauliOp
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
    def __init__(self, H_q: np.ndarray, m0: float = 2.0, evolution_time: float = 1e-18):
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
