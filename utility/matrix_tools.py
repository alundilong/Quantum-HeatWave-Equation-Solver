from scipy.linalg import eig, norm, schur, sqrtm
import numpy as np

def is_normal(H, tol=1e-10):
    """
    Check if a matrix H is normal: H H† = H† H
    """
    H_dag = H.conj().T
    commutator = H @ H_dag - H_dag @ H
    return norm(commutator) < tol

def is_diagonalizable_with_real_eigenvalues(H, tol=1e-10):
    """
    Check if H is diagonalizable with real eigenvalues.
    """
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = eig(H)

    # Check if eigenvalues are all real
    if not np.all(np.abs(np.imag(eigvals)) < tol):
        return False

    # Check if matrix is diagonalizable:
    # This means the eigenvectors form a complete basis (i.e., full rank)
    rank = np.linalg.matrix_rank(eigvecs)
    return rank == H.shape[0]

def test_svd_decomposability(H_tilde: np.ndarray, rtol=1e-10):
    """
    Test if a given non-Hermitian matrix H_tilde can be decomposed via SVD.

    Parameters:
        H_tilde : np.ndarray
            The non-Hermitian matrix to decompose (shape n x n)
        rtol : float
            Relative tolerance to assess rank and numerical stability

    Returns:
        result : dict
            A dictionary with SVD components and diagnostic info
    """
    if not isinstance(H_tilde, np.ndarray) or H_tilde.ndim != 2:
        raise ValueError("H_tilde must be a 2D numpy array.")

    if H_tilde.shape[0] != H_tilde.shape[1]:
        raise ValueError("H_tilde must be a square matrix.")

    try:
        U, S, Vh = np.linalg.svd(H_tilde, full_matrices=True)
        rank = np.sum(S > rtol * S[0])
        cond_number = S[0] / S[-1] if S[-1] > 0 else np.inf
        reconstructed = U @ np.diag(S) @ Vh
        residual = np.linalg.norm(reconstructed - H_tilde)

        return {
            'U': U,
            'S': S,
            'Vh': Vh,
            'rank': rank,
            'condition_number': cond_number,
            'residual_norm': residual,
            'success': True,
            'message': 'SVD decomposition succeeded.'
        }

    except np.linalg.LinAlgError as e:
        return {
            'success': False,
            'message': f"SVD failed: {str(e)}"
        }

def is_pseudo_hermitian(Q: np.ndarray, tol: float = 1e-10):
    """
    Test whether a given matrix Q is pseudo-Hermitian.

    Returns:
        is_pseudo (bool): True if Q is pseudo-Hermitian.
        eta (np.ndarray or None): The Hermitian metric operator eta if found.
        H (np.ndarray or None): The similar Hermitian matrix if found.
    """
    if not np.allclose(Q.shape[0], Q.shape[1]):
        raise ValueError("Q must be a square matrix")

    # Try diagonalizing Q
    eigvals, V = eig(Q)
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        return False, None, None  # not diagonalizable

    # Construct candidate eta = (V† V)^(-1)
    eta_inv = V.conj().T @ V
    if np.linalg.matrix_rank(eta_inv) < Q.shape[0]:
        return False, None, None

    eta = np.linalg.inv(eta_inv)
    eta = 0.5 * (eta + eta.conj().T)  # Ensure Hermitian

    # Check the pseudo-Hermiticity condition
    Q_dagger = Q.conj().T
    pseudo_check = eta @ Q
    rhs = Q_dagger @ eta

    if np.allclose(pseudo_check, rhs, atol=tol):
        # Construct similar Hermitian matrix
        try:
            eta_sqrt = sqrtm(eta)
            eta_inv_sqrt = np.linalg.inv(eta_sqrt)
            H = eta_sqrt @ Q @ eta_inv_sqrt
            H = 0.5 * (H + H.conj().T)  # Symmetrize in case of numerical error
            return True, eta, H
        except np.linalg.LinAlgError:
            return False, eta, None
    else:
        return False, None, None

def analyze_matrix(H):
    """
    Analyze the matrix H to see if it is normal or diagonalizable with real eigenvalues.
    """
    normal = is_normal(H)
    diagonalizable_real = is_diagonalizable_with_real_eigenvalues(H)

    print("Matrix analysis:")
    print(f" - Normal?                         {'Yes' if normal else 'No'}")
    print(f" - Diagonalizable with real eigs?  {'Yes' if diagonalizable_real else 'No'}")

    status = test_svd_decomposability(H)
    print(f'SVD : {status["message"]}')

    status,_,_ = is_pseudo_hermitian(H)
    print(f'peudo_hermit: {status}')

    return normal, diagonalizable_real


