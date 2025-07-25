from scipy.linalg import eig, norm, schur
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

def analyze_matrix(H):
    """
    Analyze the matrix H to see if it is normal or diagonalizable with real eigenvalues.
    """
    normal = is_normal(H)
    diagonalizable_real = is_diagonalizable_with_real_eigenvalues(H)

    print("Matrix analysis:")
    print(f" - Normal?                         {'Yes' if normal else 'No'}")
    print(f" - Diagonalizable with real eigs?  {'Yes' if diagonalizable_real else 'No'}")

    return normal, diagonalizable_real


