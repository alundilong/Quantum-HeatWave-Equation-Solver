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

# -------- CONSTANTS --------
FORWARD_FD_COEFF: Dict[int, List[float]] = {
        1: [-1, 1],
        2: [-3/2, 2, -1/2],
        3: [-11/6, 3, -3/2, 1/3],
        4: [-25/12, 4, -3, 4/3, -1/4]
    }


# -------- CLASSES --------
class FDTransform1DA:
    """
    Class that calculates the transformation matrices and hamiltonian
    for the 1D CV equation solver subject to boundary conditions.
    """

    def __init__(self, alpha: np.ndarray, dx: float, nx: int,
                 order: int, bcs: dict) -> None:
        self.alpha = alpha
        self.dx = dx
        self.nx = nx
        self.order = order
        self.bcs = bcs

        # Define FD operator
        self.d = boundary(scale(self.get_d(self.order, self.nx, self.dx), rows=1), self.bcs)

        # Define cholesky decomposition
        self.u = self.get_u(self.alpha, self.d)

        # Define stiffness matrix
        self.l = self.get_l(self.u)

        self.m = self.get_m(self.l, self.get_i(self.nx), self.get_z(self.nx))

        # Define transformation matrices
        self.t = self.get_t(self.u, scale(self.get_z(self.nx), rows=1),
                        scale(self.get_i(self.nx), rows=1))

        z = self.get_z(self.nx)
        i = self.get_i(self.nx)
        self.i = np.block([[i,z],[z,i]])

        self.inv_t = self.get_inv_t(self.t)

        self.q = self.get_q(self.l, self.get_z(self.nx), self.get_i(self.nx))
        # Define hamiltonian
        self.h_emb = self.get_h_emb(scale(self.u, cols=1), self.get_z(self.nx+1))

        H = 1j*self.q
        self.h_herm = 0.5*(H + H.conj().T)
        self.h_non_herm = 0.5*(H - H.conj().T)

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

    def get_u(self, alpha: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Calculates the analytical Cholesky decomposition of
        the FD operator with the medium parameters.
        
        Args:
            alpha (np.ndarray): The medium viscosities.
            d (np.ndarray): The 1D FD matrix.
        
        Returns:
            np.ndarray: The analytic Cholesky decomposition matrix.
        """

        return np.diag(np.sqrt(np.array(alpha))) @ d

    def get_l(self, u: np.ndarray)  -> np.ndarray:
        """
        Calculates the stiffness matrix.
        
        Args:
            u (np.ndarray): The Cholesky decomposition matrix.
            
        Returns:
            np.ndarray: The stiffness matrix.
        """
        return -u.T @ u

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

    def get_h_emb(self, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the non-Hermitian hamiltonian matrix.
        
        Args:
            u (np.ndarray): The Cholesky decomposition matrix.
            z (np.ndarray): The zero matrix.
            
        Returns:
            np.ndarray: The hamiltonian matrix.
        """
        return np.block([[z, 1j*u],[-1j*u.T, z]])

    def get_q(self, l: np.ndarray, z: np.ndarray, i: np.ndarray) -> np.ndarray:
        """
        Calculates the non-Hermitian hamiltonian matrix.
        
        Args:
            u (np.ndarray): The Cholesky decomposition matrix.
            z (np.ndarray): The zero matrix.
            
        Returns:
            np.ndarray: The hamiltonian matrix.
        """
        return np.block([[z, i],[l, z]])

    def get_m(self, l: np.ndarray, i: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the non-Hermitian hamiltonian matrix.
        
        Args:
            l (np.ndarray): The Cholesky decomposition matrix.
            i (np.ndarray): The zero matrix.
            
        Returns:
            np.ndarray: The hamiltonian matrix.
        """
        return np.block([[z, i],[l, z]])

    def get_dict(self) -> dict:
        """
        Returns a dictionary containing the transformation matrices.
        
        Returns:
            dict: The transformation matrices.
        """
        return {'h_emb': self.h_emb,
                'h_herm': self.h_herm,
                'h_non_herm': self.h_non_herm,
                'l': self.l,
                'm': self.m,
                'i': self.i,
                'q': self.q,
                't': self.t,
                'inv_t': self.inv_t,
                'u': self.u,
                'd': self.d
                }

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
