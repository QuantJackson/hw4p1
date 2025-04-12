import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_swap = np.moveaxis(Z, self.dim, -1)
        self.Z_swap_shape = Z_swap.shape
        Z_2d = Z_swap.reshape(-1, Z_swap.shape[-1])
        Z_shifted = Z_2d - np.max(Z_2d, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
        A_2d = exp_Z / sum_exp_Z
        A_swap = A_2d.reshape(self.Z_swap_shape)
        A = np.moveaxis(A_swap, -1, self.dim)
        self.A = A
        return A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
        # Reshape input to 2D
        if len(shape) > 2:
            A_moved = np.moveaxis(self.A, self.dim, -1)
            dLdA_moved = np.moveaxis(dLdA, self.dim, -1)
            A_2d = A_moved.reshape(-1, C)
            dLdA_2d = dLdA_moved.reshape(-1, C)
            dot = np.sum(A_2d * dLdA_2d, axis=1, keepdims=True)
            dLdZ_2d = A_2d * (dLdA_2d - dot)
            dLdZ_moved =dLdZ_2d.reshape(A_moved.shape)
            dLdZ = np.moveaxis(dLdZ_moved, -1, self.dim)
        else:
            # For 1D case
            dot = np.sum(self.A * dLdA, axis=0, keepdims=True)
            dLdZ = self.A * (dLdA - dot)
        return dLdZ
 

    