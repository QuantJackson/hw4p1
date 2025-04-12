import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A
        A_flatten = A.reshape(-1, A.shape[-1])
        Z = A_flatten @ self.W.T + self.b
        Z_unflatten = Z.reshape(*A.shape[:-1], -1)
        return Z_unflatten

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Compute gradients (refer to the equations in the writeup)
        dLdZ_2d = dLdZ.reshape(-1, dLdZ.shape[-1])
        self.dLdW = dLdZ_2d.T @ self.A.reshape(-1, self.A.shape[-1])
        self.dLdb = np.sum(dLdZ_2d, axis=0)  # Sum over the batch dimension
        dLdA_2d = dLdZ_2d @ self.W
        self.dLdA = dLdA_2d.reshape(*self.A.shape)
        return self.dLdA
