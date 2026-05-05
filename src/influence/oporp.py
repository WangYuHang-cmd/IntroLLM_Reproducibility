"""
OPORP (One Permutation One Random Projection) gradient compression.

Compresses high-dimensional gradient vectors while preserving dot products,
enabling efficient influence function computation. Based on Li and Li (2023).

The RapidGrad class implements:
1. Random sign flip: multiply each element by +1 or -1
2. Multi-level permutation: permute the vector using factored permutations
3. Bin-and-sum: divide into K bins and sum within each bin

This reduces a 42M-dim gradient vector (160MB) to 65K dims (128KB).
"""

import os
import pickle
from copy import copy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class RapidGrad:
    """OPORP-based gradient compressor.

    Args:
        shuffle_lambda: Number of permutation rounds.
        filepath: Directory to cache the permutation/random matrices.
        device: Torch device.
        seed: Random seed for reproducibility.
    """

    def __init__(self, shuffle_lambda, filepath, device, seed=42):
        self.is_init = False
        self.D = None
        self.K = None
        self.random_mat = None
        self.M = 1
        self.shuffle_lambda = shuffle_lambda
        self.perm_mat_list = []
        self.perm_dim_list = []
        self.filepath = filepath
        self.device = device
        self.seed = seed

    def __call__(self, vec, K):
        """Compress a gradient vector to K dimensions.

        Args:
            vec: 1-D tensor of gradient values (already padded).
            K: Target dimension(s). Int or list of ints, must divide self.D.

        Returns:
            Compressed vector(s) of dimension K.
        """
        if not self.is_init:
            print("Creating random and shuffling matrices. This may take a few minutes.")
            D = len(vec)
            self.init(D)

        # Apply multi-level permutation
        for i, (dim, perm_mat) in enumerate(zip(self.perm_dim_list, self.perm_mat_list)):
            if i % 2 == 0:
                vec = vec.reshape((dim, -1))
                vec = vec[perm_mat, :]
            else:
                vec = vec.reshape((-1, dim))
                vec = vec[:, perm_mat]
        vec = vec.reshape(-1)

        # Random sign flip
        vec = vec * self.random_mat

        # Bin-and-sum compression
        if isinstance(K, list):
            return [torch.sum(vec.reshape((-1, self.D // k)), axis=1) for k in K]
        else:
            step = self.D // K
            return torch.sum(vec.reshape((-1, step)), axis=1)

    def init(self, D):
        self.is_init = True
        np.random.seed(self.seed)
        self.D = D
        self.file_name = os.path.join(
            self.filepath,
            f"RapidGrad_D{self.D}_n{self.shuffle_lambda}_seed{self.seed}.obj",
        )
        if not self.load():
            self.create_random_mat(D)
            self.create_perm_mat(D)
            self.save()
        self.random_mat = (
            torch.from_numpy(self.random_mat).to(dtype=torch.float16).to(self.device)
        )

    def create_random_mat(self, D):
        self.random_mat = np.random.randint(0, 2, (D,), dtype=np.int8)
        self.random_mat[self.random_mat < 1e-8] = -1

    def create_perm_mat(self, D):
        # Factorize D into primes
        lt = []
        d = D
        while d != 1:
            for i in range(2, int(d + 1)):
                if d % i == 0:
                    lt.append(i)
                    d = d / i
                    break
        # Create shuffle_lambda random permutations on factor dimensions
        for _ in tqdm(range(self.shuffle_lambda), desc="Creating permutation matrices"):
            x = np.random.randint(len(lt) // 4, len(lt) // 2 + 1)
            np.random.shuffle(lt)
            dim = np.prod(lt[:x], dtype=np.longlong)
            self.perm_dim_list.append(dim)
            self.perm_mat_list.append(np.random.permutation(dim))

    def save(self):
        if os.path.exists(self.file_name):
            return
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)
        with open(self.file_name, "wb") as f:
            pickle.dump(self, f)

    def load(self):
        if not os.path.exists(self.file_name):
            return False
        with open(self.file_name, "rb") as f:
            new_obj = pickle.load(f)
        device = self.device
        self.__dict__ = copy(new_obj.__dict__)
        self.device = device
        return True


def pad_gradient(x):
    """Pad gradient vector to nearest multiple of 2^24 for clean OPORP factorization."""
    D = len(x)
    K = 2**24
    new_D = ((D - 1) // K + 1) * K
    return F.pad(x, (0, new_D - D), "constant", 0)
