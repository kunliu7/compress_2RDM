import numpy as np


class Compressor():
    def __init__(self,
        num_particles: int, num_spin_orbitals: int) -> None:
        self.num_particles = num_particles
        self.num_spin_orbitals = num_spin_orbitals
        pass


    def compress(self, rdm):
        
        N = self.num_spin_orbitals ** 2 // 4

        # get num of elements by square formula of triangle
        S = N * N // 2

        rdm = self._tensor2matrix(rdm)

        utri_arr = np.zeros((3*S, ))
        utri_arr[: S] = \
            self._compress_matrix_to_upper_triangle_array(rdm[: N, : N])
        utri_arr[S: 2*S] = \
            self._compress_matrix_to_upper_triangle_array(rdm[N: 2*N, N: 2*N])
        utri_arr[2*S: ] = \
            self._compress_matrix_to_upper_triangle_array(rdm[3*N: , 3*N: ])
        
        return utri_arr


    def decompress(self, utri_arr):
        rdm = np.zeros((self.num_spin_orbitals ** 2,) * 2) # matrix
        N = self.num_spin_orbitals ** 2 // 4

        # get num of elements by square formula of triangle
        S = N * N // 2

        # restore from the second triangle
        A = self._restore_matrix_by_upper_triangle_array(utri_arr[S: 2*S], N)
        A_tensor = self._matrix2tensor(A)

        B = - A_tensor.transpose([0, 1, 3, 2])
        B = self._tensor2matrix(B)

        C = A_tensor.transpose([1, 0, 3, 2])
        C = self._tensor2matrix(C)

        # restore middle 4
        rdm[N: 2*N, N: 2*N] = A
        rdm[N: 2*N, 2*N: 3*N] = B
        rdm[2*N: 3*N, 2*N, 3*N] = C

        # restore upper left
        rdm[: N, N: N] = \
            self._restore_matrix_by_upper_triangle_array(utri_arr[: S], N)

        # restore button right
        rdm[3*N: 4*N, 3*N: 4*N] = \
            self._restore_matrix_by_upper_triangle_array(utri_arr[2*S:], N)

        rdm = np.triu(rdm)
        rdm = rdm + rdm.conj().T - np.diag(np.diag(rdm))

        return self._matrix2tensor(rdm)


    @staticmethod
    def _restore_matrix_by_upper_triangle_array(utri_arr, n):
        assert utri_arr.shape[0] == int(n * n / 2)
        cnt = 0
        utri = np.zeros((n,) * 2) # upper triangular matrix
        for i in range(n):
            for j in range(n - i):
                utri[i, j] = utri_arr[cnt]
                cnt += 1

        mat = utri + utri.conj().T - np.diag(np.diag(utri))
        return mat


    @staticmethod
    def _compress_matrix_to_upper_triangle_array(mat, n):
        num_elements = n * n // 2
        utri_arr = np.zeros((num_elements))

        cnt = 0
        for i in range(n):
            for j in range(n - i):
                utri_arr[cnt] = mat[i, j]
                cnt += 1

        return utri_arr
    

    @staticmethod
    def _matrix2tensor(mat):
        n = mat.shape[0] // 2
        tensor = mat.reshape((n,) * 4).transpose([0, 1, 3, 2])
        return tensor


    @staticmethod
    def _tensor2matrix(tensor):
        n = tensor.shape[0]
        mat = tensor.transpose([0, 1, 3, 2]).reshape((n,) * 4)
        return mat
        

