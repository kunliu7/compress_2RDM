import numpy as np


class Compressor():
    def __init__(self,
        num_particles: int, num_spin_orbitals: int, rdm_ideal=None) -> None:
        self.num_particles = num_particles
        self.num_spin_orbitals = num_spin_orbitals
        self.rdm_ideal = rdm_ideal
        pass


    def compress(self, rdm):
        
        N = self.num_spin_orbitals ** 2 // 4

        # get num of elements by square formula of triangle
        S = self._get_num_elems_of_tri_mat(N)
        n = self.num_spin_orbitals // 2

        mat = self._tensor2matrix(rdm)

        utri_arr = np.zeros((3*S,))
        utri_arr[: S] = \
            self._compress_matrix_to_upper_triangle_array(self._tensor2matrix(rdm[:n, :n, :n, :n]))
        utri_arr[S: 2*S] = \
            self._compress_matrix_to_upper_triangle_array(self._tensor2matrix(rdm[:n, n:, :n, n:]))
        utri_arr[2*S: ] = \
            self._compress_matrix_to_upper_triangle_array(self._tensor2matrix(rdm[n:, n:, n:, n:]))
        
        return utri_arr


    def decompress(self, utri_arr):
        # rdm = np.zeros((self.num_spin_orbitals ** 2,) * 2) # matrix
        rdm = np.zeros((self.num_spin_orbitals,) * 4) # tensor
        N = self.num_spin_orbitals ** 2 // 4
        n = self.num_spin_orbitals // 2

        # get num of elements by square formula of triangle
        S = self._get_num_elems_of_tri_mat(N)

        # restore from the second triangle
        A = self._restore_matrix_by_upper_triangle_array(utri_arr[S: 2*S], N)
        A_tensor = self._matrix2tensor(A)

        B = - A_tensor.transpose([0, 1, 3, 2])
        # B = self._tensor2matrix(B)

        C = A_tensor.transpose([1, 0, 3, 2])
        # C = self._tensor2matrix(C)

        D = - A_tensor.transpose([1, 0, 2, 3])
        
        # restore middle 4
        # rdm[N: 2*N, N: 2*N] = A
        # diff = np.linalg.norm(self._tensor2matrix(self.rdm_ideal)[N: 2*N, N: 2*N] - A)
        rdm[:n, n:, :n, n:] = A_tensor
        diff = np.linalg.norm(self.rdm_ideal[:n, n:, :n, n:] - A_tensor)
        print('A', diff)

        # rdm[N: 2*N, 2*N: 3*N] = B
        # diff = np.linalg.norm(self._tensor2matrix(self.rdm_ideal)[N: 2*N, 2*N: 3*N] - B)
        rdm[:n, n:, n:, :n] = B
        diff = np.linalg.norm(self.rdm_ideal[:n, n:, n:, :n] - B)
        print('B', diff)

        # rdm[2*N: 3*N, 2*N: 3*N] = C
        # diff = np.linalg.norm(self._tensor2matrix(self.rdm_ideal)[2*N: 3*N, 2*N: 3*N] - C)
        rdm[n:, :n, n:, :n] = C
        diff = np.linalg.norm(self.rdm_ideal[n:, :n, n:, :n] - C)
        print('C', diff)

        rdm[n:, :n, :n, n:] = D
        diff = np.linalg.norm(self.rdm_ideal[n:, :n, :n, n:] - D)
        print('D', diff)

        # rdm = self._tensor2matrix(rdm)
        # restore upper left
        rdm[:n, :n, :n, :n] = \
            self._matrix2tensor(self._restore_matrix_by_upper_triangle_array(utri_arr[: S], N))

        diff = np.linalg.norm(self.rdm_ideal[:n, :n, :n, :n] - rdm[:n, :n, :n, :n])
        print('upper left', diff)

        # restore button right
        rdm[n:, n:, n:, n:] = \
            self._matrix2tensor(self._restore_matrix_by_upper_triangle_array(utri_arr[2*S:], N))

        diff = np.linalg.norm(self.rdm_ideal[n:, n:, n:, n:] - rdm[n:, n:, n:, n:])
        print('button right', diff)
        
        # rdm = self._tensor2matrix(rdm)

        # utri = np.triu(rdm)
        # diag = np.diag(np.diag(rdm))
        # utri -= diag
        # rdm = utri + utri.T + diag

        return rdm #self._matrix2tensor(rdm)


    @staticmethod
    def _restore_matrix_by_upper_triangle_array(utri_arr, n):
        cnt = 0
        utri = np.zeros((n,) * 2) # upper triangular matrix
        for i in range(n):
            for j in range(i, n):
                utri[i, j] = utri_arr[cnt]
                cnt += 1

        diag = np.diag(np.diag(utri))
        mat = utri + utri.T - diag
        return mat


    @staticmethod
    def _compress_matrix_to_upper_triangle_array(mat):
        n = mat.shape[0]
        num_elements = Compressor._get_num_elems_of_tri_mat(n)
        utri_arr = np.zeros((num_elements))

        cnt = 0
        for i in range(n):
            for j in range(i, n):
                utri_arr[cnt] = mat[i, j]
                cnt += 1

        return utri_arr
    

    @staticmethod
    def _get_num_elems_of_tri_mat(n):
        return (n + 1) * n // 2


    @staticmethod
    def _matrix2tensor(mat, transpose=False):
        n = int(np.sqrt(mat.shape[0]))
        if transpose:
            tensor = mat.reshape((n,) * 4).transpose([0, 1, 3, 2])
        else:
            tensor = mat.reshape((n,) * 4)
        return tensor


    @staticmethod
    def _tensor2matrix(tensor, transpose=False):
        n = tensor.shape[0]
        if transpose:
            mat = tensor.transpose([0, 1, 3, 2]).reshape((n*n,) * 2)
        else:
            mat = tensor.reshape((n*n,) * 2)

        return mat
        

    @staticmethod
    def _utri_mat2real_sym_mat(utri):
        diag = np.diag(np.diag(utri))
        mat = utri + utri.T - diag
        return mat
