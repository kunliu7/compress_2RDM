import pickle
import numpy as np
from sympy import false
from compressor import Compressor

import rdm_mapping_functions as mapper


def load_data_from_file(n_qubits, n_particles, n_gates):
    r"""load_data_from_file
    return : 
        noisy: list of noisy_2RDM
        ideal: list of ideal_2RDM
    """
    # f_ideal = open('data/ideal_qubits={n}particle={p}_gates={g}noise_one=0.001_two=0.01_tr={tr}.pkl'.
    #                format(n=n_qubits, p=n_particles, g=n_gates, tr=1), 'rb')
    # f_noisy = open('data/noisy_qubits={n}particle={p}_gates={g}noise_one=0.001_two=0.01_tr={tr}.pkl'.
    #                format(n=n_qubits, p=n_particles, g=n_gates, tr=1), 'rb')
    f_ideal = open('data/ideal_qubits={n}particle={p}noise=0.001_tr={tr}.pkl'.
                   format(n=n_qubits, p=n_particles, tr=1), 'rb')
    f_noisy = open('data/noisy_qubits={n}particle={p}noise=0.001_tr={tr}.pkl'.
                   format(n=n_qubits, p=n_particles, tr=1), 'rb')
    ideal = pickle.load(f_ideal)
    noisy = pickle.load(f_noisy)
    f_ideal.close()
    f_noisy.close()
    return noisy, ideal


def load_data_from_files(n_qubits, n_particles, n_gates_list):
    r"""
    Load data from multiple files 
    """
    noisy, ideal = load_data_from_file(n_qubits, n_particles, n_gates_list[0])
    if len(n_gates_list) > 1:
        for i in n_gates_list[1:]:
            noisy_tmp, ideal_tmp = load_data_from_file(
                n_qubits, n_particles, i)
            noisy = np.concatenate((noisy, noisy_tmp), axis=0)
            ideal = np.concatenate((ideal, ideal_tmp), axis=0)
    return noisy, ideal


def test1():
    # data_dir = './data/'
    n_qubits = 6
    n_spin_orbitals = 6
    n_particles = 2
    n_gates_list = [0]
    _, rdm_ideals = load_data_from_files(n_qubits, n_particles, n_gates_list)
    
    D = np.array(rdm_ideals[0], dtype=float)
    # mat = Compressor._tensor2matrix(D)
    # utri = np.triu(mat)
    # ltri = np.tril(mat)
    # diag = np.diag(np.diag(utri))
    # restored_D = utri + ltri - diag
    
    # diff = np.linalg.norm(restored_D - mat)
    # print(diff)

    # print(np.linalg.norm(utri - ltri.T))
    
    com = Compressor(n_particles, n_spin_orbitals, D)

    # check trace of 2D 2Q 2G
    if False:
        print(com._tensor2matrix(D).trace())
        one_pdm = mapper.map_two_pdm_to_one_pdm(D, n_particles)
        G = mapper.map_two_pdm_to_particle_hole_dm(D, one_pdm)
        print(com._tensor2matrix(G).trace())
        Q = mapper.map_two_pdm_to_two_hole_dm(D, one_pdm)
        print(com._tensor2matrix(Q).trace())

    feature = com.compress(D)
    restored_D = com.decompress(feature)

    diff = np.linalg.norm(restored_D - D)
    print(diff)

    return


if __name__ == '__main__':
    test1()


