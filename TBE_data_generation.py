from mimetypes import init
import openfermion
import cirq
import numpy as np
import itertools
import scipy
import pickle
from openfermion.ops.operators import FermionOperator
np.set_printoptions(threshold=np.inf)

from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from cirq.sim import sparse_simulator, density_matrix_simulator

# Print out familiar OpenFermion "FermionOperator" form of H.
def generate_hamiltonian(H, T, Act, init, end):
    for p in range(init, end):
        for q in range(init, end):
            term = ((p, 1), (q, 0))
            H += openfermion.FermionOperator(term, T[p, q])
            Act[p, q] = T[p, q]

    return H, Act


def get_rdm(n_qubits, psi):
    """
    Calculate the 2-particle-reduced-density-matrix for a given state
    """
    two_rdm = np.zeros((n_qubits,) * 4, dtype=complex)

    for i, j, k, l in itertools.product(range(n_qubits), repeat=4):
        transformed_operator = jordan_wigner(
            FermionOperator(((i, 1), (j, 1), (k, 0), (l, 0))))
        transformed_operator = get_sparse_operator(transformed_operator, n_qubits=n_qubits)
        temp = openfermion.linalg.expectation(transformed_operator, psi)
        two_rdm[i, j, k, l] = temp

    return two_rdm


def generate_circuit(n_qubits, n_particles, simulation_time, random_seed):
    # Generate the random one-body operator.
    T = openfermion.random_hermitian_matrix(n_qubits, seed=random_seed)

    # Generate the random Hamiltonian
    H = openfermion.FermionOperator()
    Act = np.zeros((n_qubits, n_qubits)) + 1.j * np.zeros((n_qubits, n_qubits))
    H, Act = generate_hamiltonian(H, T, Act, 0, int(n_qubits/2))
    H, Act = generate_hamiltonian(H, T, Act, int(n_qubits/2), n_qubits)
    # print(H)

    # Diagonalize Act and obtain basis transformation matrix (aka "u").
    eigenvalues, eigenvectors = np.linalg.eigh(Act)
    basis_transformation_matrix = eigenvectors.transpose()

    # Initialize the qubit register.
    qubits = cirq.LineQubit.range(n_qubits)

    # Start circuit with the inverse basis rotation, print out this step.
    inverse_basis_rotation = cirq.inverse(openfermion.bogoliubov_transform(qubits, basis_transformation_matrix))
    circuit = cirq.Circuit(inverse_basis_rotation)

    # Add diagonal phase rotations to circuit.
    for k, eigenvalue in enumerate(eigenvalues):
        phase = -eigenvalue * simulation_time
        circuit.append(cirq.rz(rads=phase).on(qubits[k]))

    # Finally, restore basis.
    basis_rotation = openfermion.bogoliubov_transform(qubits, basis_transformation_matrix)
    circuit.append(basis_rotation)

    # Initialize a Hartree-Fock state
    initial_state = np.zeros(2**n_qubits).astype(np.complex64)
    initial_state += 1.j*np.zeros(2**n_qubits).astype(np.complex64)
    
    s = '1'
    for i in range(1, n_qubits):
        if 0 < i and i < int(n_particles/2):
            s += '1'
        elif int(n_qubits/2) <= i and int(n_qubits/2) + int(n_particles/2) > i:
            s += '1'
        else:
            s += '0'

    idx = int(s, 2)
    initial_state[idx] = 1+0.j
    
    return circuit, qubits, initial_state, H


def TBE_generate_data(n_qubits, circuit, qubits, initial_state, H, noisy_rate):
    # Use Cirq simulator to apply circuit.
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit, qubit_order=qubits,
                                initial_state=initial_state)
    
    # Compute the ideal state for the given circuit
    ideal_state = result.final_state_vector

    # ideal state fidelity
    # hamiltonian_sparse = openfermion.get_sparse_operator(H)
    # exact_state = scipy.sparse.linalg.expm_multiply(-1j * simulation_time * hamiltonian_sparse, initial_state)
    # fidelity = abs(np.dot(ideal_state, np.conjugate(exact_state)))**2
    # print('ideal state fidelity is', fidelity)
    
    # Generate the ideal rdm data for training
    ideal_rdm = get_rdm(n_qubits, ideal_state)

    # Test ideal RDM
    # d_aaab = ideal_rdm[0:3, 3:6, 3:6, 0:3]
    # print(d_aaab)
    # print(np.shape(d_aaab))
    # Compute the noisy state for the given circuit
    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(noisy_rate))
    noisy_simulator = cirq.DensityMatrixSimulator(noise=noise)
    noisy_result = noisy_simulator.simulate(circuit, qubit_order=qubits,
                                initial_state=initial_state)
    noisy_state = noisy_result.final_density_matrix
    noisy_state = scipy.sparse.coo_matrix(noisy_state)
    
    # Generate the noisy rdm data for training
    noisy_rdm = get_rdm(n_qubits, noisy_state)
    
    return ideal_rdm, noisy_rdm


def test_two_rdm(rdm_set, n):
    for item in rdm_set:
        # two_rdm = np.transpose(item, (0, 1, 3, 2))
        two_rdm = item.reshape((n ** 2, n ** 2))
        u, s, vh = np.linalg.svd(two_rdm, full_matrices=True)
        print(s)
        print(s.sum())


if __name__ == "__main__":
    train = 1
    train_size = 100
    test_size = 1
    n_qubits = 6
    n_particles = 2
    simulation_time = 1.
    noisy_rate = 1e-3
    train_random_seeds = np.random.randint(1000, size=train_size)
    test_random_seeds = np.random.randint(1000, size=test_size)
    ideal_data = []
    noisy_data = []

    if train:
        size = train_size
        random_seeds = train_random_seeds
    else:
        size = test_size
        random_seeds = test_random_seeds

    for i, random_seed in zip(range(size), random_seeds):
        print('the current round is', i)
        circuit, qubits, initial_state, H = generate_circuit(n_qubits, n_particles, simulation_time, random_seed)
        ideal_rdm, noisy_rdm = TBE_generate_data(n_qubits, circuit, qubits, initial_state, H, noisy_rate)
        ideal_data.append(ideal_rdm)
        noisy_data.append(noisy_rdm)
        
        # test data
        # test_two_rdm(ideal_data, n_qubits)
        # test_two_rdm(noisy_data, n_qubits)

    # save the ideal data
    output = open('./data/ideal_qubits={n}particle={p}noise={one}_tr={tr}.pkl'.\
            format(n=n_qubits,p=n_particles, one=noisy_rate, tr=train), 'wb')
    pickle.dump(ideal_data, output)
    output.close()

    # save the noisy data
    output = open('./data/noisy_qubits={n}particle={p}noise={one}_tr={tr}.pkl'.\
            format(n=n_qubits,p=n_particles, one=noisy_rate, tr=train), 'wb')
    pickle.dump(ideal_data, output)
    output.close()
