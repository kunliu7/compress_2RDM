from os import replace
from qiskit import *
import numpy as np

#Operator Imports
from qiskit.opflow import Z, X, I

#Circuit imports
from qiskit_nature.circuit.library import HartreeFock, UCCSD, UCC
from qiskit import Aer, BasicAer
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
import matplotlib.pyplot as plt
import matplotlib
from qiskit.tools.visualization import circuit_drawer
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit.opflow import Z, X, I, StateFn, CircuitStateFn, SummedOp
from qiskit.opflow.converters import CircuitSampler
from qiskit.utils import QuantumInstance
from qiskit.opflow import expectations
from qiskit.circuit import ParameterVector
from qiskit.opflow.converters import CircuitSampler
from qiskit.opflow.expectations import PauliExpectation
from qiskit.providers.aer.noise import depolarizing_error, NoiseModel, QuantumError, ReadoutError
import qiskit.providers.aer.noise as noise
import math, copy, random, itertools, pickle
# import rdm_mapping_functions as rmf
from qiskit.tools.visualization import circuit_drawer
import qiskit.quantum_info as qi

matplotlib.use('Agg')
np.set_printoptions(threshold=np.inf)


def get_excitation_list(num_spin_orbitals, num_particles, num_gates, pa_idx, pb_idx, ha_idx, hb_idx):
    def custom_excitation_list(num_spin_orbitals, num_particles):
        # generate your list of excitations...
        # my_excitation_list = [((0, 1), (2, 3), (0, 1, 2, 3))]
        temp = []
        # a_particle_idx = [x for x in range(num_particles[0])]
        # b_particle_idx = [x for x in range(int(num_spin_orbitals/2), int(num_spin_orbitals/2)+num_particles[0])]
        # a_hole_idx = [x for x in range(num_particles[0], int(num_spin_orbitals/2))]
        # b_hole_idx = [x for x in range(int(num_spin_orbitals/2)+num_particles[0], num_spin_orbitals)]

        for i in range(num_gates):
            a_p = np.random.choice(pa_idx, 1)
            a_h = np.random.choice(ha_idx, 1)
            b_p = np.random.choice(pb_idx, 1)
            b_h = np.random.choice(hb_idx, 1)
            a = [a_p[0], a_h[0]]
            b = [b_p[0], b_h[0]]
            temp.append(a)
            temp.append(b)

        for i in range(len(temp)):
            if len(temp[i]) == 2:
                temp[i] = (tuple([temp[i][0]]), tuple([temp[i][1]]))
            else:
                temp[i] = ((temp[i][0], temp[i][1]), (temp[i][2], temp[i][3]))
        my_excitation_list = temp
        return my_excitation_list

    return custom_excitation_list


def get_uccsd_circ(num_spin_orbitals, num_particles, num_topo, num_gates):
    # setup the mapper and qubit converter
    mapper = JordanWignerMapper()
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
    init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
    circs = []

    # Generate the permutation for constant particle number for the inital state
    a = np.ones(num_particles[0])
    b = np.zeros(int(num_spin_orbitals/2)-num_particles[0])
    temp = np.concatenate((a,b), axis=0)
    circs = []

    for i in range(num_topo):
        # Generate the random initial state, which has equal particles in the spin up and down sector
        rand_init1 = np.random.permutation(temp)
        rand_init2 = np.random.permutation(temp)
        idx1 = np.argwhere(rand_init1 == 1)
        idx2 = np.argwhere(rand_init2 == 1)
        idx1 = list(idx1.reshape((1,-1))[0])
        idx2 = [x+int(num_spin_orbitals/2) for x in list(idx2.reshape((1,-1))[0])]
        idx3 = np.argwhere(rand_init1 == 0)
        idx4 = np.argwhere(rand_init2 == 0)
        idx3 = list(idx3.reshape((1,-1))[0])
        idx4 = [x+int(num_spin_orbitals/2) for x in list(idx4.reshape((1,-1))[0])]
        # Initial state
        ansatz = QuantumCircuit(num_spin_orbitals)
        ansatz.x(idx1)
        ansatz.x(idx2)
        # print(ansatz)

        pa_idx = idx1 # particle in spin alpha orbital indices
        pb_idx = idx2
        ha_idx = idx3 # hole indices
        hb_idx = idx4

        # setup the ansatz for VQE
        custom_excitation_list = get_excitation_list(num_spin_orbitals, num_particles, num_gates, pa_idx, pb_idx, ha_idx, hb_idx)
        # print(custom_excitation_list(num_spin_orbitals, num_particles))
        ansatz =  UCC(qubit_converter=converter, num_particles=num_particles, num_spin_orbitals=num_spin_orbitals, excitations=custom_excitation_list,  reps=1)
        ansatz = ansatz.compose(init_state, front=True, inplace=False)
        phi = list(np.random.random(len(ansatz.parameters))*2*np.pi)
        # phi = list(np.random.randint(-100, 100, len(ansatz.parameters)))
        ansatz = ansatz.bind_parameters(phi)
        # state = qi.Statevector(ansatz)
        # print(state)
        circs.append(ansatz)

    return circs


def get_two_rdm(num_spin_orbitals, num_particles, circs, ideal, noise_model=None):
    r"""
    Args:
        ideal: ideal circuit or not
    """
    mapper = JordanWignerMapper()
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
    rdm_set = []
    counter = 0

    for circ in circs:
        print('the current round for generating ideal={} RDM is'.format(ideal), counter)
        # circuit_drawer(circ, scale=1, filename='./uccsd.png', output='mpl')
        two_rdm = np.zeros((num_spin_orbitals,) * 4)
        num_qubits = num_spin_orbitals
        if ideal:
            for i, j, k, l in itertools.product(range(num_spin_orbitals), repeat=4):
                if (i < int(num_qubits/2) and j < int(num_qubits/2) and k < int(num_qubits/2) and l < int(num_qubits/2)) or\
                (i < int(num_qubits/2) and j > int(num_qubits/2)-1 and k < int(num_qubits/2) and l > int(num_qubits/2)-1) or\
                (i > int(num_qubits/2)-1 and j < int(num_qubits/2) and k > int(num_qubits/2)-1 and l < int(num_qubits/2)) or\
                (i > int(num_qubits/2)-1 and j > int(num_qubits/2)-1 and k > int(num_qubits/2)-1 and l > int(num_qubits/2)-1):
                    s = "+_{i} +_{j} -_{k} -_{l}".format(i=str(i),j=str(j),k=str(k),l=str(l))
                    fermi_term = FermionicOp(s, register_length=num_spin_orbitals)
                    qubit_term = converter.convert(fermi_term, num_particles=num_particles)

                    # Evaluate the 2-RDM term w.r.t. the given circuit
                    temp = ~StateFn(qubit_term) @ CircuitStateFn(primitive=circ, coeff=1.)
                    temp = temp.eval()
                    two_rdm[i,j,k,l] = temp
            two_rdm = np.transpose(two_rdm, (0, 1, 3, 2))
            rdm_set.append(two_rdm)
            # print(two_rdm)
        else:
            backend =  Aer.get_backend('aer_simulator')
            # backend.set_options(device='GPU')
            quantum_instance = QuantumInstance(backend=backend, 
                                           shots=1000, 
                                           noise_model=noise_model)
            circuit_sampler = CircuitSampler(quantum_instance)

            for i, j, k, l in itertools.product(range(num_spin_orbitals), repeat=4):
                if (i < int(num_qubits/2) and j < int(num_qubits/2) and k < int(num_qubits/2) and l < int(num_qubits/2)) or\
                (i < int(num_qubits/2) and j > int(num_qubits/2)-1 and k < int(num_qubits/2) and l > int(num_qubits/2)-1) or\
                (i > int(num_qubits/2)-1 and j < int(num_qubits/2) and k > int(num_qubits/2)-1 and l < int(num_qubits/2)) or\
                (i > int(num_qubits/2)-1 and j > int(num_qubits/2)-1 and k > int(num_qubits/2)-1 and l > int(num_qubits/2)-1):
                    s = "+_{i} +_{j} -_{k} -_{l}".format(i=str(i),j=str(j),k=str(k),l=str(l))
                    fermi_term = FermionicOp(s, register_length=num_spin_orbitals)
                    qubit_term = converter.convert(fermi_term, num_particles=num_particles)

                    # Evaluate the 2-RDM term w.r.t. the given circuit
                    temp = ~StateFn(qubit_term) @ CircuitStateFn(primitive=circ, coeff=1.)
                    temp = circuit_sampler.convert(
                        PauliExpectation().convert(temp)
                        ).eval()
                    two_rdm[i,j,k,l] = temp
            two_rdm = np.transpose(two_rdm, (0, 1, 3, 2))
            rdm_set.append(two_rdm)
    
    return rdm_set


def test_two_rdm(rdm_set):
    for item in rdm_set:
        # two_rdm = np.transpose(item, (0, 1, 3, 2))
        two_rdm = item.reshape((num_spin_orbitals ** 2, num_spin_orbitals ** 2))
        u, s, vh = np.linalg.svd(two_rdm, full_matrices=True)
        print(s)
        print(s.sum())


if __name__ == "__main__":
    train = 1
    num_topo = 200
    num_gates_set = [1, 2]
    num_particles = (1, 1)
    num_spin_orbitals = 6
    # Error probabilities
    prob_1 = 1e-3  # 1-qubit gate
    prob_2 = 1e-2  # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['x', 'y', 'z', 's', 'h', 'rx', 'ry', 'rz'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'swap'])  

    # Generate ideal and noisy data
    for num_gates in num_gates_set:
        ideal_circs = get_uccsd_circ(num_spin_orbitals, num_particles, num_topo, num_gates)
        ideal_rdm_set = get_two_rdm(num_spin_orbitals, num_particles, ideal_circs, ideal=True)
        noisy_rdm_set = get_two_rdm(num_spin_orbitals, num_particles, ideal_circs, ideal=False, noise_model=noise_model)
        # test_two_rdm(ideal_rdm_set)
        # test_two_rdm(noisy_rdm_set)

        # save the ideal data
        output = open('./data/ideal_qubits={n}particle={p}_gates={g}noise_one={one}_two={two}_tr={tr}.pkl'.\
            format(n=num_spin_orbitals,p=num_particles[0]+num_particles[1], g=num_gates, one=prob_1, two=prob_2, tr=train), 'wb')
        pickle.dump(ideal_rdm_set, output)
        output.close()

        # save the noisy data
        output = open('./data/noisy_qubits={n}particle={p}_gates={g}noise_one={one}_two={two}_tr={tr}.pkl'.\
            format(n=num_spin_orbitals,p=num_particles[0]+num_particles[1], g=num_gates, one=prob_1, two=prob_2, tr=train), 'wb')
        pickle.dump(noisy_rdm_set, output)
        output.close()