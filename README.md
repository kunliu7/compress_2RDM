# compress_2RDM

Compress 2RDM (reduced density matrix) with Hermiticity, spin parity and anti-symmetry to information-completely decrease matrix elements to 3/32.

## Theory

### Hermiticity

2-RDM can be transformed to a real symmetry matrix.

### Spin parity 

A 2-RDM is a 4-index tensor, with totally N^4 elements. N must be an even number.

Not all the elements in a 2-RDM is meaningful.

Spin orbitals have sectors. Alpha sector ranges [0, N/2), while Beta sector
ranges [N/2, N). The indices of elements of 2-RDM should have equal number of
alpha and beta sectors.

E.g.
Let N = 6.
![](https://latex.codecogs.com/svg.latex?\Large&space;\\langle a_0^\\dagger a_1^\\dagger a_2 a_3\\rangle) must be 0, since 0,1,2 are in alpha sector while 3 is in beta sector, having
different number of indices in these two sectors.

### Anti-symmetry

For each element of 2-RDM, exchanging two indices will cause a negative sign.

![](https://latex.codecogs.com/svg.latex?\Large&space;\\langle a_i^\\dagger a_j^\\dagger a_k a_l\\rangle = - \\langle a_j^\\dagger a_i^\\dagger a_k a_l \\rangle)

### Compress rate

A 2-RDM is a 4-index tensor, with totally N^4 elements.

Taking all these above into consideration, we only need 3/32 N^4 elements to reconstruct the 2-RDM.

## Usage

See [run.py](./run.py) for detail.

```python
com = Compressor(n_particles, n_spin_orbitals, D)
feature = com.compress(D)
restored_D = com.decompress(feature)
diff = np.linalg.norm(restored_D - D)
print('total', diff)
```

`diff` is at the order of 10^{-16}.