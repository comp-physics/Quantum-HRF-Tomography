## Quantum-HRF-Tomography

Efficient and Robust Reconstruction of Real-Valued Quantum States using Hadamard Random Forests


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

### Summary

⚡ **Fast**: Reduces required quantum circuits from exponential to linear in number of qubits  
🎯 **Accurate**: Achieves 89% fidelity on 10-qubit real states using IBM hardware  
🧠 **Smart**: Uses random forest over hypercube graphs for efficient sign reconstruction  

### Introduction

This is the code that accompanies the following paper:

<div align="center">
<img src="https://github.com/comp-physics/Quantum-HRF-Tomography/blob/master/assets/overview.png" height="300px"> 
</div>

### Install 

We recommend download all the files and finish the installation locally,

```bash
git clone https://github.com/comp-physics/Quantum-HRF-Tomography
cd Quantum-HRF-Tomography
pip install -e .
```

To visualize the tree structure, one need to install `Graphviz` to enforce the graph layout. For macOS,

```bash
brew install graphviz
pip install pygraphviz
```

Please refer to [here](https://www.graphviz.org/download/) for more instructions. Then one can use 

```python
import hadamard_random_forest as hrf
hrf.get_statevector(num_qubits, num_trees, samples, save_tree=True, show_tree=True)
```


### Citation



### License

MIT
