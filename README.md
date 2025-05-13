## Quantum HRF Tomography
<img src="assets/logo-qht.png" alt="HRF Banner" width="300"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
<a href="https://github.com/comp-physics/Quantum-HRF-Tomography/actions">
<img src="https://github.com/comp-physics/Quantum-HRF-Tomography/actions/workflows/ci.yml/badge.svg" />
</a>
[![Coverage Status](https://coveralls.io/repos/github/comp-physics/Quantum-HRF-Tomography/badge.svg)](https://coveralls.io/github/comp-physics/Quantum-HRF-Tomography)

Efficient and Robust Reconstruction of Real-Valued Quantum States using **Hadamard Random Forests**

### Summary

⚡ **Fast**: Reduces required quantum circuits from exponential to linear in the number of qubits  
🎯 **Accurate**: Achieves 89% fidelity on 10-qubit real states using IBM quantum hardware  
🧠 **Smart**: Uses a random forest over hypercube graphs for efficient sign reconstruction  

### Introduction

This is the code that accompanies the following paper: [arxiv.org/abs/2505.06455](https://arxiv.org/abs/2505.06455)

<div align="center">
<img src="https://github.com/comp-physics/Quantum-HRF-Tomography/blob/master/assets/overview.png" height="300px"> 
</div>

### Install 

We recommend downloading all the files and finishing the installation locally,

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

```bibtex
@journal{
  author = {Zhixin Song and Hang Ren and Melody Lee and Bryan Gard and Nicolas Renaud and Spencer H. Bryngelson},
  title = {Reconstructing Real-Valued Quantum States},
  journal = {arXiv preprint arXiv:2505.06455},
  year = {2025},
}
```


### License

MIT
