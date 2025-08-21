"""
Install library to site-packages
"""

import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name='hadamard-random-forest',
    version='0.1.0',
    description='Hadamard Random Forest: real-valued quantum state reconstruction',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url='https://github.com/comp-physics/Quantum-HRF-Tomography',
    packages=[
        'hadamard_random_forest',
    ],
    install_requires=[
        'networkx>=3.3',
        'matplotlib>=3.9.1',
        'pylatexenc>=2.10',
        'pandas>=2.2.2',
        'numpy',
        'scipy',
        'dill',
        'tqdm',
        'sympy',
        'treelib>=1.7.0',
        'qiskit>=2.1.2',
        'qiskit-aer>=0.17.1',
        'qiskit-ibm-runtime>=0.41.1',
        'qiskit-experiments>=0.12.0',
        'mthree==3.0.0',
    ],
    extras_require={
        'dev': [
            'pytest','pytest-runner', 'coverage', 'coveralls'
        ],
    },
    python_requires='>=3.8, <4',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)