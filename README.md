# Self-Driving Laboratory for Coacervate Property Optimization

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Bayesian optimization framework for autonomous optimization of multicomponent coacervate properties using active learning and Gaussian Process regression.

## Abstract

*[To be filled with publication abstract]*

## Repository Contents

```
self-driving-lab-for-coacervate-property-optimization/
│
├── coacervopti/                   # Main Python package
│   ├── __init__.py               # Package initialization
│   ├── acquisition.py            # Acquisition functions for active learning
│   │                             # (EI, UCB, target-based acquisition, etc.)
│   ├── experiment.py             # Core experiment management and orchestration
│   ├── format.py                 # Path definitions and formatting utilities
│   ├── hyperparams.py            # Hyperparameter configurations for GP kernels
│   ├── mlmodel.py                # Machine learning models (Gaussian Process Regressor)
│   ├── sampling.py               # Sampling strategies (FPS, Voronoi, random)
│   └── utils.py                  # General utility functions
│
├── script/                        # Executable scripts
│   ├── insilico_lab_al_simulation.py        # Main simulation script
│   └── insilico_lab_al_simulation_config.yaml  # Configuration template
│
├── datasets/                      # Example datasets
│   ├── coacervate_mean_diameter_maternw_TEST.csv
│   ├── coacervate_number_of_particles_maternw_TEST.csv
│   └── hartmann3d_lhs_noise10.csv
│
├── insilico_al/                   # Output directory for experiment results
│                                  # (created automatically during runs)
│
├── pyproject.toml                 # Python package configuration
├── environment.yaml               # Conda environment specification
├── LICENSE                        # License file
└── README.md                      # This file
```

### Module Descriptions

- **acquisition.py**: Implements various acquisition functions for active learning including Expected Improvement (EI), and target-based acquisition strategies with penalization mechanisms.
  
- **experiment.py**: Provides high-level functions for experiment setup, data pool management, ML model configuration, and validation workflows.

- **mlmodel.py**: Contains the Gaussian Process Regressor implementation with integrated GridSearchCV for hyperparameter optimization.

- **sampling.py**: Implements diverse sampling strategies such as Farthest Point Sampling (FPS), Voronoi tessellation-based sampling, and random sampling.

- **hyperparams.py**: Defines kernel recipes and hyperparameter configurations for Gaussian Process models.

- **utils.py**: General utilities for experiment naming, synthetic data generation, and file management.

## Installation

### Option 1: Using Conda (Recommended)

1. Clone the repository:

```bash
git clone https://github.com/molML/self-driving-lab-for-coacervate-property-optimization.git
cd self-driving-lab-for-coacervate-property-optimization
```

2. Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate coacervopti
```

The package will be installed in editable mode automatically.

### Option 2: Using pip

1. Clone the repository:

```bash
git clone https://github.com/molML/self-driving-lab-for-coacervate-property-optimization.git
cd self-driving-lab-for-coacervate-property-optimization
```

2. Install the package in editable mode:

```bash
pip install -e .
```

### Dependencies

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- pyyaml >= 5.4
- joblib >= 1.0.0

## Running Test Experiments

The current setup was tested on a *Ubuntu 22.04.5 LTS* desktop machine.

### Quick Start

Run a test experiment using the provided configuration file:

```bash
python script/insilico_lab_al_simulation.py -c script/insilico_lab_al_simulation_config.yaml
```

The additional `--rerun` flag allows to re-run an experiment **overwriting** the folder content.

### Configuration

Edit the `insilico_lab_al_simulation_config.yaml` file to customize your experiment. Key parameters include:

```yaml
experiment_name: "my_experiment"          # Name for your experiment
ground_truth_file: "dataset.csv"          # Ground truth data file (in datasets/)
search_space_variables: ["x1", "x2"]      # Input feature columns
target_variables: ["y"]                   # Target property column

n_cycles: 10                              # Number of active learning cycles
init_batch_size: 8                        # Initial training set size
init_sampling: "random"                   # Initial sampling strategy

acquisition_protocol:
  stage_1:
    cycles: N_cycles_stage_1
    acquisition_modes: ['acquisition_function_1', 'acquisition_function_2']
    n_points: [N_acqui_1, N_acqui_2]
```

### Output

Results are saved in the `insilico_al/` directory with the following structure:

```
insilico_al/
└── [experiment_name]/
    ├── dataset/                  # Training/pool/candidate datasets
    ├── models/                   # Saved ML models per cycle
    ├── results/                  # Acquisition landscapes and selections
    └── experiment_summary.json   # Experiment metadata and results
```

### Custom Experiments

To create your own experiment:

1. Prepare your dataset as a CSV file with feature columns and target column(s)
2. Place it in the `datasets/` directory
3. Create a configuration file based on the template
4. Run the simulation script with your config file

## How to Cite

*[To be filled after publication]*

```bibtex
@article{your_citation_key,
  title={Self-Driving Laboratory for Multicomponent Coacervate Property Optimization},
  author={Your Name},
  journal={Journal Name},
  year={2026},
  volume={XX},
  pages={XXX-XXX},
  doi={XX.XXXX/XXXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
