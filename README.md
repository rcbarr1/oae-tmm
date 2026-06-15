# oae-tmm

Ocean alkalinity enhancement (OAE) transport matrix model (TMM). Uses an implicit Euler solver with the OCIM2-48L transport matrix to simulate the evolution of dissolved inorganic carbon (CT) and total alkalinity (AT) in response to alkalinity perturbations, with air-sea CO₂ exchange.

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd oae-tmm
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate oae-tmm
```

### 3. Install PyCO2SYS

Must be installed without dependencies to avoid overwriting conda-managed packages:

```bash
pip install --no-deps git+https://github.com/mvdh7/PyCO2SYS@v2.0.0-b5
```

### 4. Install pyTRACE

pyTRACE is vendored in the repository. Check `https://github.com/d-sandborn/TRACE` for a newer version before installing.

```bash
pip install --no-deps -e pyTRACE/ --config-settings editable-mode=compat
```

### 5. Install the oae-tmm package

```bash
pip install -e .
```

## Running experiments

```bash
python -m experiments.exp24 --test   # test run
python -m experiments.exp24          # full run
```

## Running tests

```bash
python tests/unit_test.py
python tests/invariant_test.py
```

