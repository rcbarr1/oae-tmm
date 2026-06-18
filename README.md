# oae-tmm

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Ocean alkalinity enhancement (OAE) transport matrix model (TMM). Uses an implicit Euler solver with the OCIM2-48L transport matrix to simulate the global evolution of dissolved inorganic carbon (CT) and total alkalinity (AT) in response to alkalinity perturbations, with air-sea CO₂ exchange. Includes experiments for maximum-AT addition targeting preindustrial surface pH (`max_AT`) and per-cell impulse-response efficiency mapping (`impulse_response`).

## Data sources

Download each dataset and place it in the `data/` directory at the path shown. All datasets are open access; please cite the original sources.

| Dataset | Directory | Citation |
|---------|-----------|----------|
| OCIM2-48L transport matrix | `data/OCIM2_48L_base/` | Holzer et al. (2021) — [DOI PLACEHOLDER] |
| GLODAPv2.2016b Mapped Product | `data/GLODAPv2.2016b.MappedProduct/` | [CITATION PLACEHOLDER] — [DOI PLACEHOLDER] |
| NCEP/DOE Reanalysis II (ice fraction, wind speed) | `data/NCEP_DOE_Reanalysis_II/` | Kanamitsu et al. (2002) — https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html |
| NOAA Extended Reconstruction SST V5 | `data/NOAA_Extended_Reconstruction_SST_V5/` | [CITATION PLACEHOLDER] — [DOI PLACEHOLDER] |
| TRACE anthropogenic CO₂ scenarios | `data/TRACE_gridded/` | [CITATION PLACEHOLDER] — [DOI PLACEHOLDER] |

## Data preparation

After downloading all raw data, run these two scripts once to produce the regridded inputs the model expects:

```bash
python scripts/generate_input_data.py   # regrids GLODAP + NCEP/NOAA to OCIM2-48L grid
python scripts/make_trace_gridded.py    # generates data/TRACE_gridded/ (Canth 2000–2100)
```

Expected `data/` directory structure after downloading and running the preparation scripts:

```
data/
  OCIM2_48L_base/
    OCIM2_48L_base_transport.mat
    OCIM2_48L_base_data.nc
  GLODAPv2.2016b.MappedProduct/
    AT.nc
    CT.nc
    phosphate.nc
    pHtsinsitutp.nc
    salinity.nc
    silicate.nc
    temperature.nc  
  NCEP_DOE_Reanalysis_II/
    icec.nc
    wspd.nc
  NOAA_Extended_Reconstruction_SST_V5/
    sst.nc
  TRACE_gridded/
    OCIM_CanthFromTRACECO2Pathway1.nc
    OCIM_CanthFromTRACECO2Pathway2.nc
    OCIM_CanthFromTRACECO2Pathway3.nc
    OCIM_CanthFromTRACECO2Pathway4.nc
    OCIM_CanthFromTRACECO2Pathway5.nc
    OCIM_CanthFromTRACECO2Pathway6.nc
    OCIM_CanthFromTRACECO2Pathway7.nc
    OCIM_CanthFromTRACECO2Pathway8.nc
    OCIM_CanthFromTRACECO2Pathway9.nc
    OCIM_CanthFromTRACECO2Pathway10.nc
    
```

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

This package uses PyCO2SYS v2. Currently, it must be installed via GitHub without dependencies to avoid overwriting conda-managed packages:

```bash
pip install --no-deps git+https://github.com/mvdh7/PyCO2SYS@v2.0.0-b5
```

### 4. Install pyTRACE

Install in editable mode from GitHub so pyTRACE's internal data files are accessible at their expected paths:

```bash
pip install --no-deps -e git+https://github.com/d-sandborn/TRACE@d089107#egg=TRACE --config-settings editable-mode=compat
```

### 5. Install the oae-tmm package

```bash
pip install -e .
```

## Docker

As an alternative to the conda environment, a Dockerfile is provided. The image does not include data — mount your `data/` and `outputs/` directories at runtime:

```bash
docker build -t oae-tmm .
docker run \
    -v /path/to/data:/app/data \
    -v /path/to/outputs:/app/outputs \
    oae-tmm python -m experiments.max_AT --test
```

## Running experiments

```bash
# list all experiment configurations
python -m experiments.max_AT --list
python -m experiments.impulse_response --list

# run a short test
python -m experiments.max_AT --test
python -m experiments.impulse_response --test

# run a specific experiment by index
python -m experiments.max_AT --exp-id 0
python -m experiments.impulse_response --exp-id 0
```

## Running tests

```bash
python tests/unit_test.py
python tests/invariant_test.py
```

## Published outputs

Pre-computed model outputs are archived on Zenodo at https://doi.org/10.5281/zenodo.XXXXXXX. Available outputs include `max_AT` results across scenarios and time-step resolutions, and the `impulse_response` per-cell air-sea gas exchange efficiency maps. These can be downloaded directly for analysis without re-running the experiments. `impulse_response` in particular is computationally expensive, as it runs one simulation per surface ocean grid cell.

## Citation

If you use this code, please cite:

```bibtex
@software{barrett_oae_tmm_2025,
  author  = {Barrett, Reese},
  title   = {oae-tmm},
  year    = {2025},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/<REPO_URL>}
}
```

## License

MIT — see [LICENSE](LICENSE).

---

> [!WARNING]
> **PRE-PUBLICATION CHECKLIST — DELETE THIS SECTION BEFORE MAKING THE REPO PUBLIC**
>
> - [ ] Fill in OCIM2-48L citation and DOI
> - [ ] Fill in GLODAPv2.2016b citation and DOI
> - [ ] Fill in NOAA Extended Reconstruction SST V5 citation and DOI
> - [ ] Replace all `XXXXXXX` Zenodo DOI placeholders (code DOI + outputs DOI are separate deposits)
> - [ ] Replace `<REPO_URL>` in the citation block
> - [ ] Push Docker image to GitHub Container Registry (`ghcr.io`) and add pull instructions
> - [ ] Verify the `data/` directory tree matches what the download instructions actually produce
> - [ ] Confirm `pip install -e git+...@d089107` installs cleanly in a fresh environment
