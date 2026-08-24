# oae-tmm

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21829151.svg)](https://doi.org/10.5281/zenodo.21829151)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21829243.svg)](https://doi.org/10.5281/zenodo.21829243)

Ocean alkalinity enhancement (OAE) transport matrix model (TMM). Uses an implicit Euler solver with the OCIM2-48L transport matrix to simulate the global evolution of dissolved inorganic carbon ($C_\textrm{T}$) and total alkalinity ($A_\textrm{T}$) in response to alkalinity perturbations, with air-sea CO₂ exchange. Includes experiments for maximum-$A_\textrm{T}$ addition targeting preindustrial surface pH (`max_AT`) and per-cell impulse-response efficiency mapping (`impulse_response`).

## Data sources

Download each dataset and place it in the `data/` directory at the path shown. TRACE data is not downloaded from the TRACE gridded product (https://doi.org/10.5281/zenodo.15692788), but higher resolution versions are precomputed as explained in "data preparation" below. All datasets are open access; please cite the original sources.

| Dataset | Directory | Citation |
|---------|-----------|----------|
| OCIM2-48L transport matrix | `data/OCIM2_48L_base/` | Holzer et al. (2021) — https://doi.org/10.5281/zenodo.19944665 |
| GLODAPv2.2016b Mapped Product | `data/GLODAPv2.2016b.MappedProduct/` | Lauvset et al. (2016) — https://www.nodc.noaa.gov/archive/arc0107/0162565/1.1/data/0-data/mapped/ |
| NCEP/DOE Reanalysis II (ice concentration, wind speed) | `data/NCEP_DOE_Reanalysis_II/` | Kanamitsu et al. (2002) — https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html |
| NOAA Extended Reconstruction SST V5 | `data/NOAA_Extended_Reconstruction_SST_V5/` | Huang et al. (2017) — https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html |
| TRACE gridded data product | `data/TRACE_gridded/` | Carter et al. (2025) — https://doi.org/10.5281/zenodo.15003059 |

## Data preparation

After downloading all raw data, run these two scripts once (see installation instructions below) to produce the regridded inputs the model expects:

```bash
python scripts/generate_input_data.py   # regrids GLODAP + NOAA PSL data to OCIM2-48L grid
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
git clone https://github.com/rcbarr1/oae-tmm.git
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
# Pull pre-built image from GitHub Container Registry
docker pull ghcr.io/rcbarr1/oae-tmm:v1.0.0

# Or build locally
docker build -t ghcr.io/rcbarr1/oae-tmm:v1.0.0 .

docker run \
    -v /path/to/data:/app/data \
    -v /path/to/outputs:/app/outputs \
    ghcr.io/rcbarr1/oae-tmm:v1.0.0 python -m experiments.max_AT --test
```

## Reproduce figures

To reproduce all manuscript figures without re-running experiments, download the simulation outputs from Zenodo (https://doi.org/10.5281/zenodo.21829243) and run:

```bash
docker pull ghcr.io/rcbarr1/oae-tmm:v1.0.0
docker run --rm \
    -v /path/to/zenodo_data:/app/data \
    -v /path/to/zenodo_data/outputs:/app/outputs \
    ghcr.io/rcbarr1/oae-tmm:v1.0.0 \
    python dataviz/impulse_response_dataviz.py   # Figure 2
docker run --rm \
    -v /path/to/zenodo_data:/app/data \
    -v /path/to/zenodo_data/outputs:/app/outputs \
    ghcr.io/rcbarr1/oae-tmm:v1.0.0 \
    python dataviz/max_AT_dataviz.py             # Figures 3, 4, 5
docker run --rm \
    -v /path/to/zenodo_data:/app/data \
    -v /path/to/zenodo_data/outputs:/app/outputs \
    ghcr.io/rcbarr1/oae-tmm:v1.0.0 \
    python dataviz/timestepping_dataviz.py       # Figures S1, S2, S3
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

Some pre-computed model outputs are archived on Zenodo at https://doi.org/10.5281/zenodo.21829243 for the timestepping, max AT, and impulse response experiments.

## Citation

If you use this code, please cite:

```bibtex
@software{barrett_oae_tmm_2026,
  author  = {Barrett, Reese},
  title   = {oae-tmm},
  year    = {2026},
  doi     = {10.5281/zenodo.21829151},
  url     = {https://github.com/rcbarr1/oae-tmm}
}
```

## License

MIT — see [LICENSE](LICENSE).

---