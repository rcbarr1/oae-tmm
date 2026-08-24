# syntax=docker/dockerfile:1.6
FROM continuumio/miniconda3:latest
WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

ENV PATH /opt/conda/envs/oae-tmm/bin:$PATH

COPY . .

RUN pip install --no-deps git+https://github.com/mvdh7/PyCO2SYS@v2.0.0-b5
RUN pip install --no-deps -e git+https://github.com/d-sandborn/TRACE@d089107#egg=pytrace --config-settings editable-mode=compat
RUN pip install .
RUN pip install "numpy<2"

RUN conda clean -afy

ENV MPLBACKEND=Agg
