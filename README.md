# engineered-water-co2-injectivity

PHREEQC-based kinetic screening of engineered-water conditioning strategies for CO₂ storage injectivity and sustainability assessment.

## Overview
This repository provides the core workflow used to:
1) render PHREEQC input files from templates,
2) run fully kinetic PHREEQC simulations,
3) parse selected outputs, and
4) compute key performance indicators (KPIs) for multi-objective screening.

## Repository structure
- `phreeqc/` : PHREEQC templates used for baseline and engineered-water cases
- `python/`  : automation scripts (screening, KPI computation, Pareto analysis, Sustainability Index)
- `data/`    : post-processed datasets used to generate manuscript results

## Scope of shared code
This repository includes simulation automation and KPI computation used in the study.
Scripts used exclusively for figure formatting/visualization are not included.
All numerical values required to reproduce the figures are provided in `data/`.

## Requirements
- Python 3.9+ (recommended: 3.10+)
- PHREEQC v3 installed locally

Install Python dependencies:
```bash
pip install -r requirements.txt
