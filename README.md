# Master Thesis Analytics

This repository contains the full analysis pipeline, statistical evaluation, and figure generation for the master's thesis.

## Project Structure

```text
.
├── data/
│   └── raw/
│       ├── all_apps_wide-<date>.csv
│       ├── tasks_main_trials.csv
│       └── ...
│
├── src/
│   ├── thesis/
│   │   ├── 6_1_descriptives/
│   │   ├── 6_2_accuracy_analysis/
│   │   ├── 6_3_switching_analysis/
│   │   ├── 6_4_reliance_analysis/
│   │   ├── 6_5_confidence_analysis/
│   │   ├── figure_creation/
│   │   ├── variable_construction/
│   │   └── ...
│   │
│   ├── data_loader.py
│   └── ...
│
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Repository Contents

### `data/raw/`

Contains the raw experimental export files and supporting datasets.

Typical files include:

* `all_apps_wide-<date>.csv`

  * Main experimental export
* `tasks_main_trials.csv`

  * Task information (input for the survey)


## Thesis Source Code

All thesis-related code is located in:

```text
src/thesis/
```

The folder structure follows the thesis chapter organization.

## Environment Setup

The project uses Poetry for dependency management.

### Install Dependencies and Activate Environment

```bash
poetry run python <script.py>
```