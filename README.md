# hybrid_modeling_team

Project management and development hub for the **Hybrid Modeling Team**. This team supports the development, implementation, and evaluation of machine learning based parameterizations within the Community Earth System Model (CESM), advancing hybrid (physics + ML) modeling approaches for Earth system science.

## Overview

This repository hosts the team's active sub-projects, shared tools, and documentation for building hybrid (ML + physics) components into CESM. Work spans everything from exploratory ML modeling on observational datasets to full integration and calibration of ML-based parameterizations within CESM.

## Sub-projects

| Project | Description |
|---|---|
| [`Convection_trigger_function/`](./Convection_trigger_function) | Improving convection trigger functions in CESM using the TOOCAN dataset. Trains ML classifiers (e.g., XGBoost, neural networks) to predict convection onset from large-scale environmental variables, evaluated offline against TOOCAN-tracked mesoscale convective systems. |

*(Additional sub-projects will be added here as they're brought into the team repo.)*

## Getting Started

- See [`CLAUDE.md`](./CLAUDE.md) for repo conventions and context if you're using Claude Code to contribute.
- See `Start Here.md` for the general ML-parameterization workflow used across sub-projects.
- Each sub-project folder contains its own README with project-specific setup and usage instructions.

## Shared Tools & Guides

| Guide | Description |
|---|---|
| [`integration_tools/FTorch_CAM_integration.md`](./integration_tools/FTorch_CAM_integration.md) | Step-by-step guide for coupling PyTorch ML emulators into CAM via FTorch on Derecho (the `ML Model → FTorch → CAM7` layer in step 4 of the workflow). |

## Contributing

We welcome contributions from the team and collaborators. You can participate by:

- Opening a GitHub Issue to propose new work, request support, or report a problem.
- Submitting a Pull Request with code, documentation, or example improvements.
- Reaching out to the team leads to discuss bringing in a new sub-project.

