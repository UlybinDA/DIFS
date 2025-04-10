# DIFS: Diffraction Instrumentation and Strategy Simulator

**DIFS** is a Python module designed to calculate and analyze strategies for single-crystal diffraction experiments. It enables researchers to calculate and visualize diffraction conditions of crystalline lattices. The Calculation allows users to factor in a set of experimental features such as detector shape and possible ray obstacles (e.g., beam stops) and restrict experimental parameters variation by a set of logical constraints. DIFS provides a flexible graphical instrument for optimizing both the diffraction setup configuration and the experimental strategies in crystallography and materials science.

## Features
- **Design experimental setup**: Configure the diffraction system, make the goniometer model (including angle corrections), set the wavelength of the incident beam.
- **Diffraction Condition Calculation**: Determine whether reciprocal lattice vectors satisfy diffraction conditions for given experimental parameters.
- **Detector and Obstacle Modeling**: Simulate detectors, beam stops, and other obstacles to predict scattered-rays detection feasibility.
- **Collision Prevention**: Identify potential collisions between instrument components using logical constraints.
- **Strategy Analysis Tools**: Calculate **completeness**, **redundancy**, and **multiplicity** of diffraction data; visualize their dependence on interplanar spacing (*d-spacing*) via customizable plots.
- **Reciprocal Space Analysis**: Analyze coverage of reciprocal space by experimental and simulated scans and predict gaps in data collection.
- **Diffraction Mapping**: Generate 3D, 2D, and 1D diffraction maps (occurrences of diffraction conditions) in goniometer axis coordinates (e.g., φ, ω, κ angles).
- **Basic Graphical Interface (GUI)**: Interactive tools based on [Dash](https://github.com/plotly/dash) for adjusting strategy in real time and visualizing results.

