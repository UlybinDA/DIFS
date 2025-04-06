# DIFS: Diffraction Instrumentation and Strategy Simulator

**DIFS** is a Python module designed to calculate and analyze strategies for single-crystal diffraction experiments. It enables researchers to simulate diffraction conditions for reciprocal space vectors of crystalline lattices, model detector and obstacles (e.g., beam stops), and detect instrument collisions through logical constraints. DIFS provides a flexible framework for optimizing experimental configurations in crystallography and materials science.

## Features
- **Diffraction Condition Calculation**: Determine whether reciprocal lattice vectors satisfy diffraction conditions for given experimental parameters.
- **Detector and Obstacle Modeling**: Simulate detectors, beam stops, and other obstacles to predict scattered-ray paths.
- **Collision Detection**: Identify potential collisions between instrument components using logical constraints.
- **Strategy Analysis Tools**: Calculate **completeness**, **redundancy**, and **multiplicity** of diffraction data; visualize their dependence on interplanar spacing (*d-spacing*) via customizable plots.
- **Reciprocal Space Analysis**: Analyze coverage of reciprocal space by experimental scans and predict gaps in data collection.
- **Diffraction Mapping**: Generate 3D, 2D, and 1D diffraction maps showing diffraction condition occurrences in goniometer axis coordinates (e.g., φ, ω, κ angles).
- **Basic Graphical Interface (GUI)**: Interactive tools for real-time strategy adjustments, parameter sweeps, and visualization of results.
- **Extensible API**: Customize simulations with user-defined parameters, constraints, and experimental geometries.
