# Nuclear Reactor MDP Control System

A Python-based simulation and control system that utilizes Markov Decision Processes (MDP) to derive optimal control policies for a nuclear reactor operating under stochastic uncertainty. 

Developed as part of the Artificial Intelligence coursework at Universidad Carlos III de Madrid, this project addresses the complex challenge of managing reactor power levels amidst unpredictable xenon poisoning effects, which often cause non-deterministic responses to control rod adjustments.

## Key Features

* **Stochastic System Modeling:** Models a nuclear reactor's state space (100 power intervals) and actions (Decrease, Maintain, Increase) as a fully defined MDP.
* **Value Iteration Control:** Leverages the `pymdptoolbox` library to solve the MDP at each time step, generating optimal policies that dynamically respond to shifting power demands.
* **Custom Reward Functions:** Implements a rigorous cost matrix that heavily penalizes actions moving the reactor state away from the target demand.
* **Realistic Demand Generation:** Synthesizes realistic, noisy recursive signals to simulate fluctuating real-world power demands using moving average filters.
* **Comprehensive Evaluation Suite:** Evaluates control policies across diverse theoretical reactor probability distributions (e.g., Ideal, Overshoot, Resistant) using MAE, MSE, R², and Pearson Correlation.

## Project Structure

* `main.py`: The entry point of the simulation. Handles argument parsing, initializes the reactor, runs the control loop, and outputs performance metrics and visualizations.
* `ControlModule.py`: Contains the core MDP logic. Generates the Transition Matrix (P) and Cost/Reward Matrix (R), and executes the Value Iteration algorithm to find the optimal action.
* `Reactor.py`: Defines the physical characteristics and probability distributions of the reactor being simulated.
* `DemandGenerator.py`: Generates the time-series data representing power demand using Gaussian noise and recursive scaling.
* `Metrics.py`: Implementation of the evaluation metrics (MAE, MSE, R², Pearson Correlation).
* `Plotter.py`: Visualization toolkit using Matplotlib to generate radar plots, time-series overlays, and metric bar charts.

## Requirements

Ensure you have Python 3.x installed. The required dependencies are:

```bash
pip install numpy matplotlib pymdptoolbox
```

## Usage
* To run the simulation, execute main.py with the required arguments. You must provide a JSON file representing the reactor's physical parameters and probability distributions.

```bash
python main.py --input-reactor <path_to_json> --gamma <discount_factor> --random-seed <seed_value>
```

## Example:

```bash
python main.py --input-reactor reactors/RBMK_II.json --gamma 0.9 --random-seed 42
```

## Arguments
* -i, --input-reactor: Path to the reactor's JSON configuration file.

* -g, --gamma: The discount factor (γ) used in the MDP Value Iteration (e.g., 0.9).

* -r, --random-seed: Seed for the pseudo-random number generator to ensure reproducible results.

## Evaluation & Metrics
The system visualizes the performance of the MDP control policy against the demand curve. The control module is stress-tested against various theoretical reactor types, including:

* **RBMK-II** (Near-Deterministic): High probability of expected outcomes. Serves as a baseline for optimal controllability.

* **RBMK-OVERSHOOT:** A reactor that consistently overshoots by two power levels. Surprisingly, the MDP policy effectively harnesses this momentum to track rapid demand changes with exceptional accuracy.

* **RBMK-BROKEN / RESISTANT:** Edge-case reactors with inverted dynamics to identify the absolute physical limits of the control system.

Upon execution, the script generates a radar plot of the reactor's stochastic dynamics, time-series overlays of demand vs. response, control rod usage charts, and metric summaries.
