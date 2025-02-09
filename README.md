# Poligrav

Poligrav is a collection of simple political simulation scripts that let you explore how opinions evolve over time. The project includes both command-line scripts (with progress indicators) and an interactive web app—making it easy for anyone to experiment with and visualize the simulation.

## Repository Structure

```
poligrav/
├── app.py                         # Interactive Streamlit web app.
├── README.md                      # This file.
├── requirements.txt               # Python package requirements.
└── scripts/
    ├── __init__.py                # Marks the scripts folder as a package.
    ├── simulation.py              # Contains the shared simulation logic.
    ├── evolution.py               # Local script: simple evolution simulation.
    └── population_party_study.py  # Local script: population & party study.
```

## Getting Started

### Requirements

- **Python 3.6 or later**

Install the required packages by running:

```bash
pip install -r requirements.txt
```

### Option 1: Running Locally

You can run the simulation scripts from the command line.

#### (Optional) Create a Virtual Environment

It's a good idea to create and activate a virtual environment to keep dependencies isolated.

1. **Create the Virtual Environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment:**

   - On **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - On **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

#### Running the Local Scripts

- **Evolution Simulation**

  This script runs a simulation of opinion evolution and displays a terminal-based progress bar. It saves the resulting graph in the `results` folder.

  ```bash
  python scripts/evolution.py
  ```

- **Population & Party Study**

  This script runs simulations for various population and party sizes, displays progress bars, and saves the resulting graph in the `results` folder.

  ```bash
  python scripts/population_party_study.py
  ```

### Option 2: Using the Web App

For an interactive experience, use the built-in web app.

1. **Run the Web App:**

   In the repository root, run:

   ```bash
   streamlit run app.py
   ```

2. **Interact with the App:**

   Your default web browser will open an interface where you can:
   - Adjust simulation parameters (e.g., number of individuals, number of parties, simulation steps, etc.).
   - Run the simulation and view the resulting graph directly in the browser.

## Centralized Simulation Code

The core simulation logic is located in `scripts/simulation.py`. Both local scripts and the web app import this module. Any changes made to the simulation code there will automatically update all parts of the project, ensuring consistency and ease of maintenance.

## Have Fun!

No technical expertise is required—just run the scripts or launch the web app to explore these political models. Feel free to tweak the parameters and experiment with different scenarios to see how opinions converge (or diverge) over time.

Happy exploring!
