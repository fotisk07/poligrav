# Poligrav

Poligrav is a collection of simple scripts that simulate political models. These scripts let you explore how political opinions evolve over time using fun, interactive simulations.

## Getting Started

### What You Need
- **Python** (version 3.6 or later)
- **numpy**
- **matplotlib**
- **tqdm**

That's it! If you have Python installed, you're good to go.

### How to Run the Scripts

1. **Download or Clone the Repository**  
   Get all the files onto your computer.

2. **Open a Terminal or Command Prompt**  
   Navigate to the folder where the files are located.

3. **(Optional) Set Up a Virtual Environment**  
   This step is optional but recommended if you want to keep the project dependencies separate.

   - **Create the Virtual Environment:**  
     ```bash
     python -m venv venv
     ```
   - **Activate the Virtual Environment:**  
     - On **Windows:**
       ```bash
       venv\Scripts\activate
       ```
     - On **macOS/Linux:**
       ```bash
       source venv/bin/activate
       ```
   - **Install Dependencies:**  
     With the virtual environment activated, run:
     ```bash
     pip install -r requirements.txt
     ```
     This will install **numpy** and **matplotlib**.

4. **Run the Scripts**  
   Poligrav comes with two scripts:

   - **simple_evolution.py**  
     Simulates opinion evolution in a high-dimensional space and tracks an alignment metric.  
     To run it:
     ```bash
     python alignment_evolution.py
     ```
     The script runs the simulation and saves a graph in the `results` folder.

   - **population_party_study.py**  
     Studies the effects of different population sizes and party sizes on opinion convergence.  
     To run it:
     ```bash
     python population_party_study.py
     ```
     This script also saves a graph in the `results` folder showing the evolution of the alignment metric.

5. **View the Results**  
   Open the graphs saved in the `results` folder to see how the opinions evolve and how different parameters affect convergence.

## Have Fun!

No technical expertise is needed—just run the scripts and explore the visualizations. Feel free to tweak the parameters in the code to see how different models behave.

Happy exploring!
