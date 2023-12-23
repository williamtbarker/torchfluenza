import numpy as np
from scipy.integrate import odeint

class InfluenzaBioreactorModel:
    def __init__(self, initial_virus_conc, initial_cell_conc, reactor_volume, params):
        """
        Initialize the model with initial conditions and parameters.
        :param initial_virus_conc: Initial concentration of the virus (virions/mL)
        :param initial_cell_conc: Initial concentration of host cells (cells/mL)
        :param reactor_volume: Volume of the bioreactor (L)
        :param params: Dictionary of model parameters
        """
        self.virus_conc = initial_virus_conc
        self.cell_conc = initial_cell_conc
        self.reactor_volume = reactor_volume
        self.params = params

    def growth_kinetics(self, y, t):
        """
        Defines the differential equations for virus and cell concentrations.
        :param y: Current state of the system [virus concentration, cell concentration]
        :param t: Time variable
        :return: Derivatives [dV/dt, dC/dt]
        """
        V, C = y
        mu = self.params['growth_rate']
        kd = self.params['death_rate']
        beta = self.params['infection_rate']

        dVdt = beta * V * C - kd * V  # Virus growth and death
        dCdt = mu * C * (1 - C / self.params['carrying_capacity']) - beta * V * C  # Cell growth and infection

        return [dVdt, dCdt]

    def run_simulation(self, time_span):
        """
        Runs the simulation over the given time span.
        :param time_span: Array of time points at which to solve the system
        :return: Simulation results (virus and cell concentrations over time)
        """
        initial_conditions = [self.virus_conc, self.cell_conc]
        results = odeint(self.growth_kinetics, initial_conditions, time_span)
        return results

# Parameters and Initial Conditions
params = {
    'growth_rate': 0.02,          # Cell growth rate
    'death_rate': 0.01,           # Virus death rate
    'infection_rate': 0.0001,     # Rate of infection of cells by virus
    'carrying_capacity': 1e6      # Carrying capacity of the bioreactor (cells/mL)
}
initial_virus_conc = 1e3  # Initial virus concentration (virions/mL)
initial_cell_conc = 1e5   # Initial cell concentration (cells/mL)
reactor_volume = 10       # Volume of the bioreactor (L)

# Create and run the model
model = InfluenzaBioreactorModel(initial_virus_conc, initial_cell_conc, reactor_volume, params)
time_span = np.linspace(0, 100, 1000)  # Simulate for 100 hours
results = model.run_simulation(time_span)

# Post-process and analyze the results
import matplotlib.pyplot as plt

virus_conc, cell_conc = results.T
plt.plot(time_span, virus_conc, label='Virus Concentration')
plt.plot(time_span, cell_conc, label='Cell Concentration')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.title('Influenza Virus Growth in Bioreactor')
plt.legend()
plt.show()

