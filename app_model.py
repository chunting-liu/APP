import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import APPModel as appmodel

class APPModel:
    def __init__(self):
        # Model parameters
        self.T = 12  # Number of time periods
        self.I = 5   # Number of products
        self.S = 3   # Number of scenarios
        self.K = 5   # Number of intervals for piecewise approximation
        
        # Initialize random seed for reproducibility
        np.random.seed(42)
        
        # Generate parameters
        self._generate_parameters()
        
    def _generate_parameters(self):
        """Generate model parameters based on the paper specifications"""
        # Probabilities for scenarios
        self.p_s = np.array([0.25, 0.50, 0.25])  # Low, Average, High
        
        # Production costs
        self.c_p = np.random.uniform(50, 150, self.I)
        
        # Holding costs (10% of production cost)
        self.c_h = 0.1 * self.c_p
        
        # Backordering costs
        self.c_b = np.full(self.I, 200)
        
        # Energy and emission costs
        self.c_e = 0.1  # Energy cost per unit
        self.c_c = 50   # Carbon cost per ton
        
        # Production adjustment costs
        self.c_u = np.random.uniform(10, 30, self.I)
        
        # Capacity constraints
        self.cap = np.full((self.I, self.T), 250)
        self.max_b = np.full((self.I, self.T), 50)
        self.max_c = 2500  # Maximum emissions per period
        
        # Generate demand scenarios
        self._generate_demand()
        
        # Emission parameters
        self._generate_emission_parameters()
    
    def _generate_demand(self):
        """Generate demand scenarios with seasonality"""
        base_demand = 120
        std_dev = 30
        seasonal_factors = 1 + 0.2 * np.sin(np.linspace(0, 2*np.pi, self.T))
        
        self.demand = np.zeros((self.S, self.I, self.T))
        scenario_factors = np.array([0.8, 1.0, 1.2])
        
        for s in range(self.S):
            for i in range(self.I):
                mean_demand = base_demand * seasonal_factors * scenario_factors[s]
                self.demand[s,i,:] = np.maximum(0, 
                    np.random.normal(mean_demand, std_dev))
    
    def _generate_emission_parameters(self):
        """Generate parameters for different emission functions"""
        # Linear emission parameters
        self.alpha_linear = np.random.uniform(0.1, 0.5, self.I)
        
        # Quadratic emission parameters
        self.alpha_quad = np.random.uniform(0.05, 0.2, self.I)
        self.beta_quad = np.random.uniform(0.001, 0.005, self.I)
        
        # Exponential emission parameters
        self.alpha_exp = np.random.uniform(0.1, 0.3, self.I)
        self.beta_exp = np.random.uniform(0.01, 0.03, self.I)
        
        # Logarithmic emission parameters
        self.alpha_log = np.random.uniform(1, 3, self.I)
        self.beta_log = np.random.uniform(0.1, 0.3, self.I)
        
        # Energy consumption per unit
        self.e = np.random.uniform(0.5, 1.5, self.I)

    def build_model(self, emission_type='linear'):
        """Build and solve the APP model with specified emission function"""
        model = gp.Model("APP_Model")
        
        # Decision Variables
        Q = model.addVars(self.S, self.I, self.T, name="Q")
        I = model.addVars(self.S, self.I, self.T, name="I")
        B = model.addVars(self.S, self.I, self.T, name="B")
        E = model.addVars(self.S, self.I, self.T, name="E")
        DQ_plus = model.addVars(self.S, self.I, self.T, name="DQ_plus")
        DQ_minus = model.addVars(self.S, self.I, self.T, name="DQ_minus")
        
        # Objective Function
        obj = gp.quicksum(
            self.p_s[s] * (
                gp.quicksum(
                    self.c_p[i] * Q[s,i,t] +
                    self.c_h[i] * I[s,i,t] +
                    self.c_b[i] * B[s,i,t] +
                    self.c_e * self.e[i] * Q[s,i,t] +
                    self.c_c * E[s,i,t] +
                    self.c_u[i] * (DQ_plus[s,i,t] + DQ_minus[s,i,t])
                    for i in range(self.I)
                    for t in range(self.T)
                )
            )
            for s in range(self.S)
        )
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Add constraints based on emission type
        self._add_emission_constraints(model, Q, E, emission_type)
        
        # Add other constraints
        self._add_basic_constraints(model, Q, I, B, DQ_plus, DQ_minus)
        
        return model
    
    def _add_emission_constraints(self, model, Q, E, emission_type):
        """Add emission constraints based on specified emission function"""
        if emission_type == 'linear':
            for s in range(self.S):
                for i in range(self.I):
                    for t in range(self.T):
                        model.addConstr(
                            E[s,i,t] == self.alpha_linear[i] * Q[s,i,t]
                        )
        
        elif emission_type == 'quadratic':
            # Use piecewise linear approximation for quadratic function
            self._add_piecewise_approximation(model, Q, E, 
                lambda x, i: self.alpha_quad[i] * x + self.beta_quad[i] * x**2)
    
    def _add_basic_constraints(self, model, Q, I, B, DQ_plus, DQ_minus):
        """Add basic operational constraints"""
        # Demand fulfillment
        for s in range(self.S):
            for i in range(self.I):
                for t in range(self.T):
                    if t == 0:
                        model.addConstr(
                            Q[s,i,t] + B[s,i,t] == 
                            self.demand[s,i,t] + I[s,i,t]
                        )
                    else:
                        model.addConstr(
                            I[s,i,t-1] + Q[s,i,t] + B[s,i,t-1] == 
                            self.demand[s,i,t] + I[s,i,t] + B[s,i,t]
                        )
        
        # Capacity constraints
        for s in range(self.S):
            for i in range(self.I):
                for t in range(self.T):
                    model.addConstr(Q[s,i,t] <= self.cap[i,t])
                    model.addConstr(B[s,i,t] <= self.max_b[i,t])
        
        # Production adjustment constraints
        for s in range(self.S):
            for i in range(self.I):
                for t in range(1, self.T):
                    model.addConstr(
                        Q[s,i,t] - Q[s,i,t-1] == 
                        DQ_plus[s,i,t] - DQ_minus[s,i,t]
                    )

def main():
    # Create model instance
    app = APPModel()
    
    # Build and solve model with linear emissions
    model = app.build_model(emission_type='linear')
    model.optimize()
    
    # Print results
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal:.2f}")
    else:
        print("Model could not be solved to optimality")

if __name__ == "__main__":
    main()