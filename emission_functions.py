import numpy as np

class EmissionFunctions:
    @staticmethod
    def linear(Q, alpha):
        return alpha * Q
    
    @staticmethod
    def quadratic(Q, alpha, beta):
        return alpha * Q + beta * Q**2
    
    @staticmethod
    def exponential(Q, alpha, beta):
        return alpha * np.exp(beta * Q)
    
    @staticmethod
    def logarithmic(Q, alpha, beta):
        return alpha * np.log(beta * Q + 1)
    
    @staticmethod
    def get_piecewise_parameters(func, Q_max, K, *args):
        """Generate parameters for piecewise linear approximation"""
        Q_points = np.linspace(0, Q_max, K+1)
        slopes = []
        intercepts = []
        
        for k in range(K):
            Q1, Q2 = Q_points[k], Q_points[k+1]
            E1 = func(Q1, *args)
            E2 = func(Q2, *args)
            
            slope = (E2 - E1) / (Q2 - Q1)
            intercept = E1 - slope * Q1
            
            slopes.append(slope)
            intercepts.append(intercept)
            
        return slopes, intercepts