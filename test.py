import numpy as np
from scipy.optimize import minimize

def f(w):
  return w**2


def objective(w):
  return np.linalg.norm(f(w))

# Initial guess for the vector w
initial_guess = np.array([1, 2, 3])  # Replace with your initial guess

# Perform minimization
result = minimize(objective, initial_guess)

# Access the minimized vector
minimized_w = result.x
print(minimized_w)
