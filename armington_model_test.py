import numpy as np

# this is the fuckup
def excess_demand_vars(a, tau, w, L, sigma, A):
  """
  This function calculates the excess demand for the Armington Model.

  Args:
      a: A numpy array of taste shocks.
      tau: A numpy array of transport costs.
      w: A numpy array of wages.
      L: A numpy array of labor supplies.
      sigma: The elasticity of substitution.
      A: A numpy array of productivity levels.

  Returns:
      A numpy array of excess demands.
  """

  S = len(a)  # Number of goods
  Z = np.zeros(S)
  for i in range(S):
    denom = 0
    for j in range(S):
      denom += a[i, j] * tau[i, j]**(1-sigma) * (w[j]/A[j])**(1-sigma)
    for j in range(S):
      Z[i] += (a[i, j] * tau[i, j]**(1-sigma) * (w[j]/A[j])**(1-sigma) / denom) * w[j] * L[j]
    Z[i] -= L[i]
  return Z

# Example usage
S = 4  # Number of goods
sigma = 3  # Elasticity of substitution
a = np.eye(4)+1
# trade costs
tau = np.array([[1, 1.1, 1.2, 1.3],
              [1.3, 1, 1.3, 1.4],
              [1.2, 1.2, 1, 1.1],
              [1.1, 1.1, 1.1, 1]])


L = np.array([2, 1, 1, 1])
A = np.ones(S) * 0.6

# modifications to check old hw
tau = np.array([[1, 1.2, 1.3, 1.2],
        [1.1, 1, 1.4, 1.1],
        [1.3, 1.4, 1, 1.2],
        [1.2, 1.4, 1.2, 1]])
L = np.array([1, 1, 1, 1])
A = np.array([1, .7, .7, .7])
sigma = 3 
A = np.ones(4)

#Z = excess_demand(a, tau, w, L, sigma, A)
#print(Z)
def excess_demand(w):
  return excess_demand_vars(a, tau, w, L, sigma, A)


from scipy.optimize import root, least_squares

def approximate_equilibrium(excess_demand, initial_wages):
  """
  This function approximates the solution where excess_demand is close to zero.

  Args:
      excess_demand: A function that takes a vector of wages and returns a vector of excess demand.
      initial_wages: An initial guess for the vector of wages.

  Returns:
      A vector of wages that approximates the equilibrium point.
  """
  # Define a function that takes wages and returns the negative of excess demand.
  # This is because we want to find the root (zero) of the excess demand function.
  def negative_excess_demand(wages):
    return -excess_demand(wages)

  # Use the root function from scipy.optimize to find the root of the negative excess demand function.
  result = root(negative_excess_demand, initial_wages, method='lm', options={'maxiter': 1000})

  # Return the solution (wages) from the root finding process.
  return result.x

# Initial guess for wages (adjust as needed)
initial_wages = [1, 1.1, 1.2, 1.3]

# Find the approximate equilibrium wages
equilibrium_wages = approximate_equilibrium(excess_demand, initial_wages)

# Print the equilibrium wages
print("Equilibrium wages: ", equilibrium_wages)
print("Excess demand: ", excess_demand(equilibrium_wages))
