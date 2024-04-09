import numpy as np
from scipy.optimize import root, least_squares
from functools import partial



# Calculate the absolute trade
def calculate_trade(w, a, tau, sigma, A):
  trade = np.zeros([w.shape[0],w.shape[0]])
  S = w.shape[0]
  for i in range(S):
    for j in range(S):
      top = a[i, j] * tau[i, j]**(1-sigma)* (w[i]/A[i])**(1-sigma)
      bottom = 0
      for k in range(S):
        bottom += a[k, j] * tau[k, j]**(1-sigma)* (w[k]/A[k])**(1-sigma)
    
      trade[i,j] = top/bottom
  return(trade)


def calculate_welfare(w, a, tau, sigma, A):
  S = w.shape[0]
  trade_matrix = calculate_trade(w, a, tau, sigma, A)
  welfare = np.zeros(S)
  for i in range(S):
    welfare[i] = trade_matrix[i,i]**(1/(1-sigma)) * a[i, i]**(1/(sigma-1)) * A[i]
  return welfare

def excess_demand(w, a, tau, L, sigma, A):
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

  S = len(a)  
  Z = np.zeros(S)
  for i in range(S):
    summation = 0
    trade_matrix = calculate_trade(w, a, tau, sigma, A)
    for j in range(S):
      summation += trade_matrix[i,j] * w[j] * L[j] / w[i]

    Z[i] = summation - L[i]
  return Z


def approximate_equilibrium(excess_demand, initial_wages, a, tau, L, sigma, A):
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
  def negative_excess_demand(w, a, tau, L, sigma, A):
    return -excess_demand(w, a, tau, L, sigma, A)
  

  # Use the root function from scipy.optimize to find the root of the negative excess demand function.
  partial_neg_excess_demand = partial(negative_excess_demand, a=a, tau=tau, L=L, sigma=sigma, A=A)

  result = root(partial_neg_excess_demand, initial_wages, method='lm', options={'maxiter': 1000})

  # Return the solution (wages) from the root finding process.
  return result.x

def calculate_lambda_hat(tau_new, tau_old, lambda_old,w_new,w_old,sigma):
  S = len(tau_new)
  lambda_hat = np.zeros([S,S])
  for i in range(S):
    for j in range(S):
      numerator[i,j] = tau_new[i,j]/tau_old[i,j] * (w_new[i]/w_old[i]) ** (1-sigma)
      denominator = 0
      for k in range(S):
        denominator += lambda_old[k,j] * tau_new[k,j]/tau_old[k,j] * (w_new[k]/w_old[k]) ** (1-sigma)
      lambda_hat[i,j] = numerator[i,j]/denominator
  return (lambda_hat)

# declare parameters #
S = 4  # Number of goods

# 2024 parameters
sigma24 = 3  # Elasticity of substitution
a24 = np.eye(S)+1
# trade costs
tau24 = np.array([[1, 1.1, 1.2, 1.3],
              [1.3, 1, 1.3, 1.4],
              [1.2, 1.2, 1, 1.1],
              [1.1, 1.1, 1.1, 1]])


L24 = np.array([2, 1, 1, 1])
A24 = np.ones(S) * 0.6

A24shock = A24.copy()
A24shock[1] = 1.2

#2nd part parameters
tau24_new = np.array([[1 , 1 , 1.2 , 1.2],
            [1 , 1 , 1.2 , 1.2],
            [1 , 1.2 , 1 , 1.3],
            [1 , 1.2 , 1.2 , 1]])

#paramters to check against last years
sigma23 = 4
L23 = np.ones(S)
A23 = np.ones(S)*.6
A23[0] = 1
a23 = np.ones([S,S])
tau23 = np.array([[1, 1.1, 1.2, 1.1],
        [1.4, 1, 1.3, 1.4],
        [1.2, 1.3, 1, 1.1],
        [1.1, 1.3, 1.1, 1]])

# last years productivity shock
A23shock = A23.copy()
A23shock[1] = 1.2

# 2nd part parameters
tau23_new = np.array([[1, 1, 1.2, 1.2],
            [1, 1, 1.2, 1.2],
            [1, 1.2, 1, 1.3],
            [1, 1.2, 1.2, 1]])
def print_results(a, tau, L, sigma, A):
  #Initial guess for wages (adjust as needed)
  initial_wages = [1, .8,.76, .77]
  # Find the approximate equilibrium wages
  equilibrium_wages = approximate_equilibrium(excess_demand, initial_wages, a, tau, L, sigma, A)
  # normalize wages relative to first index
  normalized_wages = equilibrium_wages/(equilibrium_wages[0])

  print("Equilibrium wages: ", np.around(equilibrium_wages,4))
  print("Normalized Equilibrium wages: ", np.around(normalized_wages,4))
  print("Excess demand: ", excess_demand(normalized_wages, a, tau, L, sigma, A))
  print("Lambda: \n", np.around(calculate_trade(equilibrium_wages,a, tau, sigma, A),4))
  print("Welfare: ", np.around(calculate_welfare(equilibrium_wages,  a, tau, sigma, A),4))
        
print("2024")
print_results(a24, tau24, L24, sigma24, A24)
print("2024: Productivity Shock")
print_results(a24, tau24, L24, sigma24, A24shock)
print("2024: New Trade Costs")
print_results(a24, tau24_new, L24, sigma24, A24)

print("2023")
print_results(a23, tau23, L23, sigma23, A23)
print("2023: Productivity Shock")
print_results(a23, tau23, L23, sigma23, A23shock)
print("2023: New Trade Costs")
print_results(a23, tau23_new, L23, sigma23, A23)

