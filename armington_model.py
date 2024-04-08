import numpy as np
from scipy.optimize import root, least_squares

# declare parameters #

# first part paramaters
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
A[1] = 1.2

#2nd part parameters
tau = np.array([[1 , 1 , 1.2 , 1.2],
            [1 , 1 , 1.2 , 1.2],
            [1 , 1.2 , 1 , 1.3],
            [1 , 1.2 , 1.2 , 1]])

# #paramters to check against last years
# S = 4
# sigma = 4
# L = np.ones(S)
# A = np.ones(S)*.6
# A[0] = 1
# a = np.ones([S,S])
# tau = np.array([[1, 1.1, 1.2, 1.1],
#         [1.4, 1, 1.3, 1.4],
#         [1.2, 1.3, 1, 1.1],
#         [1.1, 1.3, 1.1, 1]])

# # last years productivity shock
# A[1] = 1.2

# Calculate the absolute trade
def calculate_trade(w):
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

# def calculate_trade_shares(w):
#   S = w.shape[0]
#   trade_matrix = calculate_trade(w)
#   trade_shares = np.zeros([S,S])
#   for i in range(S):
#     for j in range(S):
#       trade_shares[i,j] = trade_matrix[i,j] / w[j]*L[j]
#   return trade_shares

# def calculate_lambda(w):
#     S = w.shape[0]
#     trade_shares = calculate_trade_shares(w)
#     lambda_matrix = np.zeros([S,S])
#     for i in range(S):
#         for j in range(S):
#             lambda_matrix[i,j] = trade_shares[i,j] / (w[j]*L[j])
#     return lambda_matrix

def calculate_welfare(w):
  S = w.shape[0]
  trade_matrix = calculate_trade(w)
  welfare = np.zeros(S)
  for i in range(S):
    welfare[i] = trade_matrix[i,i]**(1/(1-sigma)) * a[i, i]**(1/(sigma-1)) * A[i]
  return welfare

def excess_demand(w): #a, tau, w, L, sigma, A):
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
    summation = 0
    # for j in range(S):
    #   top = a[i, j] * tau[i, j]**(1-sigma)* (w[i]/A[i])**(1-sigma)
    #   bottom = 0
    #   for k in range(S):
    #     bottom += a[k, j] * tau[k, j]**(1-sigma)* (w[k]/A[k])**(1-sigma)
      
    #   top = top * w[j] * L[j]
    #   bottom = bottom * w[i] 
    #   summation += top/bottom
    trade_matrix = calculate_trade(w)
    for j in range(S):
      summation += trade_matrix[i,j] * w[j] * L[j] / w[i]

    Z[i] = summation - L[i]
  return Z


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

#Initial guess for wages (adjust as needed)
initial_wages = [1, .8,.76, .77]

# Find the approximate equilibrium wages
equilibrium_wages = approximate_equilibrium(excess_demand, initial_wages)
# normalize wages relative to first index
normalized_wages = equilibrium_wages/(equilibrium_wages[0])

#Print the equilibrium wages
print("Equilibrium wages: ", np.around(normalized_wages,4))
print("Excess demand: ", excess_demand(normalized_wages))

print("Bilateral Trade Shares: \n", np.around(calculate_trade(normalized_wages),4))

print("Welfare: ", np.around(calculate_welfare(normalized_wages),4))