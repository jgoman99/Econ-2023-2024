import numpy as np



def f(w):
  # Implement your function f here
  return w**2  # Example function

# Define your observed lambda
observed_lambda = 16

# Define the range for w (adjust these values)
w_min, w_max = 2, 5
w_values = np.linspace(w_min, w_max, 100)

# Find the w that gives the closest lambda
estimated_w = None
min_difference = np.inf
for w in w_values:
  current_lambda = f(w)
  difference = np.abs(current_lambda - observed_lambda)
  if difference < min_difference:
    min_difference = difference
    estimated_w = w

# Estimated lambda based on the closest w
estimated_lambda = f(estimated_w)

print("Estimated w:", estimated_w)
print("Estimated lambda:", estimated_lambda)
