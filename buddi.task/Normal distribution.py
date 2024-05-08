import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#import lance
#from  import LoadStoryBoard, StoryBoard
def normal_distribution(x, mu, sigma):
    
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Example usage
x_values = np.linspace(-3, 3, 1000)  # Generate 100 points between -3 and 3
mu_values = [1]  # Variable mean values
sigma_values = [1, 2,3]  # Variable standard deviation values

# Plotting
plt.figure(figsize=(10, 6))
for mu in mu_values:
    for sigma in sigma_values:
        y = normal_distribution(x_values, mu, sigma)
        label = f"μ={mu}, σ={sigma}"
        plt.plot(x_values, y, marker='o', label=label)

plt.xlabel('x distribution point')
plt.ylabel('Normal Distribution Value')
plt.title('Normal Distribution Function for Different Parameters')
plt.legend()
plt.grid(True)
plt.figtext(0.01,0.01,"Normal distribution curve with  fixed mean and variant standard deviation")
plt.show()
plt.close()

# Example usage
x_values = np.linspace(-3, 3, 1000)  # Generate 100 points between -3 and 3
mu_values1 = [1,2,3]  # Variable mean values
sigma_values1 = [1]  # Variable standard deviation values

# Plotting
plt.figure(figsize=(10, 6))
for mu in mu_values1:
    for sigma in sigma_values1:
        y = normal_distribution(x_values, mu, sigma)
        label = f"μ={mu}, σ={sigma}"
        plt.plot(x_values, y, marker='o', label=label)

plt.xlabel('x distribution point')
plt.ylabel('Normal Distribution Value')
plt.title('Normal Distribution Function for Different Parameters')
plt.legend()
plt.grid(True)
plt.figtext(0.01,0.01,"Normal distribution curve with variant of mean and fixed standard deviation")
plt.show()