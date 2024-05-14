import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    """
    Generates synthetic data for linear regression.

    Returns:
        X1 (numpy.ndarray): Transposed matrix of features (including a column of ones).
        Y1 (numpy.ndarray): Transposed vector of target values.
        x (list): Original scaled x values.
        y (list): Corresponding y values.
    """
    x = []
    y = []
    x1 = [[], []]
    for i in range(-500, 500):
        x_val = i / 1000 # Scale the x value
        x.append(x_val) # Append scaled value to x list
        x1[0].append(1) 
        x1[1].append(i) # Append original value to second list in 'x1'
        n = np.random.normal(0, 5) # Generate a random noise value
        y_val = 2 * x_val - 3 + n # Calculate the corresponding y value
        y.append(y_val) # Append the y value to the y list
    return np.transpose(x1), np.transpose(y), x, y # Convert lists to numpy arrays for matrix operations

def calculate_coefficients(X1, Y1):
    """
    Calculates regression coefficients using the Normal Equation method.

    Args:
        X1 (numpy.ndarray): Transposed matrix of features (including a column of ones).
        Y1 (numpy.ndarray): Transposed vector of target values.

    Returns:
        B0 (float): Intercept coefficient.
        B1 (float): Slope coefficient.
    """
    # Calculate coefficients using the Normal Equation method
    b = np.matmul(np.linalg.inv(np.matmul(X1.T, X1)), np.matmul(X1.T, Y1))
    B0 = b[0]
    B1 = b[1]
    return B0, B1

def gradient_descent(x, y, lr=0.001, threshold=10e-6):
    """
    Performs gradient descent optimization to find regression coefficients.

    Args:
        x (list): Original scaled x values.
        y (list): Corresponding y values.
        lr (float, optional): Learning rate. Defaults to 0.001.
        threshold (float, optional): Convergence threshold. Defaults to 10e-6.

    Returns:
        b0 (float): Intercept coefficient.
        b1 (float): Slope coefficient.
        error (float): Final mean squared error.
        epoch (int): Number of epochs.
        Epoch (list): List of epoch numbers.
        E (list): List of mean squared errors during training.
    """
    # Initialize coefficients for Gradient Descent
    b0f = np.random.normal(0, 1)
    b1f = np.random.normal(0, 1)
    # Calculate initial error
    errorf = np.mean((y - (b0f + b1f * np.array(x)))**2)
    # Initialize variables for Gradient Descent
    error = errorf
    b0 = b0f
    b1 = b1f
    epoch = 0
    
    # Lists to store the progress of the Gradient Descent
    Epoch = [0]
    E = [errorf]
    
    flag = 0
    while not flag:
        y_pred = b0 + b1 * np.array(x)
          # Calculate gradients for each data point
        db0 = [-2 * (y[i] - (b0 + b1 * x[i])) for i in range(len(x))]
        db1 = [-2 * (y[i] - (b0 + b1 * x[i])) * x[i] for i in range(len(x))]
        
        grad_b0 = np.mean(db0)
        grad_b1 = np.mean(db1)
        
        # Update coefficients
        b0 -= lr * grad_b0
        b1 -= lr * grad_b1
        
        # Calculate new error
        new_error = np.mean([(y[i] - (b0 + b1 * x[i]))**2 for i in range(len(x))])
        
         # Update epoch count and error history
        epoch += 1
        Epoch.append(epoch)
        E.append(new_error)
        
        # Check for convergence
        if abs(error - new_error) < threshold:
            flag = 1
        else:
            error = new_error
    
    return b0, b1, error, epoch, Epoch, E

# Main execution
X1, Y1, x, y = generate_data()
B0, B1 = calculate_coefficients(X1, Y1)
b0, b1, error, epoch, Epoch, E = gradient_descent(x, y)

# Output the results
print("Closed Form: b0:", B0, "b1:", B1, "error:", error)
print("Gradient Descent: b0:", b0, "b1:", b1, "error:", error, "epoch:", epoch)

# Plot the error convergence
plt.figure(figsize=(8, 4))
plt.plot(Epoch, E)
plt.xlabel('Number of Epoch Cycles')
plt.ylabel('Mean Squared Error')
plt.title('Gradient Descent Convergence')
plt.figtext(0.5, 0.01, "The graph shows the Mean Squared Error decreasing over the number of epochs.", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.show()