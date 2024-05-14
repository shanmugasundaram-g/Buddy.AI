import numpy as np
import matplotlib.pyplot as plt

# Initialize a list 'x' with sublists to store powers of x values
x = [[1]*101, [], [], [], []]  # The first sublist is filled with 101 ones for the intercept term

# Initialize lists for storing the generated x and y values
x1 = []  # List to store the scaled x values
y = []   # List to store the corresponding y values

# Generate data points for x and y
for i in range(-50, 51, 1):
    i = i / 10  # Scale the x value
    x1.append(i)  # Append the scaled x value to 'x1'
    # Append powers of the scaled x value to the respective sublists in 'x'
    x[1].append(i)
    x[2].append(i**2)
    x[3].append(i**3)
    x[4].append(i**4)
    n = np.random.normal(0, 3)  # Generate a random noise value
    # Calculate the corresponding y value using a polynomial function and add noise
    fx = ((2*(i**4)) - (3*(i**3)) + (7*(i**2)) - (23*i) + 8 + n)
    y.append(fx)

# Convert the list 'x' to a numpy array and transpose it for matrix operations
X = np.transpose(x)
Y = np.transpose(y)

# Compute the coefficients 'b' using the Normal Equation method
b = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))

# Print the computed coefficients
print(b)

# Initialize lists to store the function values for different models
y1 = []  # Linear model
y2 = []  # Quadratic model
y3 = []  # Cubic model
y4 = []  # Biquadratic model

# Compute the function values for each model
for i in x1:
    # Linear function
    f1 = b[0] + b[1]*i
    y1.append(f1)
    # Quadratic function
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    y2.append(f2)
    # Cubic function
    f3 = b[0] + b[1]*i + b[2] * (i**2) + b[3]*(i**3)
    y3.append(f3)
    # Biquadratic function
    f4 = b[0] + b[1]*i + b[2] * (i**2) + b[3]*(i**3) + b[4]*(i**4)
    y4.append(f4)

# Define the Lagrange Interpolation function
def lagrangeInterpolation(x, y, xInterp):
    n = len(x)  # Number of data points
    m = len(xInterp)  # Number of interpolation points
    yInterp = np.zeros(m)  # Initialize the list for interpolated y values
    
    # Compute the interpolated y values using Lagrange's formula
    for j in range(m):
        p = 0
        for i in range(n):
            L = 1
            for k in range(n):
                if k != i:
                    L *= (xInterp[j] - x[k]) / (x[i] - x[k])
            p += y[i] * L
        yInterp[j] = p
    return yInterp

# Compute the interpolated y values using the original x values
yInte = lagrangeInterpolation(x1, y, x1)

# Plot all the functions
plt.figure(figsize=(8, 4))
plt.plot(x1, y1, label="linear")
plt.plot(x1, y2, label="quadratic")
plt.plot(x1, y3, label="cubic")
plt.plot(x1, y4, label="biquadratic")
plt.plot(x1, yInte, label="Lagrange")
plt.xlabel('X')
plt.ylabel('Y = F(X)')
plt.title("Different Models and Their Plots")
plt.figtext(0.5, 0.01, "The graph shows the generated y values for different models (linear, quadratic, cubic, biquadratic) and the Lagrange interpolation based on a biquadratic polynomial with 101 x values in the range (-5, 5).", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.legend()
# plt.show()  # Uncomment to display the plot

# Split the data into training and testing sets
# 80% for training
X1 = x1[:81]
np.random.shuffle(X1)  # Shuffle the training data
# 20% for testing
X2 = x1[81:]
np.random.shuffle(X2)  # Shuffle the testing data

# Initialize lists to store the function values for the training set
y1_train = []  # Linear model
y2_train = []  # Quadratic model
y3_train = []  # Cubic model
y4_train = []  # Biquadratic model

# Compute the function values for the training set
for i in X1:
    # Linear function
    f1 = b[0] + b[1]*i
    y1_train.append(f1)
    # Quadratic function
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    y2_train.append(f2)
    # Cubic function
    f3 = b[0] + b[1]*i + b[2] * (i**2) + b[3]*(i**3)
    y3_train.append(f3)
    # Biquadratic function
    f4 = b[0] + b[1]*i + b[2] * (i**2) + b[3]*(i**3) + b[4]*(i**4)
    y4_train.append(f4)

# Compute the squared error for the training set
e1train = []  # Linear model
e2train = []  # Quadratic model
e3train = []  # Cubic model
e4train = []  # Biquadratic model
com=[1,2,3,4]
for i in range(81):
    e1train.append((y1_train[i] - y[i])**2)
    e2train.append((y2_train[i] - y[i])**2)
    e3train.append((y3_train[i] - y[i])**2)
    e4train.append((y4_train[i] - y[i])**2)

# Calculate the mean squared error for the training set
Etrain = [
    sum(e1train) / len(y1_train),
    sum(e2train) / len(y2_train),
    sum(e3train) / len(y3_train),
    sum(e4train) / len(y4_train)
]

# Initialize lists to store the function values for the testing set
ytest1 = []  # Linear model
ytest2 = []  # Quadratic model
ytest3 = []  # Cubic model
ytest4 = []  # Biquadratic model

# Extract the original y values for the testing set
Yorig = Y[81:]

# Compute the function values for the testing set
for i in X2:
    # Linear function
    f1 = b[0] + b[1]*i
    ytest1.append(f1)
    # Quadratic function
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    ytest2.append(f2)
    # Cubic function
    f3 = b[0] + b[1]*i + b[2] * (i**2) + b[3]*(i**3)
    ytest3.append(f3)
    # Biquadratic function
    f4 = b[0] + b[1]*i + b[2] * (i**2) + b[3]*(i**3) + b[4]*(i**4)
    ytest4.append(f4)

# Compute the squared error for the testing set
e1test = []  # Linear model
e2test = []  # Quadratic model
e3test = []  # Cubic model
e4test = []  # Biquadratic model
for i in range(len(ytest1)):
    e1test.append((ytest1[i] - Yorig[i])**2)
    e2test.append((ytest2[i] - Yorig[i])**2)
    e3test.append((ytest3[i] - Yorig[i])**2)
    e4test.append((ytest4[i] - Yorig[i])**2)

# Calculate the mean squared error for the testing set
Etest = [
    sum(e1test) / len(ytest1),
    sum(e2test) / len(ytest2),
    sum(e3test) / len(ytest3),
    sum(e4test) / len(ytest4)
]

# Plot the error values for the training and testing sets
plt.figure(figsize=(8, 4))
plt.plot(com, Etest, label="Variance")
plt.plot(com, Etrain, label="Bias")
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.title("Bias-Variance Tradeoff")
plt.figtext(0.5, 0.01, "The above graph shows the bias variance tradeoff for models such as linear,quadratic,cubic and biquadratic models", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})
plt.legend()
plt.show()