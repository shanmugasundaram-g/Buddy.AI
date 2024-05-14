import numpy as np
import matplotlib.pyplot as plt
txt="This is the plot that shows the Lagrange's polynomial"

def lagrange(X, Y, xi, n): # lagrange function
    res=0.0
    for i in range(n):
        t=Y[i]
        for j in range(n):
            if(j!=i):
                t=t*(xi-X[j])/(X[i]-X[j])
        res+=t
    return res

# Reading number of unknowns
n = int(input('Enter number of data points: '))

# Making numpy array of n & n x n size and initializing 
# to zero for storing x and y value along with differences of y
x = np.zeros((n))
y = np.zeros((n))


# Reading data points
print('Enter data for x and y: ')
for i in range(n):
    x[i] = float(input( 'x['+str(i)+']='))
    y[i] = float(input( 'y['+str(i)+']='))

Y_cap=[lagrange(x,y,x[i],n) for i in range(n)]
print(Y_cap)

#plotting interpolitan points with respect to data points
plt.title("Applying Lagrange's polynomial ")
plt.scatter(x,y, marker="x", label="Original data points")
plt.plot(x,Y_cap, color="purple", label="Lagrange's polynomial ")
plt.figtext(0.2, 0.01, txt, wrap=True, fontsize=8)
plt.legend()
plt.show()