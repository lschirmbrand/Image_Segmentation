
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
 
# Creating vectors X and Y
x = np.linspace(0, 20, 10)
y = (x ** 2) 
 
fig = plt.figure(figsize = (10, 5))
# Create the plot

plt.ylabel("Amount of Pixel")
plt.xlabel("Dimension of Image")
plt.plot(x, y)
 
# Show the plot
plt.show()