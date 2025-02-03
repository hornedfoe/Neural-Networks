import matplotlib.pyplot as plt
import sys

# Update the file path with the correct path to your input file

file_path = sys.argv[1]
output_file_path = sys.argv[2]
title = sys.argv[3]

# Read data from the input file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Extract x and y values from the file
x_values = []
y_values = []
for line in lines:
    data = line.split()
    if data[0] == 'Test':
        break
    elif data[0] == 'Train':
        continue
    x_values.append(float(data[0][1:-1]))  # Remove the brackets and convert to float
    y_values.append(float(data[1][1:-1]))   # Remove the brackets and convert to float

# Plot the data using Matplotlib
plt.scatter(x_values, y_values, label='Data Points')
plt.title(title)
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.savefig(output_file_path)
