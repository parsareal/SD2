import matplotlib.pyplot as plt

# Data
x = [1.5, 2, 2.5]
y1 = [1.34, 1.24, 1.20]
y2 = [1.38, 1.35, 1.32]
y3 = [1.92, 1.94, 1.93]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y1, '--', marker='^', label='Vicuna 7b')
plt.plot(x, y2, '--', marker='^', label='Vicuna 13b')
plt.plot(x, y3, '--', marker='^', label='LLaMA Chat 70b')

# Adding labels
plt.xlabel('Thresholds')
plt.ylabel('Speedup Ratio')
plt.xticks(ticks=x, labels=['0.5, 0.5, 0', '0.75, 0.7, 0', '0.9, 0.85, 0'])

# Adding a legend
plt.legend()

# Annotating points
for i in range(len(x)):
    plt.annotate(f'{y1[i]}x', (x[i], y1[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.annotate(f'{y2[i]}x', (x[i], y2[i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.annotate(f'{y3[i]}x', (x[i], y3[i]), textcoords="offset points", xytext=(0,5), ha='center')


# Title
plt.title('Speedup Ratios based on Different Confidence Thresholds')

# Save the figure
plt.savefig('line_chart.png')
