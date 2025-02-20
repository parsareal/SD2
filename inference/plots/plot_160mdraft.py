import matplotlib.pyplot as plt
import numpy as np

# Data
target_models = ['Vicuna 7b', 'Vicuna 13b', 'LLaMA 2 Chat 70b']
spc = [0, 1.07, 0]
spc_soft = [0, 1.05, 0]
spc_soft_mid = [0, 0, 0]
spc_soft_lowest = [0, 0, 0]
s3_soft = [0, 1.12, 0]

# Number of target_models
N = len(target_models)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12,6))

# Positions of the bars on the x-axis
ind = np.arange(N)

# Width of the bars
width = 0.15

# Plotting
bars1 = ax.bar(ind - 2*width, spc, width, label='Speculative')
bars2 = ax.bar(ind - width, spc_soft, width, label='Speculative - SoFT Checkpoint Layer 12')
bars3 = ax.bar(ind, spc_soft_mid, width, label='Speculative - SoFT Checkpoint Layer 9', color='none', edgecolor='red', hatch='//')
bars4 = ax.bar(ind + width, spc_soft_lowest, width, label='Speculative - SoFT Checkpoint Layer 6', color='none', edgecolor='black', hatch='//')
bars5 = ax.bar(ind + 2*width, s3_soft, width, label='Sorted Speculative Sampling (S3)')

# Adding labels and title
ax.set_xlabel('Target Models')
ax.set_ylabel('Speed-up Ratio')
ax.set_title('Different Inference Acceleration Algorithms Comparison. \n Temp=1.0, Draft = Vicuna 160m SFT and SoFT (submodels 6, 9, 12) trained for 1 epoch')
ax.set_xticks(ind)
ax.set_xticklabels(target_models)
ax.legend()

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{height:.2f}x' if height != 0 else f'N/A',  # Formatting as float with one decimal place 
            ha='center', 
            va='bottom'
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)
add_labels(bars5)


# Show the plot
plt.savefig('./160mdraft.png')
