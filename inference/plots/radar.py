import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Data
categories = ['Multi-turn \n Conversation', 'Translation', 'Summarization', 'Question Answering', 'Math Reasoning', 'Retrieval-augmented \n Reasoning']
N = len(categories)

# # Vicuna 7b - draft 7b - tmp = 0
vicuna_160m_values = [1.05, 0.78, 1.06, 0.93, 1.08, 1.04]
spc_sft_values = [1.19, 0.93, 1.10, 0.93, 1.13, 1.15]
spc_soft_values = [1.17, 0.89, 1.08, 0.99, 1.13, 1.18]
spc_soft_mid_values = [1.24, 0.92, 1.26, 1.05, 1.24, 1.22]
spc_soft_lowest_values = [1.38, 1.10, 1.29, 1.15, 1.31, 1.27]
s3_soft_values = [1.34, 1.20, 1.23, 1.03, 1.28, 1.20]
spc_sft_ee_l6 = [0.98, 0.88, 0.88, 0.93, 0.98, 0.94]
spc_sft_ee_l9 = [0.99, 0.82, 0.94, 0.92, 0.97, 0.98]

# # # Vicuna 13b - draft 7b - tmp = 0
# vicuna_160m_values = [1.13, 0.79, 1.14, 1.10, 1.11, 1.16]
# spc_sft_values = [1.21, 0.98, 1.20, 1.05, 1.24, 1.22]
# spc_soft_values = [1.23, 0.97, 1.22, 1.05, 1.25, 1.20]
# spc_soft_mid_values = [1.31, 0.96, 1.33, 1.16, 1.34, 1.30]
# spc_soft_lowest_values = [1.38, 0.96, 1.23, 1.21, 1.36, 1.25]
# s3_soft_values = [1.38, 0.97, 1.33, 1.14, 1.38, 1.26]
# spc_sft_ee_l6 = [1.00, 0.86, 0.90, 0.95, 1.09, 1.05]
# spc_sft_ee_l9 = [1.05, 0.84, 0.99, 0.96, 1.03, 1.02]

# # # LM Chat 70b - draft 7b - tmp = 0
# vicuna_160m_values = [1.93, 1.80, 1.78, 1.80, 1.85, 2.03]
# spc_sft_values = [1.94, 1.73, 1.73, 1.66, 1.82, 1.95]
# spc_soft_values = [1.91, 1.69, 1.75, 1.65, 1.81, 1.89]
# spc_soft_mid_values = [1.92, 1.70, 1.76, 1.68, 1.81, 1.93]
# spc_soft_lowest_values = [1.83, 1.67, 1.67, 1.65, 1.77, 1.75]
# s3_soft_values = [1.95, 1.72, 1.78, 1.69, 1.84, 1.94]
# spc_sft_ee_l6 = [1.32, 1.31, 1.27, 1.32, 1.31, 1.23]
# spc_sft_ee_l9 = [1.54, 1.46, 1.44, 1.46, 1.49, 1.47]

# Vicuna 7b - draft 7b - tmp = 1
# vicuna_160m_values = [0.96, 0.74, 0.97, 0.89, 0.96, 0.96]
# spc_sft_values = [1.04, 0.89, 1.04, 0.92, 1.11, 1.09]
# spc_soft_values = [1.02, 0.85, 1.03, 0.88, 1.03, 1.13]
# spc_soft_mid_values = [1.13, 0.86, 1.14, 0.96, 1.29, 1.13]
# spc_soft_lowest_values = [1.19, 0.94, 1.13, 1.05, 1.23, 1.17]
# s3_soft_values = [1.27, 0.92, 1.21, 1.07, 1.33, 1.24]

# # Vicuna 13b - draft 7b - tmp = 1
# vicuna_160m_values = [1.10, 0.79, 1.11, 1.00, 1.07, 1.08]
# spc_sft_values = [1.10, 0.91, 1.13, 0.94, 1.12, 1.06]
# spc_soft_values = [1.10, 0.82, 1.13, 0.97, 1.23, 1.12]
# spc_soft_mid_values = [1.23, 0.87, 1.21, 0.98, 1.19, 1.15]
# spc_soft_lowest_values = [1.24, 0.89, 1.16, 1.10, 1.29, 1.13]
# s3_soft_values = [1.38, 0.95, 1.32, 1.13, 1.39, 1.31]

# Vicuna 70b - draft 7b - tmp = 1
# vicuna_160m_values = [1.67, 1.53, 1.61, 1.60, 1.62, 1.77]
# spc_sft_values = [1.74, 1.56, 1.67, 1.51, 1.73, 1.70]
# spc_soft_values = [1.69, 1.54, 1.60, 1.49, 1.61, 1.75]
# spc_soft_mid_values = [1.64, 1.53, 1.60, 1.50, 1.66, 1.66]
# spc_soft_lowest_values = [1.58, 1.45, 1.47, 1.47, 1.55, 1.51]
# s3_soft_values = [1.98, 1.74, 1.79, 1.69, 1.92, 1.97]



'''
    Greedy Draft with temperature = 1.0
'''
# Vicuna 7b - draft 7b - tmp = 1
vicuna_160m_values = [1.02, 0.79, 1.05, 0.98, 1.02, 1.08]
spc_sft_values = [1.07, 0.88, 1.05, 0.96, 1.11, 1.07]
spc_soft_values = [1.02, 0.85, 1.03, 0.88, 1.03, 1.13]
spc_soft_mid_values = [1.26, 0.92, 1.19, 1.06, 1.22, 1.21]
spc_soft_lowest_values = [1.30, 0.97, 1.19, 1.19, 1.31, 1.27]
s3_soft_values = [1.27, 0.92, 1.21, 1.07, 1.33, 1.24]
spc_sft_ee_l6 = [0.99, 0.86, 0.87, 0.94, 0.90, 1.06]
spc_sft_ee_l9 = [0.98, 0.83, 0.99, 0.94, 0.92, 1.06]

# # # Vicuna 13b - draft 7b - tmp = 1
# vicuna_160m_values = [1.07, 0.72, 1.18, 1.07, 1.14, 1.18]
# spc_sft_values = [1.10, 0.88, 1.10, 1.01, 1.11, 1.03]
# spc_soft_values = [1.20, 0.95, 1.27, 1.05, 1.27, 1.18]
# spc_soft_mid_values = [1.32, 0.94, 1.35, 1.14, 1.35, 1.28]
# spc_soft_lowest_values = [1.37, 0.95, 1.23, 1.04, 1.20, 1.05]
# s3_soft_values = [1.38, 0.95, 1.32, 1.13, 1.39, 1.31]
# spc_sft_ee_l6 = [1.03, 0.87, 0.92, 0.93, 0.96, 0.96]
# spc_sft_ee_l9 = [1.12, 0.86, 1.02, 0.95, 1.07, 1.04]

# # # Vicuna 70b - draft 7b - tmp = 1
# vicuna_160m_values = [2.01, 1.76, 1.84, 1.87, 1.91, 2.10]
# spc_sft_values = [1.94, 1.73, 1.81, 1.68, 1.93, 2.03]
# spc_soft_values = [1.96, 1.68, 1.81, 1.69, 1.96, 1.96]
# spc_soft_mid_values = [1.94, 1.67, 1.79, 1.71, 1.83, 1.96]
# spc_soft_lowest_values = [1.87, 1.68, 1.64, 1.65, 1.81, 1.78]
# s3_soft_values = [1.98, 1.74, 1.79, 1.69, 1.92, 1.97]
# spc_sft_ee_l6 = [1.37, 1.33, 1.26, 1.33, 1.33, 1.27]
# spc_sft_ee_l9 = [1.57, 1.45, 1.43, 1.48, 1.53, 1.50]


# values = [vicuna_160m_values, spc_sft_values, spc_sft_ee_l6, spc_sft_ee_l9, spc_soft_values, spc_soft_mid_values, spc_soft_lowest_values, s3_soft_values]
values = [spc_sft_values, spc_sft_ee_l6, spc_sft_ee_l9, spc_soft_values, spc_soft_mid_values, spc_soft_lowest_values, s3_soft_values]

# labels = ['Vicuna 160m', 'SFT+SD', 'SFT+EE L6+SD', 'SFT+EE L9+SD', 'SoFT L12+SD', 'SoFT L9+SD', 'SoFT L6+SD', 'SoFT+S2D']
labels = ['SFT+SD', 'SFT+EE L6+SD', 'SFT+EE L9+SD', 'SoFT L12+SD', 'SoFT L9+SD', 'SoFT L6+SD', 'SoFT+S2D']

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
# colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b', '#e377c2']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b']
colors = ['#1f77b4', '#e377c2', '#2ca02c', '#d62728', '#ff7f0e', '#17becf', '#9467bd']


# Create the radar chart
fig, ax = plt.subplots(figsize=(13, 12), subplot_kw=dict(polar=True))

# Angle of each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], categories, color='black', size=25)

# Draw ylabels
# ax.set_rlabel_position(0)
# plt.yticks([0.5, 1.0, 1.5], ["0.5", "1.0", "1.5"], color="black", size=10)
plt.yticks([0.8, 1.0, 1.2, 1.4], ["0.8", "1.0", "1.2", "1.4"], color="black", size=23)
# plt.yticks([1.4, 1.6, 1.8, 2.0], ["1.4", "1.6", "1.8", "2.0"], color="black", size=23)
# plt.yticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.1], ["1.2", "1.4", "1.6", "1.8", "2.0", "2.1"], color="black", size=23)
# plt.yticks([1.4, 1.6, 1.8, 2.0], ["1.4", "1.6", "1.8", "2.0"], color="black", size=23)

# plt.yticks(size=22)
plt.ylim(0.7, 1.4)
# plt.ylim(1.5, 2.1)
# plt.ylim(1, 2.1)


# Plot each product
for i, values_set in enumerate(values):
    values_set += values_set[:1]
    ax.plot(angles, values_set, linewidth=2, linestyle='solid', label=labels[i], color=colors[i])
    ax.fill(angles, values_set, color=colors[i], alpha=.1)

# Add legend
# plt.legend(loc='upper right', bbox_to_anchor=(0.05, 0.1), fontsize=15)
plt.legend(loc='upper right', bbox_to_anchor=(0.08, 0.1), fontsize=17)

# Add title
plt.title('Vicuna 7b Target (Temperature=1.0)', size=30, color='black', y=1.05)
# plt.title('LLaMA2 Chat 70b Target (Temperature=1.0)', size=30, color='black', y=1.05)


# Display the radar chart
# plt.savefig('./test1.png')
plt.savefig('./radar_vicuna7b_7bDraft_tmp=1_final.pdf')
# plt.savefig('./radar_lmchat70b_7bDraft_tmp=1_final.pdf')


