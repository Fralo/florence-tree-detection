import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi

# 1. Setup the Data
# ---------------------------
# Replace the values in 'base_model' and 'augmented_model' with your actual data
data = {
    'Metric': ['Average IoU', 'Box Recall', 'Box Precision'],
    'Base Model': [0.273765004009733, 0.3194789918412347, 0.49238242360772],
    'Fine-tuned': [0.5642114309709385, 0.7403952245082723, 0.5165728166302054],
    'Fine-tuned + Aug': [0.5116293370000476, 0.6378175791541686, 0.5107377972335686]
}

df = pd.DataFrame(data)
df_melted = df.melt(id_vars="Metric", var_name="Model", value_name="Score")

# 2. Grouped Bar Chart
# ---------------------------
plt.figure(figsize=(10, 6))
# Define width of a bar and positions
barWidth = 0.25
r1 = np.arange(len(df['Metric']))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create bars
plt.bar(r1, df['Base Model'], width=barWidth, label='Base Model', color='#e0e0e0', edgecolor='grey')
plt.bar(r2, df['Fine-tuned'], width=barWidth, label='Fine-tuned', color='#4a90e2', edgecolor='grey')
plt.bar(r3, df['Fine-tuned + Aug'], width=barWidth, label='Fine-tuned + Aug', color='#003366', edgecolor='grey')

# Add specific labels
plt.xlabel('Metrics', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(df['Metric']))], df['Metric'])
plt.ylabel('Score')
plt.title('Comparison of Model Versions')
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save bar chart
plt.savefig('model_comparison_bar.png', bbox_inches='tight')

# 3. Radar Chart
# ---------------------------
# Prepare data for Radar
categories = list(df['Metric'])
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] # Close the loop

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
plt.ylim(0, 1)

# Plot Base Model
values_base = df['Base Model'].tolist()
values_base += values_base[:1]
ax.plot(angles, values_base, linewidth=1, linestyle='solid', label="Base Model", color='grey')
ax.fill(angles, values_base, 'grey', alpha=0.1)

# Plot Fine-tuned
values_ft = df['Fine-tuned'].tolist()
values_ft += values_ft[:1]
ax.plot(angles, values_ft, linewidth=2, linestyle='solid', label="Fine-tuned", color='#4a90e2')
ax.fill(angles, values_ft, '#4a90e2', alpha=0.1)

# Plot Augmented
values_aug = df['Fine-tuned + Aug'].tolist()
values_aug += values_aug[:1]
ax.plot(angles, values_aug, linewidth=2, linestyle='solid', label="Fine-tuned + Aug", color='#003366')
ax.fill(angles, values_aug, '#003366', alpha=0.1)

plt.title('Model Performance Footprint', y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

plt.savefig('model_comparison_radar.png', bbox_inches='tight')

# 4. Precision-Recall Scatter
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df.loc[1, 'Base Model'], df.loc[2, 'Base Model'], s=100, color='grey', label='Base Model')
plt.scatter(df.loc[1, 'Fine-tuned'], df.loc[2, 'Fine-tuned'], s=100, color='#4a90e2', label='Fine-tuned')
plt.scatter(df.loc[1, 'Fine-tuned + Aug'], df.loc[2, 'Fine-tuned + Aug'], s=100, color='#003366', label='Fine-tuned + Aug')

plt.text(df.loc[1, 'Base Model']+0.01, df.loc[2, 'Base Model'], 'Base', fontsize=9)
plt.text(df.loc[1, 'Fine-tuned']+0.01, df.loc[2, 'Fine-tuned'], 'Fine-tuned', fontsize=9)
plt.text(df.loc[1, 'Fine-tuned + Aug']+0.01, df.loc[2, 'Fine-tuned + Aug'], 'Augmented', fontsize=9)

plt.title('Precision vs Recall Trade-off')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.savefig('model_comparison_scatter.png', bbox_inches='tight')