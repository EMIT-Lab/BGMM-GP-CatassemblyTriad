import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Use more universal font settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 3,
    'text.usetex': False,
})

# Set figure dimensions (precisely 70mm × 50mm)
WIDTH_MM = 70
HEIGHT_MM = 50
fig, ax1 = plt.subplots(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4))

# Scientific plotting style settings - precisely meeting requirements
plt.rcParams.update({
    'axes.linewidth': 1.0,  # 1pt line width for axes
    'axes.edgecolor': 'black',
    'axes.grid': False,  # Turn off grid lines
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.fancybox': True,
    'legend.edgecolor': 'black',
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    
    # Precise tick length settings (1mm = 3.54 points, 0.6mm = 2.13 points)
    'xtick.major.size': 3.54,  # Major ticks 1mm
    'xtick.minor.size': 2.13,  # Minor ticks 0.6mm
    'ytick.major.size': 3.54,  # Major ticks 1mm
    'ytick.minor.size': 2.13,  # Minor ticks 0.6mm
    
    'xtick.major.width': 0.8,  # Tick line width
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 1.0,  # 1pt line width
    'lines.markersize': 4,
})

# Iteration data
iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Best ELBO values
best_elbo = ['27.5452', '30.8565', '34.4512', '36.7469', '39.0379', '31.3860', '38.5289', '39.9130', '38.7074', '38.9396']  # from bgmm.py output
best_elbo = [float(i) for i in best_elbo]

# Calculate differences (starting from iteration 1)
diffs = [best_elbo[i] - best_elbo[i-1] for i in range(1, len(best_elbo))]

# ========= Primary Y-axis: Best ELBO curve =========
# Plot Best ELBO curve
ax1.plot(iterations, best_elbo, 
         label='Best ELBO', 
         color='#1f77b4', 
         marker='o', 
         markersize=4,
         linewidth=1.0,  # 1pt line width
         linestyle='-',
         markeredgewidth=0.5,
         zorder=3)

# Set primary axis labels
ax1.set_xlabel('Iteration', fontsize=8, fontweight='bold')
ax1.set_ylabel('ELBO Value', fontsize=8, fontweight='bold', color='#1f77b4')
ax1.set_title('ELBO Evolution during Clustering Iterations', 
             fontsize=9, fontweight='bold', pad=10)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.grid(False)  # Ensure primary axis grid lines are off
ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=6)
ax1.tick_params(axis='both', which='major', labelsize=6)

# ========= Secondary Y-axis: Best ELBO differences =========
ax2 = ax1.twinx()
ax2.set_ylabel('Best ELBO Difference', fontsize=8, fontweight='bold', color='#ff7f0e')
ax2.tick_params(axis='y', which='major', labelsize=6, colors='#ff7f0e')
ax2.grid(False)  # Ensure secondary axis grid lines are off

# Plot difference bars (starting from iteration 1)
bar_positions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
bars = ax2.bar(bar_positions, diffs, 
               color=['#ff7f0e' if abs(d) >= 2 else '#17becf' for d in diffs],
               alpha=0.7, 
               width=0.5,
               edgecolor='k',
               linewidth=0.5,
               zorder=2)

# Note: Removed the loop for adding difference value labels, this is the key modification to remove numeric labels on bars

# Add convergence threshold lines (±2)
ax2.axhline(y=2, color='r', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
ax2.axhline(y=-2, color='r', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
ax2.text(4.1, 2.2, 'Threshold (|Δ|=2)', 
         color='r', fontsize=6, fontstyle='italic', zorder=4)

# Set secondary axis range from -30 to 5
ax1.set_ylim(26, 50)
ax2.set_ylim(-30, 8)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='lower right', fontsize=6)

# Precise layout adjustment - ensure overall dimensions 70mm×50mm
plt.tight_layout(pad=1.0, rect=(0, 0, 1, 1))  # Remove extra margins

# Save as PDF (ensure text is editable)
plt.savefig('best_elbo_evolution.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)

# Save as SVG (AI compatible)
plt.savefig('best_elbo_evolution.svg', format='svg', bbox_inches='tight', pad_inches=0.01)

plt.show()
