import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 3,
    'text.usetex': False,
})

WIDTH_MM = 70
HEIGHT_MM = 50
fig, ax1 = plt.subplots(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4))

plt.rcParams.update({
    'axes.linewidth': 1.0,
    'axes.edgecolor': 'black',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.fancybox': True,
    'legend.edgecolor': 'black',
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    
    'xtick.major.size': 3.54,
    'xtick.minor.size': 2.13,
    'ytick.major.size': 3.54,
    'ytick.minor.size': 2.13,
    
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
})

iterations = [0, 1, 2, 3, 4]

best_elbo = [np.float64(59.6901150961101), np.float64(22.009429918258007), np.float64(34.36695125989896), np.float64(26.499801151174417), np.float64(27.50745569102098)]

diffs = [best_elbo[i] - best_elbo[i-1] for i in range(1, len(best_elbo))]

# ========= Main coordinate axis：Best ELBO  =========
ax1.plot(iterations, best_elbo, 
         label='Best ELBO', 
         color='#1f77b4', 
         marker='o', 
         markersize=4,
         linewidth=1.0,
         linestyle='-',
         markeredgewidth=0.5,
         zorder=3)

ax1.set_xlabel('Iteration', fontsize=8, fontweight='bold')
ax1.set_ylabel('ELBO Value', fontsize=8, fontweight='bold', color='#1f77b4')
ax1.set_title('ELBO Evolution during Clustering Iterations', 
             fontsize=9, fontweight='bold', pad=10)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=6)
ax1.tick_params(axis='both', which='major', labelsize=6)

# ========= The next coordinate axis：Best ELBO diff =========
ax2 = ax1.twinx()
ax2.set_ylabel('Best ELBO Difference', fontsize=8, fontweight='bold', color='#ff7f0e')
ax2.tick_params(axis='y', which='major', labelsize=6, colors='#ff7f0e')

bar_positions = [1, 2, 3, 4]
bars = ax2.bar(bar_positions, diffs, 
               color=['#ff7f0e' if abs(d) >= 2 else '#17becf' for d in diffs],
               alpha=0.7, 
               width=0.5,
               edgecolor='k',
               linewidth=0.5,
               zorder=2)

for idx, bar in enumerate(bars):
    height = bar.get_height()
    va = 'bottom' if diffs[idx] > 0 else 'top'
    y_offset = 0.3 if diffs[idx] > 0 else -0.5
    ax2.text(bar.get_x() + bar.get_width()/2., 
             height + y_offset,
             f'{diffs[idx]:.2f}', 
             ha='center', 
             va=va,
             color='#ff7f0e',
             fontsize=5.5,
             fontweight='bold',
             zorder=4)

ax2.axhline(y=2, color='r', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
ax2.axhline(y=-2, color='r', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
ax2.text(4.1, 2.2, 'Threshold (|Δ|=2)', 
         color='r', fontsize=6, fontstyle='italic', zorder=4)

ax2.annotate('Convergence', 
            xy=(4, diffs[-1]), 
            xytext=(3.2, -10),
            arrowprops=dict(arrowstyle="->", color='k', linewidth=0.7),
            fontsize=6,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
            zorder=5)

ax2.set_ylim(-75, 15)

lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='lower left', fontsize=6)

plt.tight_layout(pad=1.0, rect=(0, 0, 1, 1))

# plt.savefig(r'.\output\best_elbo_evolution.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)

# Save as SVG（for Adobe Illustration）
plt.savefig(r'.\output\best_elbo_evolution.svg', format='svg', bbox_inches='tight', pad_inches=0.01)

# plt.show()