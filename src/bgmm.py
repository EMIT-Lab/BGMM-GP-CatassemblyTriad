import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import os

"""
Even fixed random seeds may behave differently on different computers, but the trend is approximate.
"""

# Fixed parameters from the research description
FIXED_SEED = 4587502
CONCENTRATION_PARAM = 0.01
ELBO_ABS_THRESHOLD = 2
ELBO_REL_THRESHOLD = 0.01  # 1% relative change

class QuantifiedGMM:
    def __init__(self, initial_data, initial_ids):
        self.data_history = [initial_data.copy()]
        self.id_history = [initial_ids.copy()]
        self.results = []
        self.convergence_history = []
        
    def update_model(self, new_data=None, new_ids=None, n_components_range=(2, 5)):
        if new_data is not None and new_ids is not None:
            self.data_history.append(np.vstack([self.data_history[-1], new_data]))
            self.id_history.append(np.concatenate([self.id_history[-1], new_ids]))
        
        X = self.data_history[-1]
        point_ids = self.id_history[-1]
        
        # Cap n_components by number of samples
        max_components = min(n_components_range[1], X.shape[0])
        n_components_range_adj = (n_components_range[0], max_components)
        
        best_elbo = -np.inf
        all_elbos = {}
        
        for n in range(n_components_range_adj[0], n_components_range_adj[1] + 1):
            clf = self._get_clf(n)
            clf.fit(X)
            gmm = clf.named_steps['bayesiangaussianmixture']
            elbo = gmm.lower_bound_
            all_elbos[n] = elbo

            if elbo > best_elbo:
                best_elbo = elbo
                best_n = n
                best_clf = clf
                best_gmm = gmm
        
        convergence_metric = self._check_convergence(best_elbo)
        
        probs = best_clf.predict_proba(X)
        labels = best_clf.predict(X)
        cluster_certainty = self._calculate_certainty(probs, labels)
        
        result = {
            'model': best_clf,
            'gmm': best_gmm,
            'probs': probs, 
            'labels': labels,
            'cluster_certainty': cluster_certainty,
            'n_components_for_model': best_n,
            'elbo': best_gmm.lower_bound_,
            'weights': best_gmm.weights_,
            'all_elbos': all_elbos,
            'point_ids': point_ids, 
            'cluster_members': self._get_cluster_members(labels, point_ids),
            'data': X  # Store the data matrix
        }

        self.results.append(result)
        self.convergence_history.append(convergence_metric)
        
        return result
    
    def _get_clf(self, n_components):
        gmm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=CONCENTRATION_PARAM,
            mean_precision_prior=0.01,
            covariance_type="full",
            max_iter=1000,
            random_state=FIXED_SEED,
            n_init=5,
        )
        return make_pipeline(StandardScaler(), gmm)
    
    def _calculate_certainty(self, probs, labels):
        cluster_certainty = []
        for i in range(probs.shape[1]):
            cluster_probs = probs[labels == i, i]
            if len(cluster_probs) > 0:
                certainty = {
                    'cluster_id': i,
                    'mean_prob': np.mean(cluster_probs),
                    'weight': np.mean(cluster_probs)
                }
                cluster_certainty.append(certainty)
        return cluster_certainty
    
    def _get_cluster_members(self, labels, point_ids):
        cluster_members = {}
        for cluster_id in np.unique(labels):
            indices = np.where(labels == cluster_id)[0]
            cluster_members[cluster_id] = (point_ids[indices]+1).tolist()
        return cluster_members
    
    def _check_convergence(self, current_elbo):
        if len(self.results) == 0:
            return {
                'abs_change': 0,
                'rel_change': 0,
                'is_converged': False
            }
        
        prev_elbo = self.results[-1]['elbo']
        abs_change = abs(current_elbo - prev_elbo)
        rel_change = abs_change / (abs(prev_elbo) + 1e-10)
        
        is_converged = (abs_change < ELBO_ABS_THRESHOLD) and (rel_change < ELBO_REL_THRESHOLD)
        
        return {
            'abs_change': abs_change,
            'rel_change': rel_change,
            'is_converged': is_converged
        }


def print_iteration_report(result, iteration):
    os.makedirs("reports", exist_ok=True)
    report_path = f"./reports/iteration_{iteration}_report.txt"
    
    with open(report_path, "w") as f:
        f.write(f"==================== ITERATION {iteration} REPORT ====================\n\n")
        f.write(f"Fixed Seed: {FIXED_SEED}\n")
        f.write(f"Concentration Parameter: {CONCENTRATION_PARAM}\n")
        f.write(f"Number of points: {len(result['point_ids'])}\n")
        f.write(f"Points IDs: {result['point_ids']+1}\n")
        f.write(f"Effective clusters: {result['n_components_for_model']}\n")
        f.write(f"ELBO: {result['elbo']:.4f}\n")
        
        # Add convergence diagnostics
        if iteration > 0:
            conv = result['convergence_metric']  # Added in main execution
            f.write(f"Absolute ELBO Change: {conv['abs_change']:.4f}\n")
            f.write(f"Relative ELBO Change: {conv['rel_change']:.4%}\n")
            f.write(f"Convergence Status: {'Met' if conv['is_converged'] else 'Not Met'}\n")
        
        f.write("\nELBO values for different cluster numbers:\n")
        for n, elbo in result['all_elbos'].items():
            f.write(f"  n_components={n}: ELBO = {elbo:.4f}\n")
        
        f.write("\nCluster weights:\n")
        for i, weight in enumerate(result['weights']):
            f.write(f"  Cluster {i}: weight = {weight:.4f}\n")
        
        f.write("\nCluster members:\n")
        for cluster_id, members in result['cluster_members'].items():
            f.write(f"  Cluster {cluster_id}: {members}\n")
        
        f.write("\nCluster certainty:\n")
        for certainty in result['cluster_certainty']:
            f.write(f"  Cluster {certainty['cluster_id']}: mean probability = {certainty['mean_prob']:.4f}\n")
        
        f.write("\n=======================================================\n")
    
    with open(report_path, "r") as f:
        print(f.read())


def visualize_clusters(result, iteration):
    """Create 3D visualization of clusters with original units"""
    # Get data in original units
    data_original = result['data'].copy()
    data_original[:, 2] *= 10  # Reverse controllability scaling
    
    # Get cluster centers in original units
    centers_scaled = result['gmm'].means_
    centers_original = result['model'].named_steps['standardscaler'].inverse_transform(centers_scaled)
    centers_original[:, 2] *= 10  # Reverse controllability scaling
    
    labels = result['labels']
    point_ids = result['point_ids']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Cluster coloring scheme
    colors = ["red", "purple", "gold", "green", "blue"]
    point_colors = [colors[l] for l in labels]
    center_colors = [colors[i] for i in range(len(centers_original))]
    
    # Plot data points
    ax.scatter3D(data_original[:, 0], data_original[:, 1], data_original[:, 2], 
                c=point_colors, s=80, edgecolor='black', 
                linewidths=1, alpha=0.8)
    
    # Plot cluster centers
    ax.scatter3D(centers_original[:, 0], centers_original[:, 1], centers_original[:, 2], 
                c=center_colors, s=200, marker='*', edgecolor='black')
    
    # Label points
    for j, pid in enumerate(point_ids):
        ax.text(data_original[j, 0], data_original[j, 1], data_original[j, 2], 
               f'{pid+1}', fontsize=9, ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Label cluster centers
    for i, (x, y, z) in enumerate(centers_original):
        ax.text(x, y, z, f'C{i}', fontsize=12, ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', pad=1))
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=colors[i], markersize=10, 
                      label=f'Cluster {i}') for i in range(len(centers_original))]
    
    ax.legend(handles=legend_elements, title='Clusters', loc='best')
    
    ax.set_xlabel('Attachability')
    ax.set_ylabel('Detachability')
    ax.set_zlabel('Controllability')
    ax.set_title(f'BGMM Clustering - Iteration {iteration} (Seed: {FIXED_SEED})')
    
    plt.savefig(f"./output/iteration_{iteration}_clustering.png", dpi=300)
    plt.close(fig)


def run_analysis():
    """Run the analysis with fixed parameters from research"""
    csv_path = r".\data\data.csv"
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 2:5].values
    X[:, -1] = 0.1 * X[:, -1]  # Scale controllability

    # Batch configuration: 5 iterations with 4 samples each
    batches = [
        [0, 1, 2, 3],   # Initial batch
        [4, 5, 6, 7],    # Batch 1
        [8, 9, 10, 11],  # Batch 2
        [12, 13, 14, 15],# Batch 3
        [16, 17, 18, 19] # Batch 4
    ]
    
    # Initialize with first batch
    initial_ids = np.array(batches[0])
    qgmm = QuantifiedGMM(X[initial_ids, :], initial_ids)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Run through all 5 batches
    for i, batch_ids in enumerate(batches):
        if i == 0:
            # For first batch, limit n_components to 4 (since only 4 samples)
            result = qgmm.update_model(n_components_range=(2, 4))
        else:
            result = qgmm.update_model(
                new_data=X[batch_ids, :], 
                new_ids=np.array(batch_ids),
                n_components_range=(2, 5)
            )
        
        # Attach convergence metric to result for reporting
        result['convergence_metric'] = qgmm.convergence_history[-1] if i > 0 else {
            'abs_change': 0,
            'rel_change': 0,
            'is_converged': False
        }
        
        print_iteration_report(result, i)
        visualize_clusters(result, i)


if __name__ == "__main__":
    run_analysis()
