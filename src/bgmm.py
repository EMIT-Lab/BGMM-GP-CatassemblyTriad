import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import random
import os

# Set global random seed - ensure reproducible results
SEED = 4194
np.random.seed(SEED)
random.seed(SEED)

# Convergence threshold - based on relative change in ELBO
ELBO_CONVERGENCE_THRESHOLD = 0.01

class QuantifiedGMM:
    """
    Incremental Bayesian Gaussian Mixture Model class
    Supports batch data addition and automatically determines optimal number of clusters
    """
    
    def __init__(self, initial_data, initial_ids):
        """
        Initialize the model
        
        Parameters:
        -----------
        initial_data : ndarray
            Initial data points, shape (n_samples, n_features)
        initial_ids : ndarray
            IDs of initial data points
        """
        self.data_history = [initial_data.copy()]
        self.id_history = [initial_ids.copy()]
        self.results = []
        self.convergence_history = []
        
    def update_model(self, new_data=None, new_ids=None, n_components_range=(2, 5)):
        """
        Update model by adding new data and retraining
        
        Parameters:
        -----------
        new_data : ndarray, optional
            New data points
        new_ids : ndarray, optional  
            IDs of new data points
        n_components_range : tuple
            Search range for number of clusters
            
        Returns:
        --------
        result : dict
            Dictionary containing model results and statistics
        """
        # Add new data (if provided)
        if new_data is not None and new_ids is not None:
            self.data_history.append(np.vstack([self.data_history[-1], new_data]))
            self.id_history.append(np.concatenate([self.id_history[-1], new_ids]))
        
        X = self.data_history[-1]
        point_ids = self.id_history[-1]
        
        # 1. Try different cluster numbers and select the best (based on ELBO)
        best_elbo = -np.inf
        all_elbos = {}
        
        print(f"Trying cluster number range: {n_components_range}")
        for n in range(n_components_range[0], n_components_range[1] + 1):
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
        
        print(f"Optimal number of clusters: {best_n}, ELBO: {best_elbo:.4f}")
        
        # 2. Check convergence condition
        convergence_metric = self._check_convergence(best_elbo)
        
        # 3. Get best model results
        probs = best_clf.predict_proba(X)
        labels = best_clf.predict(X)
        cluster_certainty = self._calculate_certainty(probs, labels)

        actual_clusters = len(np.unique(labels))
        
        # Store results
        result = {
            'model': best_clf,
            'gmm': best_gmm,
            'probs': probs,
            'labels': labels,
            'cluster_certainty': cluster_certainty,
            'effective_clusters': actual_clusters,
            'elbo': best_elbo,
            'weights': best_gmm.weights_,
            'all_elbos': all_elbos,
            'point_ids': point_ids,
            'cluster_members': self._get_cluster_members(labels, point_ids),
            'convergence': convergence_metric,
            'data': X  # Add original data to results
        }

        self.results.append(result)
        self.convergence_history.append(convergence_metric)
        
        return result
    
    def _get_clf(self, n_components):
        """Create Bayesian GMM classifier pipeline"""
        gmm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.01,
            mean_precision_prior=0.01,
            covariance_type="full",
            max_iter=1000,
            random_state=SEED,
            n_init=5,
        )
        return make_pipeline(StandardScaler(), gmm)
    
    def _calculate_certainty(self, probs, labels):
        """Calculate certainty metrics for each cluster"""
        cluster_certainty = []
        for i in range(probs.shape[1]):
            cluster_probs = probs[labels == i, i]
            if len(cluster_probs) > 0:
                certainty = {
                    'cluster_id': i,
                    'mean_prob': np.mean(cluster_probs),
                    'std_prob': np.std(cluster_probs),
                    'min_prob': np.min(cluster_probs),
                    'max_prob': np.max(cluster_probs),
                    'n_members': len(cluster_probs)
                }
                cluster_certainty.append(certainty)
        return cluster_certainty
    
    def _get_cluster_members(self, labels, point_ids):
        """Get member IDs in each cluster"""
        cluster_members = {}
        for cluster_id in np.unique(labels):
            indices = np.where(labels == cluster_id)[0]
            # Note: point IDs start from 0, but displayed as +1 to match original numbering
            cluster_members[cluster_id] = {
                'ids': (point_ids[indices] + 1).tolist(),
                'count': len(indices)
            }
        return cluster_members
    
    def _check_convergence(self, current_elbo):
        """Check model convergence condition"""
        if len(self.results) == 0:
            return {
                'elbo_change': 0,
                'is_converged': False
            }
        
        prev_elbo = self.results[-1]['elbo']
        elbo_change = abs(current_elbo - prev_elbo) / (abs(prev_elbo) + 1e-10)
        
        is_converged = elbo_change < ELBO_CONVERGENCE_THRESHOLD
        
        return {
            'elbo_change': elbo_change,
            'is_converged': is_converged,
            'threshold': ELBO_CONVERGENCE_THRESHOLD
        }


def print_iteration_report(result, iteration, df):
    """
    Print and save detailed iteration report
    
    Parameters:
    -----------
    result : dict
        Model results dictionary
    iteration : int
        Current iteration number
    df : DataFrame
        Original dataframe for getting compound names and GROUP information
    """
    # Create report directory
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/iteration_{iteration}_report.txt"
    
    with open(report_path, "w", encoding='utf-8') as f:
        # Title and basic information
        f.write("=" * 60 + "\n")
        f.write(f"ITERATION {iteration} REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("BASIC INFORMATION:\n")
        f.write(f"  Number of data points: {len(result['point_ids'])}\n")
        f.write(f"  Point IDs: {result['point_ids'] + 1}\n")
        f.write(f"  Effective clusters: {result['effective_clusters']}\n")
        f.write(f"  Best ELBO: {result['elbo']:.6f}\n")
        f.write(f"  Convergence: {result['convergence']['is_converged']} "
                f"(ELBO change: {result['convergence']['elbo_change']:.6f})\n\n")
        
        # ELBO information
        f.write("ELBO VALUES FOR DIFFERENT CLUSTER NUMBERS:\n")
        for n, elbo in result['all_elbos'].items():
            marker = " *" if n == result['effective_clusters'] else ""
            f.write(f"  n_components = {n}: ELBO = {elbo:.6f}{marker}\n")
        f.write("\n")
        
        # Cluster weights
        f.write("CLUSTER WEIGHTS:\n")
        for i, weight in enumerate(result['weights']):
            f.write(f"  Cluster {i}: weight = {weight:.4f}\n")
        f.write("\n")
        
        # Cluster member details
        f.write("CLUSTER MEMBERSHIP DETAILS:\n")
        for cluster_id, members_info in result['cluster_members'].items():
            f.write(f"  Cluster {cluster_id} ({members_info['count']} members):\n")
            
            # Get detailed information for all points in this cluster
            cluster_indices = [idx for idx, label in enumerate(result['labels']) 
                             if label == cluster_id]
            
            for idx in cluster_indices:
                point_id = result['point_ids'][idx]
                compound_name = df.iloc[point_id, 0]  # Hybrid_method column
                group = df.iloc[point_id, 7] if len(df.columns) > 7 else "N/A"  # GROUP column
                probability = result['probs'][idx, cluster_id]
                
                f.write(f"    ID {point_id + 1:2d}: {compound_name:<15} "
                       f"(Group: {group}, Prob: {probability:.3f})\n")
            f.write("\n")
        
        # Cluster certainty statistics
        f.write("CLUSTER CERTAINTY STATISTICS:\n")
        for certainty in result['cluster_certainty']:
            f.write(f"  Cluster {certainty['cluster_id']}:\n")
            f.write(f"    Mean probability: {certainty['mean_prob']:.4f}\n")
            f.write(f"    Std probability:  {certainty['std_prob']:.4f}\n")
            f.write(f"    Min probability:  {certainty['min_prob']:.4f}\n")
            f.write(f"    Max probability:  {certainty['max_prob']:.4f}\n")
            f.write(f"    Member count:     {certainty['n_members']}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    # Print simplified report to console
    print(f"\n=== Iteration {iteration} Summary ===")
    print(f"Points: {len(result['point_ids'])}, Clusters: {result['effective_clusters']}")
    print(f"ELBO: {result['elbo']:.4f}, Converged: {result['convergence']['is_converged']}")
    for cluster_id, members_info in result['cluster_members'].items():
        print(f"Cluster {cluster_id}: {members_info['ids']}")


def visualize_clustering(result, iteration, df, output_dir="output"):
    """
    Generate 3D clustering visualization
    
    Parameters:
    -----------
    result : dict
        Model results dictionary
    iteration : int
        Current iteration number
    df : DataFrame
        Original dataframe
    output_dir : str
        Output directory
    """
    # Use standardized data for visualization
    data = result['model'].named_steps['standardscaler'].transform(result['data'])
    point_ids = result['point_ids']
    labels = result['labels']
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set colors for different clusters
    colors = ["red", "purple", "gold", "green", "blue", "orange", "pink"]
    cluster_colors = [colors[l % len(colors)] for l in labels]
    
    # Plot scatter points
    scatter = ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], 
                          c=cluster_colors, s=80, edgecolor='black', 
                          linewidths=1, alpha=0.8)
    
    # Add labels for each point
    for j, (x, y, z, pid) in enumerate(zip(data[:, 0], data[:, 1], data[:, 2], point_ids)):
        # compound_name = df.iloc[pid, 0]  # Hybrid_method column
        ax.text(x, y, z, f'{pid+1}', fontsize=6, ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Add legend
    unique_labels = np.unique(labels)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=colors[l], markersize=10, 
                      label=f'Cluster {l}') for l in unique_labels]
    
    ax.legend(handles=legend_elements, title='Clusters', loc='upper left')
    
    # Set axis labels and ranges
    ax.set_xlabel('Attachability (scaled)')
    ax.set_ylabel('Controllability (scaled)')
    ax.set_zlabel('Detachability (scaled)')
    # ax.set_title(f'GMM Clustering - Iteration {iteration}\n'
    #             f'Points: {[pid+1 for pid in point_ids]}, Clusters: {result["effective_clusters"]}')

    # Set axis ranges (based on standardized data)
    padding = 0.5
    ax.set_xlim(data[:, 0].min() - padding, data[:, 0].max() + padding)
    ax.set_ylim(data[:, 1].min() - padding, data[:, 1].max() + padding)
    ax.set_zlim(data[:, 2].min() - padding, data[:, 2].max() + padding)
    
    # Set viewing angle
    ax.view_init(30, -45)
    
    # Save image
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/iteration_{iteration}_clustering.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization saved: {output_dir}/iteration_{iteration}_clustering.png")


def main():
    """
    Main function: Execute incremental clustering analysis
    """
    print("Starting incremental Bayesian GMM clustering analysis...")
    print("=" * 50)
    
    # 1. Read and preprocess data
    print("Step 1: Reading data...")
    csv_path = r".\data\data_sorted_by_No_in_article.csv"  # Update data path
    df = pd.read_csv(csv_path)
    
    # Extract feature data (columns 2-5: attchability, controlability, detachability)
    X = df.iloc[:, 2:5].values
    X[:, 1] = 0.35 * X[:, 1]  # Adjust scaling factor
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Data range - Attachability: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"Data range - Controllability: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    print(f"Data range - Detachability: [{X[:, 2].min():.2f}, {X[:, 2].max():.2f}]")
    
    # 2. Define data batches (40 points divided into 10 rounds, 4 points per round)
    batches = [
        list(range(0, 4)),     # Round 1: points 0-3
        list(range(4, 8)),    # Round 2: points 4-7
        list(range(8, 12)),   # Round 3: points 8-11
        list(range(12, 16)),   # Round 4: points 12-17
        list(range(16, 20)),    # Round 5: points 16-19
        list(range(20, 24)),     # Round 6: points 20-23
        list(range(24, 28)),    # Round 7: points 24-27 
        list(range(28, 32)),   # Round 8: points 28-31
        list(range(32, 36)),   # Round 9: points 32-35
        list(range(36, 40))    # Round 10: points 36-39
    ]
    
    print(f"Batch division: {len(batches)} rounds total, {len(batches[0])} points per round")
    
    # 3. Initialize model
    print("\nStep 2: Initializing model...")
    initial_ids = np.array(batches[0])
    qgmm = QuantifiedGMM(X[initial_ids, :], initial_ids)
    
    # 4. Execute incremental learning
    print("\nStep 3: Starting incremental learning...")
    for i in range(len(batches)):
        print(f"\n--- Iteration {i+1} ---")
        
        if i == 0:
            # First round uses smaller cluster range
            result = qgmm.update_model(n_components_range=(2, 3))
        else:
            # Subsequent rounds add new data
            new_ids = np.array(batches[i])
            result = qgmm.update_model(
                new_data=X[new_ids, :], 
                new_ids=new_ids,
                n_components_range=(2, 5)  # Expand search range
            )
        
        # Generate detailed report
        print_iteration_report(result, i, df)
        
        # Generate visualization
        visualize_clustering(result, i, df, output_dir="output")
        
        # Check convergence
        if result['convergence']['is_converged']:
            print(f"Model converged at iteration {i+1}!")
            # Can terminate early, but continue to observe complete process
    
    # 5. Final summary
    print("\n" + "=" * 50)
    print("Incremental learning completed!")
    print(f"Total iterations: {len(batches)}")
    print(f"Final number of data points: {len(qgmm.data_history[-1])}")
    print(f"Final number of clusters: {qgmm.results[-1]['effective_clusters']}")
    
    # Save final model state
    final_result = qgmm.results[-1]
    print(f"\nFinal cluster distribution:")
    for cluster_id, members_info in final_result['cluster_members'].items():
        print(f"  Cluster {cluster_id}: {members_info['count']} members")
        print(f"    Member IDs: {members_info['ids']}")
    
    print(f"\nAll results saved to 'reports/' and 'output/' directories")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("reports", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Run main program
    main()
    
