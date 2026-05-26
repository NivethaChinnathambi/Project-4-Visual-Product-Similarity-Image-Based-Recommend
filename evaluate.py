"""
Evaluation Metrics Script
Calculates Precision@K, Recall@K, and other metrics for the recommendation system
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from search_engine import SearchEngine
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = 'results'
METADATA_FILE = 'data/metadata.csv'
K_VALUES = [1, 5, 10, 15, 20]
NUM_QUERIES = 50  # Number of test queries

# ============================================================================
# EVALUATION METRICS CLASS
# ============================================================================

class EvaluationMetrics:
    """Calculate evaluation metrics for recommendation system"""
    
    def __init__(self, engine, metadata_file):
        """Initialize evaluation metrics"""
        self.engine = engine
        self.metadata = pd.read_csv(metadata_file)
        self.results = {
            'precision_at_k': {},
            'recall_at_k': {},
            'mean_reciprocal_rank': {},
            'ndcg_at_k': {}
        }
        
    def get_ground_truth(self, query_product_id, k=10):
        """
        Get ground truth similar products for a query
        Ground truth = products in the same category
        """
        query_row = self.metadata[self.metadata['product_id'] == query_product_id]
        if len(query_row) == 0:
            return []
        
        query_category = query_row.iloc[0]['category']
        similar_products = self.metadata[
            (self.metadata['category'] == query_category) & 
            (self.metadata['product_id'] != query_product_id)
        ]['product_id'].tolist()
        
        return similar_products[:k]
    
    def calculate_precision_at_k(self, retrieved, ground_truth, k):
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_count = len(set(retrieved_at_k) & set(ground_truth))
        
        return relevant_count / k
    
    def calculate_recall_at_k(self, retrieved, ground_truth, k):
        """Calculate Recall@K"""
        if len(ground_truth) == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_count = len(set(retrieved_at_k) & set(ground_truth))
        
        return relevant_count / len(ground_truth)
    
    def calculate_mean_reciprocal_rank(self, retrieved, ground_truth):
        """Calculate Mean Reciprocal Rank (MRR)"""
        for i, product_id in enumerate(retrieved):
            if product_id in ground_truth:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_ndcg(self, retrieved, ground_truth, k):
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K)"""
        # Discounted Cumulative Gain
        dcg = 0.0
        for i, product_id in enumerate(retrieved[:k]):
            if product_id in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because ranking starts at 1
        
        # Ideal DCG (best possible ranking)
        idcg = 0.0
        for i in range(min(k, len(ground_truth))):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate(self, num_queries=50):
        """Run evaluation on multiple queries"""
        
        print("=" * 70)
        print("EVALUATION METRICS - RECOMMENDATION SYSTEM")
        print("=" * 70)
        
        # Get random sample of products for testing
        all_products = self.metadata['product_id'].tolist()
        test_products = np.random.choice(all_products, size=min(num_queries, len(all_products)), replace=False)
        
        print(f"\nRunning evaluation on {len(test_products)} test queries...")
        print("-" * 70)
        
        # Store results for each query
        all_precision_at_k = {k: [] for k in K_VALUES}
        all_recall_at_k = {k: [] for k in K_VALUES}
        all_mrr = []
        all_ndcg_at_k = {k: [] for k in K_VALUES}
        
        for idx, query_product_id in enumerate(test_products):
            try:
                # Get query image path
                query_row = self.metadata[self.metadata['product_id'] == query_product_id]
                if len(query_row) == 0:
                    continue
                
                query_image_path = query_row.iloc[0]['image_path']
                
                # Search for similar products
                search_results = self.engine.search_by_image(query_image_path, k=max(K_VALUES))
                retrieved_products = [r['product_id'] for r in search_results]
                
                # Get ground truth (products in same category)
                ground_truth = self.get_ground_truth(query_product_id, k=20)
                
                # Calculate metrics
                mrr = self.calculate_mean_reciprocal_rank(retrieved_products, ground_truth)
                all_mrr.append(mrr)
                
                for k in K_VALUES:
                    precision = self.calculate_precision_at_k(retrieved_products, ground_truth, k)
                    recall = self.calculate_recall_at_k(retrieved_products, ground_truth, k)
                    ndcg = self.calculate_ndcg(retrieved_products, ground_truth, k)
                    
                    all_precision_at_k[k].append(precision)
                    all_recall_at_k[k].append(recall)
                    all_ndcg_at_k[k].append(ndcg)
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(test_products)} queries")
            
            except Exception as e:
                print(f"Error processing query {query_product_id}: {str(e)}")
                continue
        
        # Calculate average metrics
        print("-" * 70)
        print("\n✅ Evaluation Complete!")
        print("\n" + "=" * 70)
        print("RESULTS - AVERAGE METRICS")
        print("=" * 70)
        
        print(f"\nMean Reciprocal Rank (MRR): {np.mean(all_mrr):.4f}")
        
        print("\nPrecision@K:")
        for k in K_VALUES:
            avg_precision = np.mean(all_precision_at_k[k])
            print(f"  Precision@{k}: {avg_precision:.4f}")
            self.results['precision_at_k'][k] = avg_precision
        
        print("\nRecall@K:")
        for k in K_VALUES:
            avg_recall = np.mean(all_recall_at_k[k])
            print(f"  Recall@{k}: {avg_recall:.4f}")
            self.results['recall_at_k'][k] = avg_recall
        
        print("\nNDCG@K:")
        for k in K_VALUES:
            avg_ndcg = np.mean(all_ndcg_at_k[k])
            print(f"  NDCG@{k}: {avg_ndcg:.4f}")
            self.results['ndcg_at_k'][k] = avg_ndcg
        
        self.results['mean_reciprocal_rank'] = np.mean(all_mrr)
        
        # Save results to CSV
        self.save_results(all_precision_at_k, all_recall_at_k, all_ndcg_at_k, all_mrr)
        
        # Generate visualizations
        self.generate_visualizations(all_precision_at_k, all_recall_at_k, all_ndcg_at_k)
        
        return self.results
    
    def save_results(self, precision_at_k, recall_at_k, ndcg_at_k, mrr):
        """Save results to CSV file"""
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'K': K_VALUES,
            'Precision@K': [np.mean(precision_at_k[k]) for k in K_VALUES],
            'Recall@K': [np.mean(recall_at_k[k]) for k in K_VALUES],
            'NDCG@K': [np.mean(ndcg_at_k[k]) for k in K_VALUES]
        })
        
        results_file = os.path.join(RESULTS_DIR, 'evaluation_metrics.csv')
        results_df.to_csv(results_file, index=False)
        
        print(f"\n✅ Results saved to {results_file}")
        print(results_df.to_string(index=False))
    
    def generate_visualizations(self, precision_at_k, recall_at_k, ndcg_at_k):
        """Generate visualization plots"""
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Prepare data
        k_values = K_VALUES
        avg_precision = [np.mean(precision_at_k[k]) for k in k_values]
        avg_recall = [np.mean(recall_at_k[k]) for k in k_values]
        avg_ndcg = [np.mean(ndcg_at_k[k]) for k in k_values]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Precision@K
        axes[0].plot(k_values, avg_precision, marker='o', linewidth=2, markersize=8, color='#FF9900')
        axes[0].set_xlabel('K', fontsize=12)
        axes[0].set_ylabel('Precision', fontsize=12)
        axes[0].set_title('Precision@K', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(k_values)
        
        # Plot 2: Recall@K
        axes[1].plot(k_values, avg_recall, marker='s', linewidth=2, markersize=8, color='#146eb4')
        axes[1].set_xlabel('K', fontsize=12)
        axes[1].set_ylabel('Recall', fontsize=12)
        axes[1].set_title('Recall@K', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(k_values)
        
        # Plot 3: NDCG@K
        axes[2].plot(k_values, avg_ndcg, marker='^', linewidth=2, markersize=8, color='#4caf50')
        axes[2].set_xlabel('K', fontsize=12)
        axes[2].set_ylabel('NDCG', fontsize=12)
        axes[2].set_title('NDCG@K', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(k_values)
        
        plt.tight_layout()
        
        # Save figure
        plot_file = os.path.join(RESULTS_DIR, 'evaluation_metrics.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to {plot_file}")
        
        plt.close()

# ============================================================================
# RUN EVALUATION
# ============================================================================

def main():
    """Main evaluation function"""
    
    try:
        # Load search engine
        print("Loading search engine...")
        engine = SearchEngine()
        
        # Initialize evaluator
        evaluator = EvaluationMetrics(engine, METADATA_FILE)
        
        # Run evaluation
        results = evaluator.evaluate(num_queries=NUM_QUERIES)
        
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved to: {RESULTS_DIR}/")
        print("  - evaluation_metrics.csv")
        print("  - evaluation_metrics.png")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
