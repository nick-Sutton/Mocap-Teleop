#!/usr/bin/env python3

from datetime import datetime
import glob
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_curve, auc
)
from collections import defaultdict
import json

class FeatureMisclassificationAnalyzer:
    """Analyze features of misclassified instances to understand decision boundaries"""
    
    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.class_names = preprocessor.label_encoder.classes_
    
    def get_raw_sequences_with_predictions(self, data_loader, indices):
        """Extract raw sequences for specific indices"""
        # Convert DataLoader back to sequences
        all_sequences = []
        all_labels = []
        
        for batch_X, batch_y in data_loader:
            all_sequences.append(batch_X.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
        
        all_sequences = np.vstack(all_sequences)
        all_labels = np.concatenate(all_labels)
        
        # Extract requested indices
        selected_sequences = all_sequences[indices]
        selected_labels = all_labels[indices]
        
        return selected_sequences, selected_labels
    
    def analyze_specific_misclassification(self, val_loader, 
                                          true_class_name, pred_class_name, 
                                          n_samples=20):
        """Analyze specific type of misclassification (e.g., standing→walking)"""
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {true_class_name} → {pred_class_name}")
        print(f"{'='*80}\n")
        
        # Get predictions
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Find target misclassification indices
        true_idx = list(self.class_names).index(true_class_name)
        pred_idx = list(self.class_names).index(pred_class_name)
        
        misclass_mask = (all_labels == true_idx) & (all_preds == pred_idx)
        misclass_indices = np.where(misclass_mask)[0]
        
        # Also get correctly classified samples for comparison
        correct_true_mask = (all_labels == true_idx) & (all_preds == true_idx)
        correct_true_indices = np.where(correct_true_mask)[0]
        
        correct_pred_mask = (all_labels == pred_idx) & (all_preds == pred_idx)
        correct_pred_indices = np.where(correct_pred_mask)[0]
        
        print(f"Found {len(misclass_indices)} instances of {true_class_name}→{pred_class_name}")
        print(f"Found {len(correct_true_indices)} correctly classified {true_class_name}")
        print(f"Found {len(correct_pred_indices)} correctly classified {pred_class_name}\n")
        
        if len(misclass_indices) == 0:
            print("No misclassifications of this type found!")
            return None
        
        # Sample for analysis
        n_misclass = min(n_samples, len(misclass_indices))
        n_correct_true = min(n_samples, len(correct_true_indices))
        n_correct_pred = min(n_samples, len(correct_pred_indices))
        
        sample_misclass = np.random.choice(misclass_indices, n_misclass, replace=False)
        sample_correct_true = np.random.choice(correct_true_indices, n_correct_true, replace=False)
        sample_correct_pred = np.random.choice(correct_pred_indices, n_correct_pred, replace=False)
        
        # Get raw sequences
        misclass_seqs, _ = self.get_raw_sequences_with_predictions(val_loader, sample_misclass)
        correct_true_seqs, _ = self.get_raw_sequences_with_predictions(val_loader, sample_correct_true)
        correct_pred_seqs, _ = self.get_raw_sequences_with_predictions(val_loader, sample_correct_pred)
        
        # Get feature names from preprocessor
        feature_names = self.preprocessor.processed_feature_names
        
        return {
            'misclass_seqs': misclass_seqs,
            'correct_true_seqs': correct_true_seqs,
            'correct_pred_seqs': correct_pred_seqs,
            'misclass_probs': all_probs[sample_misclass],
            'misclass_indices': sample_misclass,
            'feature_names': feature_names,
            'true_class': true_class_name,
            'pred_class': pred_class_name
        }
    
    def plot_feature_distributions(self, analysis_results, save_path=None):
        """Compare feature distributions between misclassified and correct samples"""
        
        misclass = analysis_results['misclass_seqs']
        correct_true = analysis_results['correct_true_seqs']
        correct_pred = analysis_results['correct_pred_seqs']
        feature_names = analysis_results['feature_names']
        true_class = analysis_results['true_class']
        pred_class = analysis_results['pred_class']
        
        n_features = misclass.shape[2]  # shape is [samples, time, features]
        n_rows = (n_features + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
        axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i]
            feature = feature_names[i]
            
            # Calculate mean across time dimension for each sample
            misclass_mean = np.mean(misclass[:, :, i], axis=1)
            correct_true_mean = np.mean(correct_true[:, :, i], axis=1)
            correct_pred_mean = np.mean(correct_pred[:, :, i], axis=1)
            
            # Plot distributions
            ax.hist(correct_true_mean, bins=20, alpha=0.5, label=f'Correct {true_class}', 
                   color='green', density=True)
            ax.hist(misclass_mean, bins=20, alpha=0.5, label=f'Misclassified', 
                   color='red', density=True)
            ax.hist(correct_pred_mean, bins=20, alpha=0.5, label=f'Correct {pred_class}', 
                   color='blue', density=True)
            
            ax.set_xlabel(f'{feature} (mean)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            
            # Add statistical annotation
            true_mean = np.mean(correct_true_mean)
            misclass_mean_val = np.mean(misclass_mean)
            pred_mean = np.mean(correct_pred_mean)
            
            textstr = f'μ_true: {true_mean:.3f}\nμ_misc: {misclass_mean_val:.3f}\nμ_pred: {pred_mean:.3f}'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Feature Distributions: {true_class} vs {pred_class} (Misclassified vs Correct)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_statistics(self, analysis_results, save_path=None):
        """Plot comprehensive statistics: mean, std, min, max"""
        
        misclass = analysis_results['misclass_seqs']
        correct_true = analysis_results['correct_true_seqs']
        correct_pred = analysis_results['correct_pred_seqs']
        feature_names = analysis_results['feature_names']
        true_class = analysis_results['true_class']
        pred_class = analysis_results['pred_class']
        
        n_features = misclass.shape[2]
        stats = ['mean', 'std', 'min', 'max']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for stat_idx, stat in enumerate(stats):
            ax = axes[stat_idx]
            
            # Calculate statistic for each feature
            misclass_stats = []
            correct_true_stats = []
            correct_pred_stats = []
            
            for i in range(n_features):
                if stat == 'mean':
                    misclass_stats.append(np.mean(misclass[:, :, i]))
                    correct_true_stats.append(np.mean(correct_true[:, :, i]))
                    correct_pred_stats.append(np.mean(correct_pred[:, :, i]))
                elif stat == 'std':
                    misclass_stats.append(np.std(misclass[:, :, i]))
                    correct_true_stats.append(np.std(correct_true[:, :, i]))
                    correct_pred_stats.append(np.std(correct_pred[:, :, i]))
                elif stat == 'min':
                    misclass_stats.append(np.min(misclass[:, :, i]))
                    correct_true_stats.append(np.min(correct_true[:, :, i]))
                    correct_pred_stats.append(np.min(correct_pred[:, :, i]))
                elif stat == 'max':
                    misclass_stats.append(np.max(misclass[:, :, i]))
                    correct_true_stats.append(np.max(correct_true[:, :, i]))
                    correct_pred_stats.append(np.max(correct_pred[:, :, i]))
            
            x = np.arange(n_features)
            width = 0.25
            
            ax.bar(x - width, correct_true_stats, width, label=f'Correct {true_class}', 
                  color='green', alpha=0.7)
            ax.bar(x, misclass_stats, width, label='Misclassified', 
                  color='red', alpha=0.7)
            ax.bar(x + width, correct_pred_stats, width, label=f'Correct {pred_class}', 
                  color='blue', alpha=0.7)
            
            ax.set_xlabel('Features', fontsize=11)
            ax.set_ylabel(stat.capitalize(), fontsize=11)
            ax.set_title(f'{stat.capitalize()} by Feature', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names[:n_features], rotation=45, ha='right')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3, axis='y')
        
        plt.suptitle(f'Feature Statistics Comparison: {true_class} → {pred_class}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_time_series_comparison(self, analysis_results, n_examples=3, save_path=None):
        """Plot time series of misclassified vs correct samples"""
        
        misclass = analysis_results['misclass_seqs']
        correct_true = analysis_results['correct_true_seqs']
        correct_pred = analysis_results['correct_pred_seqs']
        feature_names = analysis_results['feature_names']
        true_class = analysis_results['true_class']
        pred_class = analysis_results['pred_class']
        
        n_features = misclass.shape[2]
        n_examples = min(n_examples, len(misclass))
        
        fig, axes = plt.subplots(n_features, 3, figsize=(18, 3*n_features))
        if n_features == 1:
            axes = axes.reshape(1, -1)
        
        for feat_idx in range(n_features):
            for col_idx in range(3):
                ax = axes[feat_idx, col_idx]
                
                if col_idx == 0:  # Correct true class
                    data = correct_true[:n_examples, :, feat_idx]
                    title = f'Correct {true_class}'
                    color = 'green'
                elif col_idx == 1:  # Misclassified
                    data = misclass[:n_examples, :, feat_idx]
                    title = f'Misclassified as {pred_class}'
                    color = 'red'
                else:  # Correct pred class
                    data = correct_pred[:n_examples, :, feat_idx]
                    title = f'Correct {pred_class}'
                    color = 'blue'
                
                # Plot each example
                for i in range(len(data)):
                    ax.plot(data[i], alpha=0.6, linewidth=1.5, color=color)
                
                # Plot mean
                mean_data = np.mean(data, axis=0)
                ax.plot(mean_data, color='black', linewidth=2.5, label='Mean', linestyle='--')
                
                if feat_idx == 0:
                    ax.set_title(title, fontsize=12, fontweight='bold')
                
                ax.set_ylabel(feature_names[feat_idx], fontsize=10)
                
                if feat_idx == n_features - 1:
                    ax.set_xlabel('Time Step', fontsize=10)
                
                ax.grid(alpha=0.3)
                if feat_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=8)
        
        plt.suptitle(f'Time Series Comparison: {true_class} → {pred_class}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_decision_boundary_projection(self, analysis_results, save_path=None):
        """Project data to 2D using PCA and visualize decision boundary region"""
        from sklearn.decomposition import PCA
        
        misclass = analysis_results['misclass_seqs']
        correct_true = analysis_results['correct_true_seqs']
        correct_pred = analysis_results['correct_pred_seqs']
        true_class = analysis_results['true_class']
        pred_class = analysis_results['pred_class']
        
        # Flatten sequences for PCA: [samples, time*features]
        misclass_flat = misclass.reshape(len(misclass), -1)
        correct_true_flat = correct_true.reshape(len(correct_true), -1)
        correct_pred_flat = correct_pred.reshape(len(correct_pred), -1)
        
        # Combine all data for PCA fitting
        all_data = np.vstack([misclass_flat, correct_true_flat, correct_pred_flat])
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca.fit(all_data)
        
        misclass_pca = pca.transform(misclass_flat)
        correct_true_pca = pca.transform(correct_true_flat)
        correct_pred_pca = pca.transform(correct_pred_flat)
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        ax.scatter(correct_true_pca[:, 0], correct_true_pca[:, 1], 
                  c='green', alpha=0.6, s=100, label=f'Correct {true_class}', 
                  edgecolors='black', linewidth=0.5)
        ax.scatter(misclass_pca[:, 0], misclass_pca[:, 1], 
                  c='red', alpha=0.8, s=150, marker='X', 
                  label=f'Misclassified {true_class}→{pred_class}', 
                  edgecolors='black', linewidth=1)
        ax.scatter(correct_pred_pca[:, 0], correct_pred_pca[:, 1], 
                  c='blue', alpha=0.6, s=100, label=f'Correct {pred_class}', 
                  edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax.set_title(f'Decision Boundary Visualization (PCA Projection)\n{true_class} vs {pred_class}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nPCA Explained Variance: {pca.explained_variance_ratio_[0]:.2%} + {pca.explained_variance_ratio_[1]:.2%} = {sum(pca.explained_variance_ratio_[:2]):.2%}")
    
    def analyze_feature_importance_for_confusion(self, analysis_results):
        """Calculate which features differ most between misclassified and correct samples"""
        
        misclass = analysis_results['misclass_seqs']
        correct_true = analysis_results['correct_true_seqs']
        feature_names = analysis_results['feature_names']
        true_class = analysis_results['true_class']
        pred_class = analysis_results['pred_class']
        
        n_features = misclass.shape[2]
        
        print(f"\n{'='*80}")
        print(f"FEATURE IMPORTANCE FOR {true_class} → {pred_class} CONFUSION")
        print(f"{'='*80}\n")
        
        importance_scores = []
        
        for i in range(n_features):
            # Calculate statistics across all samples and time
            misclass_mean = np.mean(misclass[:, :, i])
            correct_true_mean = np.mean(correct_true[:, :, i])
            misclass_std = np.std(misclass[:, :, i])
            correct_true_std = np.std(correct_true[:, :, i])
            
            # Distance between distributions (normalized)
            mean_diff = abs(misclass_mean - correct_true_mean)
            std_avg = (misclass_std + correct_true_std) / 2
            normalized_diff = mean_diff / (std_avg + 1e-8)
            
            importance_scores.append({
                'feature': feature_names[i],
                'misclass_mean': misclass_mean,
                'correct_mean': correct_true_mean,
                'mean_diff': mean_diff,
                'importance': normalized_diff
            })
        
        # Sort by importance
        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        print("Features ranked by importance in misclassification:\n")
        print(importance_df.to_string(index=False))
        
        print(f"\n{'='*80}\n")
        
        return importance_df
    
    def generate_misclassification_report(self, val_loader, 
                                         true_class_name, pred_class_name,
                                         save_dir='misclass_analysis'):
        """Generate complete analysis report for specific misclassification"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"MISCLASSIFICATION ANALYSIS: {true_class_name} → {pred_class_name}")
        print(f"{'='*80}\n")
        
        # Analyze misclassification
        analysis_results = self.analyze_specific_misclassification(
            val_loader, true_class_name, pred_class_name, n_samples=30
        )
        
        if analysis_results is None:
            return None
        
        '''
        # Generate visualizations
        print("\n📊 Generating feature distributions...")
        self.plot_feature_distributions(analysis_results, 
                                       save_path=f'{save_dir}/feature_distributions.png')
        
        print("\n📈 Generating feature statistics...")
        self.plot_feature_statistics(analysis_results,
                                    save_path=f'{save_dir}/feature_statistics.png')
        
        print("\n📉 Generating time series comparison...")
        self.plot_time_series_comparison(analysis_results, n_examples=5,
                                        save_path=f'{save_dir}/time_series_comparison.png')
        
        print("\n🎯 Generating decision boundary projection...")
        self.plot_decision_boundary_projection(analysis_results,
                                              save_path=f'{save_dir}/decision_boundary_pca.png')
        '''
        
        # Feature importance analysis
        importance_df = self.analyze_feature_importance_for_confusion(analysis_results)
        importance_df.to_csv(f'{save_dir}/feature_importance.csv', index=False)
        
        print(f"\n✅ Analysis complete! Results saved to '{save_dir}/'")
        print(f"{'='*80}\n")
        
        return analysis_results, importance_df


# Convenience function
def analyze_misclassification(checkpoint_path, data_dir, 
                             true_class='stand', pred_class='walk'):
    """
    Complete convenience function to load model and analyze misclassifications
    
    Usage:
        analyzer, results = analyze_misclassification(
            'models/gait_model_20241023_123456.pth',
            'data/processed',
            true_class='stand',
            pred_class='walk'
        )
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocessor, checkpoint = load_model_comprehensive(checkpoint_path, device)
    
    # Recreate dataloaders
    train_loader, val_loader = recreate_dataloaders_from_checkpoint(
        checkpoint, data_dir, device
    )
    
    # Run analysis
    analyzer = FeatureMisclassificationAnalyzer(model, preprocessor, device)
    results = analyzer.generate_misclassification_report(
        val_loader, true_class, pred_class,
        save_dir=f'misclass_analysis_{true_class}_to_{pred_class}'
    )
    
    return analyzer, results

class ModelAnalyzer:
    """Comprehensive model analysis for gait classification"""
    
    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.class_names = preprocessor.label_encoder.classes_
        
    def get_predictions_and_probabilities(self, data_loader):
        """Get predictions, probabilities, and true labels"""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                outputs = self.model(batch_X)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs), np.array(all_labels)
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True, save_path=None):
        """Plot confusion matrix with percentages"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_norm
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            cm_display = cm
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add counts as text if normalized
        if normalize:
            for i in range(len(self.class_names)):
                for j in range(len(self.class_names)):
                    plt.text(j + 0.5, i + 0.7, f'({cm[i,j]})',
                            ha='center', va='center', fontsize=9, color='gray')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_per_class_metrics(self, y_true, y_pred, save_path=None):
        """Plot detailed per-class metrics"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Precision
        axes[0, 0].bar(self.class_names, precision, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Precision by Class', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_ylim([0, 1.05])
        for i, v in enumerate(precision):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Recall
        axes[0, 1].bar(self.class_names, recall, color='forestgreen', alpha=0.7)
        axes[0, 1].set_title('Recall by Class', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_ylim([0, 1.05])
        for i, v in enumerate(recall):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # F1-Score
        axes[1, 0].bar(self.class_names, f1, color='coral', alpha=0.7)
        axes[1, 0].set_title('F1-Score by Class', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_ylim([0, 1.05])
        for i, v in enumerate(f1):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Support
        axes[1, 1].bar(self.class_names, support, color='mediumpurple', alpha=0.7)
        axes[1, 1].set_title('Support (Sample Count) by Class', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Samples')
        for i, v in enumerate(support):
            axes[1, 1].text(i, v + max(support)*0.02, f'{int(v)}', 
                           ha='center', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true, y_probs, save_path=None):
        """Plot ROC curves for each class"""
        n_classes = len(self.class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_misclassifications(self, y_true, y_pred, y_probs, top_n=20):
        """Analyze the most confident misclassifications"""
        misclass_indices = np.where(y_true != y_pred)[0]
        
        if len(misclass_indices) == 0:
            print("No misclassifications found!")
            return pd.DataFrame()
        
        misclass_data = []
        for idx in misclass_indices:
            true_class = self.class_names[y_true[idx]]
            pred_class = self.class_names[y_pred[idx]]
            confidence = y_probs[idx, y_pred[idx]]
            true_prob = y_probs[idx, y_true[idx]]
            
            misclass_data.append({
                'index': idx,
                'true_class': true_class,
                'pred_class': pred_class,
                'confidence': confidence,
                'true_class_prob': true_prob,
                'confusion_margin': confidence - true_prob
            })
        
        df = pd.DataFrame(misclass_data)
        df = df.sort_values('confidence', ascending=False).head(top_n)
        
        print(f"\n{'='*80}")
        print(f"TOP {top_n} MOST CONFIDENT MISCLASSIFICATIONS")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        return df
    
    def plot_confidence_distribution(self, y_true, y_pred, y_probs, save_path=None):
        """Plot confidence distribution for correct vs incorrect predictions"""
        correct_mask = y_true == y_pred
        correct_confidences = np.max(y_probs[correct_mask], axis=1)
        incorrect_confidences = np.max(y_probs[~correct_mask], axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(correct_confidences, bins=30, alpha=0.6, label='Correct', 
                     color='green', edgecolor='black')
        axes[0].hist(incorrect_confidences, bins=30, alpha=0.6, label='Incorrect', 
                     color='red', edgecolor='black')
        axes[0].set_xlabel('Prediction Confidence', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        data_to_plot = [correct_confidences, incorrect_confidences]
        bp = axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'],
                             patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[1].set_ylabel('Prediction Confidence', fontsize=12)
        axes[1].set_title('Confidence Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        
        # Add statistics
        axes[1].text(1, np.median(correct_confidences), 
                    f'Med: {np.median(correct_confidences):.3f}',
                    ha='center', va='bottom', fontweight='bold')
        axes[1].text(2, np.median(incorrect_confidences), 
                    f'Med: {np.median(incorrect_confidences):.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nConfidence Statistics:")
        print(f"Correct predictions - Mean: {np.mean(correct_confidences):.3f}, "
              f"Std: {np.std(correct_confidences):.3f}")
        print(f"Incorrect predictions - Mean: {np.mean(incorrect_confidences):.3f}, "
              f"Std: {np.std(incorrect_confidences):.3f}")
    
    def plot_training_history(self, train_losses, val_losses, val_accuracies, save_path=None):
        """Plot comprehensive training history"""
        epochs = range(len(train_losses))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss curves
        axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2, color='blue')
        axes[0].plot(epochs, val_losses, label='Val Loss', linewidth=2, color='orange')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(epochs, val_accuracies, linewidth=2, color='green')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].axhline(y=max(val_accuracies), color='r', linestyle='--', 
                       alpha=0.5, label=f'Best: {max(val_accuracies):.2f}%')
        axes[1].legend()
        
        # Loss ratio (overfitting indicator)
        loss_ratio = np.array(val_losses) / np.array(train_losses)
        axes[2].plot(epochs, loss_ratio, linewidth=2, color='purple')
        axes[2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Overfitting')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Val Loss / Train Loss', fontsize=12)
        axes[2].set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_hyperparameter_report(self, model, train_losses, val_losses, val_accuracies):
        """Generate recommendations for hyperparameter tuning"""
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING RECOMMENDATIONS")
        print(f"{'='*80}\n")
        
        # Model architecture
        param_count = sum(p.numel() for p in model.parameters())
        print(f"📊 Model Complexity:")
        print(f"   Total Parameters: {param_count:,}")
        
        # Check if model is too simple/complex
        if param_count < 50000:
            print(f"   ⚠️  Model might be too simple. Consider:")
            print(f"      - Increasing num_channels (e.g., [128, 256, 512])")
            print(f"      - Adding more TCN blocks")
        elif param_count > 500000:
            print(f"   ⚠️  Model might be too complex. Consider:")
            print(f"      - Reducing num_channels")
            print(f"      - Adding more regularization (dropout)")
        else:
            print(f"   ✓ Model complexity seems reasonable")
        
        # Training dynamics
        print(f"\n📈 Training Dynamics:")
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        loss_ratio = final_val_loss / final_train_loss
        best_epoch = np.argmax(val_accuracies)
        convergence_epoch = len(train_losses)
        
        print(f"   Final Train Loss: {final_train_loss:.4f}")
        print(f"   Final Val Loss: {final_val_loss:.4f}")
        print(f"   Loss Ratio: {loss_ratio:.3f}")
        print(f"   Best Epoch: {best_epoch + 1}/{convergence_epoch}")
        
        # Overfitting analysis
        print(f"\n🎯 Overfitting Analysis:")
        if loss_ratio > 1.5:
            print(f"   ⚠️  Significant overfitting detected! Try:")
            print(f"      - Increase dropout (current likely 0.3, try 0.4-0.5)")
            print(f"      - Increase weight_decay (try 1e-4 or 1e-3)")
            print(f"      - Reduce model capacity")
            print(f"      - Use data augmentation")
        elif loss_ratio > 1.2:
            print(f"   ⚠️  Mild overfitting. Consider:")
            print(f"      - Slightly increase dropout")
            print(f"      - Add L2 regularization")
        else:
            print(f"   ✓ No significant overfitting")
        
        # Convergence analysis
        print(f"\n🔄 Convergence Analysis:")
        val_improvement = val_accuracies[-1] - val_accuracies[0]
        if best_epoch < convergence_epoch * 0.3:
            print(f"   ⚠️  Model converged very early. Try:")
            print(f"      - Decrease learning rate (try 5e-4 or 1e-4)")
            print(f"      - Use learning rate warmup")
        elif best_epoch > convergence_epoch * 0.8:
            print(f"   ⚠️  Model still improving at end. Try:")
            print(f"      - Increase max epochs")
            print(f"      - Increase patience for early stopping")
        else:
            print(f"   ✓ Convergence timing looks good")
        
        if val_improvement < 10:
            print(f"   ⚠️  Limited improvement ({val_improvement:.1f}%). Try:")
            print(f"      - Increase learning rate (try 2e-3 or 5e-3)")
            print(f"      - Change architecture")
        
        # Learning rate schedule
        print(f"\n📉 Learning Rate Schedule:")
        print(f"   Current: CosineAnnealingWarmRestarts")
        print(f"   Alternatives to try:")
        print(f"      - ReduceLROnPlateau (adapts to validation loss)")
        print(f"      - OneCycleLR (faster convergence)")
        print(f"      - Simple exponential decay")
        
        # Sequence parameters
        print(f"\n⏱️  Sequence Parameters to Experiment With:")
        print(f"      - sequence_length: Try 40, 60, 80, 120")
        print(f"      - stride: Try sequence_length//2 or sequence_length//4")
        print(f"      - Higher overlap (smaller stride) = more data, slower training")
        
        # Batch size
        print(f"\n📦 Batch Size Recommendations:")
        print(f"      - Current: 64 (default)")
        print(f"      - Try: 32 (more updates, less memory), 128 (faster, more stable)")
        print(f"      - Larger batch → increase learning rate proportionally")
        
        print(f"\n{'='*80}\n")
    
    def generate_full_report(self, train_loader, val_loader, 
                            train_losses, val_losses, val_accuracies,
                            save_dir='analysis_results'):
        """Generate complete analysis report with all visualizations"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE MODEL ANALYSIS REPORT")
        print("="*80 + "\n")
        
        # Get predictions
        print("📊 Computing predictions...")
        val_preds, val_probs, val_labels = self.get_predictions_and_probabilities(val_loader)
        
        # 1. Classification Report
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(val_labels, val_preds, 
                                   target_names=self.class_names, digits=3))
        
        # 2. Confusion Matrix
        print("\n📈 Generating confusion matrix...")
        self.plot_confusion_matrix(val_labels, val_preds, normalize=True,
                                   save_path=f'{save_dir}/confusion_matrix.png')
        
        # 3. Per-class metrics
        print("\n📊 Generating per-class metrics...")
        self.plot_per_class_metrics(val_labels, val_preds,
                                    save_path=f'{save_dir}/per_class_metrics.png')
        
        # 4. ROC Curves
        print("\n📈 Generating ROC curves...")
        self.plot_roc_curves(val_labels, val_probs,
                            save_path=f'{save_dir}/roc_curves.png')
        
        # 5. Confidence Analysis
        print("\n📊 Analyzing prediction confidence...")
        self.plot_confidence_distribution(val_labels, val_preds, val_probs,
                                         save_path=f'{save_dir}/confidence_dist.png')
        
        # 6. Misclassification Analysis
        print("\n🔍 Analyzing misclassifications...")
        misclass_df = self.analyze_misclassifications(val_labels, val_preds, val_probs)
        if not misclass_df.empty:
            misclass_df.to_csv(f'{save_dir}/misclassifications.csv', index=False)
        
        # 7. Training History
        print("\n📈 Plotting training history...")
        self.plot_training_history(train_losses, val_losses, val_accuracies,
                                   save_path=f'{save_dir}/training_history.png')
        
        # 8. Hyperparameter Report
        self.generate_hyperparameter_report(self.model, train_losses, 
                                           val_losses, val_accuracies)
        
        print(f"\n✅ Analysis complete! All results saved to '{save_dir}/'")
        print("="*80 + "\n")


# Convenience function to run full analysis
def analyze_trained_model(model, preprocessor, train_loader, val_loader,
                         train_losses, val_losses, val_accuracies, device):
    """Run complete analysis on a trained model"""
    analyzer = ModelAnalyzer(model, preprocessor, device)
    analyzer.generate_full_report(train_loader, val_loader,
                                  train_losses, val_losses, val_accuracies)
    return analyzer

class GaitDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.support_encoder = LabelEncoder()
        self.original_feature_names = None
        self.processed_feature_names = None
        self.is_fitted = False
        
        # Predefined labels
        known_gait_types = ['walk', 'jog', 'stand'] 
        known_support_types = ['double', 'single', 'flight']
        
        self.label_encoder.fit(known_gait_types)
        self.support_encoder.fit(known_support_types)
        
        print(f"Predefined gait types: {list(self.label_encoder.classes_)}")
        print(f"Predefined support types: {list(self.support_encoder.classes_)}")
    
    def prepare_features(self, df, feature_names):
        """Prepare features - encode categorical variables"""
        available_features = [f for f in feature_names if f in df.columns]
        features = df[available_features].copy()
        
        # Encode support_type
        if 'support_type' in features.columns:
            if features['support_type'].dtype == 'object':
                features['support_type_encoded'] = self.support_encoder.transform(features['support_type'])
                features = features.drop('support_type', axis=1)
        
        self.processed_feature_names = list(features.columns)
        return features
    
    def create_sequences_from_files(self, file_list, feature_names, sequence_length=60, 
                                    stride=30, min_purity=0.9):
        """Create sequences, skipping ambiguous label windows"""
        all_sequences = []
        all_labels = []
        
        for file_path in file_list:
            df = pd.read_csv(file_path)
            features = self.prepare_features(df, feature_names)
            labels = self.label_encoder.transform(df['gait_type'])
            
            for i in range(0, len(features) - sequence_length + 1, stride):
                label_window = labels[i:i+sequence_length]
                
                # Check label purity
                unique, counts = np.unique(label_window, return_counts=True)
                purity = counts.max() / len(label_window)
                
                if purity >= min_purity:  # At least 90% frames agree
                    label = unique[counts.argmax()]
                    all_sequences.append(features.iloc[i:i+sequence_length].values)
                    all_labels.append(label)
        
        print(f"Kept {len(all_sequences)} pure sequences (discarded ambiguous ones)")
        return np.array(all_sequences), np.array(all_labels)
    
    def fit_transform(self, train_files, feature_names, sequence_length=60, stride=30):
        """Fit scaler on ALL training data, then create sequences"""
        self.is_fitted = True
        self.original_feature_names = feature_names
        
        # Step 1: Load ALL training data and concatenate
        all_train_features = []
        for file_path in train_files:
            df = pd.read_csv(file_path)
            features = self.prepare_features(df, feature_names)
            all_train_features.append(features)
        
        all_train_features = pd.concat(all_train_features, ignore_index=True)
        
        # Step 2: Fit scaler on ALL training data
        print(f"Fitting scaler on {len(all_train_features)} total training samples")
        self.scaler.fit(all_train_features)
        
        # Step 3: Create sequences from each file with fitted scaler
        X_sequences, y_sequences = self.create_sequences_from_files(
            train_files, feature_names, sequence_length, stride
        )
        
        # Step 4: Scale the sequences
        n_samples, seq_len, n_features = X_sequences.shape
        X_flat = X_sequences.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_flat)
        X_sequences = X_scaled.reshape(n_samples, seq_len, n_features)
        
        print(f"Created {len(X_sequences)} training sequences with {n_features} features")
        print(f"Sequence shape: {X_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def transform(self, val_files, sequence_length=60, stride=30):
        """Transform validation data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences_from_files(
            val_files, self.original_feature_names, sequence_length, stride
        )
        
        # Scale the sequences
        n_samples, seq_len, n_features = X_sequences.shape
        X_flat = X_sequences.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_flat)
        X_sequences = X_scaled.reshape(n_samples, seq_len, n_features)
        
        print(f"Created {len(X_sequences)} validation sequences")
        
        return X_sequences, y_sequences


class TemporalBlock(nn.Module):
    """TCN residual block with weight normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                     padding=padding, dilation=dilation)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                     padding=padding, dilation=dilation)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.downsample = (nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, 1))
                          if in_channels != out_channels else None)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Causal truncation
        out = out[:, :, :x.size(2)]
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = residual[:, :, :x.size(2)]
            
        return self.relu(out + residual)


class GaitTCN(nn.Module):
    """Simplified TCN for gait classification"""
    def __init__(self, num_features, num_classes,
                 num_channels=[64, 128, 256],
                 kernel_size=7, dropout=0.3):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(num_features, num_channels[0], 1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # TCN blocks
        layers = []
        in_channels = num_channels[0]
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_channels, out_channels, 
                            kernel_size, dilation, dropout)
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.input_proj(x)
        x = self.tcn(x)
        logits = self.classifier(x)
        return logits
    
def setup_device():
    """Setup device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f":) Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(":( Using CPU")
    return device

def save_model_comprehensive(model, preprocessor, train_loader, val_loader,
                            train_losses, val_losses, val_accuracies,
                            config, epoch, save_path='best_model.pth'):
    """
    Save model with all necessary information for easy loading and analysis
    """
    # Get model architecture info
    state_dict = model.state_dict()
    
    # Extract architecture from model
    num_features = model.input_proj[0].in_channels
    num_classes = model.classifier[2].out_features
    
    # Try to extract num_channels from TCN layers
    num_channels = []
    for name, param in state_dict.items():
        if 'tcn' in name and 'conv1.weight' in name:
            num_channels.append(param.shape[0])
    
    # Get kernel size and dropout from config or use defaults
    kernel_size = config.get('kernel_size', 7)
    dropout = config.get('dropout', 0.3)
    
    checkpoint = {
        # Model architecture info
        'architecture': {
            'num_features': num_features,
            'num_classes': num_classes,
            'num_channels': num_channels if num_channels else [64, 128, 256],
            'kernel_size': kernel_size,
            'dropout': dropout,
        },
        
        # Model weights
        'model_state_dict': state_dict,
        
        # Preprocessor (already fitted)
        'preprocessor': preprocessor,
        
        # Training history
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': max(val_accuracies) if val_accuracies else 0,
        'final_epoch': epoch,
        
        # Training configuration
        'config': config,
        
        # Feature names
        'feature_names': preprocessor.original_feature_names,
        'processed_feature_names': preprocessor.processed_feature_names,
        
        # Class information
        'class_names': list(preprocessor.label_encoder.classes_),
        
        # Metadata
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pytorch_version': torch.__version__,
        
        # Note: Can't save DataLoaders directly, but we save the info to recreate them
        'data_info': {
            'batch_size': train_loader.batch_size,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
        }
    }
    
    torch.save(checkpoint, save_path)
    print(f"\n✓ Model saved to {save_path}")
    print(f"  Best validation accuracy: {checkpoint['best_val_accuracy']:.2f}%")
    
    # Also save a JSON file with human-readable info
    info_path = save_path.replace('.pth', '_info.json')
    info = {
        'architecture': checkpoint['architecture'],
        'best_val_accuracy': float(checkpoint['best_val_accuracy']),
        'final_epoch': int(epoch),
        'config': {k: (v if not isinstance(v, np.ndarray) else v.tolist()) 
                   for k, v in config.items()},
        'feature_names': checkpoint['feature_names'],
        'class_names': checkpoint['class_names'],
        'timestamp': checkpoint['timestamp'],
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✓ Model info saved to {info_path}")


def load_model_comprehensive(checkpoint_path, device=None):
    """
    Load model with all information for easy inference or analysis
    Returns: model, preprocessor, checkpoint_dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get architecture info
    arch = checkpoint['architecture']
    
    # Reconstruct model
    model = GaitTCN(
        num_features=arch['num_features'],
        num_classes=arch['num_classes'],
        num_channels=arch['num_channels'],
        kernel_size=arch['kernel_size'],
        dropout=arch['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get preprocessor
    preprocessor = checkpoint['preprocessor']
    
    print(f"✓ Model loaded successfully")
    print(f"  Architecture: {arch['num_channels']}")
    print(f"  Features: {arch['num_features']}")
    print(f"  Classes: {arch['num_classes']} {checkpoint['class_names']}")
    print(f"  Best accuracy: {checkpoint['best_val_accuracy']:.2f}%")
    print(f"  Trained: {checkpoint['timestamp']}")
    
    return model, preprocessor, checkpoint


def recreate_dataloaders_from_checkpoint(checkpoint, data_dir, device):
    """
    Recreate data loaders using saved preprocessor and config
    """
    preprocessor = checkpoint['preprocessor']
    config = checkpoint['config']
    feature_names = checkpoint['feature_names']
    
    # Get files and split (same way as training)
    take_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    np.random.seed(42)
    np.random.shuffle(take_files)
    split_idx = int(len(take_files) * 0.6)
    train_files = take_files[:split_idx]
    val_files = take_files[split_idx:]
    
    # Use saved preprocessor to transform
    X_train, y_train = preprocessor.transform(
        train_files, 
        config.get('sequence_length', 60),
        config.get('stride', 30)
    )
    X_val, y_val = preprocessor.transform(
        val_files,
        config.get('sequence_length', 60),
        config.get('stride', 30)
    )
    
    # Create loaders
    batch_size = checkpoint['data_info']['batch_size']
    
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train).to(device),
            torch.LongTensor(y_train).to(device)
        ),
        batch_size=batch_size, shuffle=False
    )
    
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_val).to(device),
            torch.LongTensor(y_val).to(device)
        ),
        batch_size=batch_size, shuffle=False
    )
    
    print(f"✓ Recreated data loaders")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    return train_loader, val_loader

from sklearn.utils.class_weight import compute_class_weight


# dropout may 0.2 or 0.3 or 0.5 based on experiments
# Reduced channels from 64, 128, 256 for complexity
# Consider replaceing the single 7x7 kernal with three 
def train_model(data_dir, feature_names, 
                              sequence_length=60, stride=30,
                              num_channels=[32, 64, 128],
                              kernel_size=6, dropout=0.5,
                              learning_rate=1e-3, weight_decay=5e-3,
                              batch_size=64, max_epochs=25,
                              run_analysis=True, save_dir='./models'):
    """
    Enhanced training with automatic analysis
    """
    device = setup_device()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'gait_model_{timestamp}'
    save_path = os.path.join(save_dir, f'{model_name}.pth')
    
    print(f"\n{'='*80}")
    print(f"TRAINING GAIT CLASSIFICATION MODEL")
    print(f"{'='*80}")
    print(f"Model will be saved to: {save_path}")
    print(f"{'='*80}\n")
    
    # Store configuration
    config = {
        'sequence_length': sequence_length,
        'stride': stride,
        'num_channels': num_channels,
        'kernel_size': kernel_size,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
    }
    
    # Get all files and split
    take_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"Found {len(take_files)} take files")
    
    np.random.seed(42)
    np.random.shuffle(take_files)
    split_idx = int(len(take_files) * 0.6)
    train_files = take_files[:split_idx]
    val_files = take_files[split_idx:]
    
    print(f"Training takes: {len(train_files)}")
    print(f"Validation takes: {len(val_files)}")
    
    # Initialize preprocessor and fit on training data
    preprocessor = GaitDataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(
        train_files, feature_names, sequence_length, stride
    )
    X_val, y_val = preprocessor.transform(
        val_files, sequence_length, stride
    )
    
    print(f"\nFinal shapes:")
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Val: {X_val.shape}, {y_val.shape}")
    print(f"Label distribution (train): {np.bincount(y_train)}")
    print(f"Label distribution (val): {np.bincount(y_val)}")

    class_weights = compute_class_weight(
        class_weight = 'balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor), 
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor), 
        batch_size=batch_size, shuffle=False
    )
    
    # Initialize model
    model = GaitTCN(
        num_features=X_train.shape[2],
        num_classes=len(preprocessor.label_encoder.classes_),
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Training loop
    best_val_acc = 0
    train_losses, val_losses, val_accuracies = [], [], []
    patience_counter = 0
    max_patience = 20
    
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # Metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        scheduler.step(val_loss_avg)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}: LR={current_lr:.2e}, '
                  f'Train Loss: {train_loss_avg:.4f}, '
                  f'Val Loss: {val_loss_avg:.4f}, '
                  f'Val Acc: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            
            # Save with comprehensive information
            save_model_comprehensive(
                model, preprocessor, train_loader, val_loader,
                train_losses, val_losses, val_accuracies,
                config, epoch, save_path
            )
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    print(f"{'='*80}\n")
    
    # Run analysis if requested
    if run_analysis:
        print(f"\n{'='*80}")
        print("RUNNING POST-TRAINING ANALYSIS")
        print(f"{'='*80}\n")
        
        # Load best model

        model.load_state_dict(torch.load(save_path, weights_only=False)['model_state_dict'])

        # Create analysis directory
        analysis_dir = os.path.join(save_dir, f'{model_name}_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Run comprehensive analysis
        analyzer = ModelAnalyzer(model, preprocessor, device)
        analyzer.generate_full_report(
            train_loader, val_loader,
            train_losses, val_losses, val_accuracies,
            save_dir=analysis_dir
        )
        true_class='stand'
        pred_class='walk'

        analyzer = FeatureMisclassificationAnalyzer(model, preprocessor, device)
        results = analyzer.generate_misclassification_report(
            val_loader, true_class, pred_class,
            save_dir=f'misclass_analysis_{true_class}_to_{pred_class}'
        )
        
        print(f"\n✓ Analysis complete! Results saved to: {analysis_dir}")
    
    return model, preprocessor, train_losses, val_losses, val_accuracies, save_path