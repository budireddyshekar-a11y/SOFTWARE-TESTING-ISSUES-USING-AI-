"""
AI Quality Assurance (AI-QA) Framework
A comprehensive solution for testing AI systems and addressing software testing issues in AI.

This framework provides modules for:
1. Data Quality & Bias Auditing
2. Model-Agnostic Testing
3. Fairness & Ethics Assessment
4. Continuous Monitoring & Drift Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataQualityAuditor:
    """
    Module for auditing data quality and detecting potential biases in datasets.
    """
    
    def __init__(self):
        self.audit_results = {}
    
    def profile_data(self, data, target_column=None):
        """
        Generate a comprehensive profile of the dataset.
        
        Args:
            data (pd.DataFrame): The dataset to profile
            target_column (str): Name of the target variable column
        
        Returns:
            dict: Data profiling results
        """
        profile = {
            'shape': data.shape,
            'columns': list(data.columns),
            'data_types': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'numerical_summary': data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {}
        }
        
        # Analyze categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            profile['categorical_summary'][col] = {
                'unique_values': data[col].nunique(),
                'value_counts': data[col].value_counts().to_dict()
            }
        
        # Target variable analysis
        if target_column and target_column in data.columns:
            profile['target_analysis'] = {
                'distribution': data[target_column].value_counts().to_dict(),
                'balance_ratio': data[target_column].value_counts().min() / data[target_column].value_counts().max()
            }
        
        self.audit_results['data_profile'] = profile
        return profile
    
    def detect_bias(self, data, protected_attributes, target_column):
        """
        Detect potential biases in the dataset based on protected attributes.
        
        Args:
            data (pd.DataFrame): The dataset to analyze
            protected_attributes (list): List of column names representing protected attributes
            target_column (str): Name of the target variable column
        
        Returns:
            dict: Bias detection results
        """
        bias_results = {}
        
        for attr in protected_attributes:
            if attr in data.columns:
                # Calculate outcome rates for each group
                group_outcomes = data.groupby(attr)[target_column].agg(['count', 'mean']).reset_index()
                group_outcomes.columns = [attr, 'count', 'positive_rate']
                
                # Calculate disparate impact ratio
                max_rate = group_outcomes['positive_rate'].max()
                min_rate = group_outcomes['positive_rate'].min()
                disparate_impact = min_rate / max_rate if max_rate > 0 else 0
                
                bias_results[attr] = {
                    'group_outcomes': group_outcomes.to_dict('records'),
                    'disparate_impact_ratio': disparate_impact,
                    'bias_detected': disparate_impact < 0.8  # 80% rule
                }
        
        self.audit_results['bias_analysis'] = bias_results
        return bias_results
    
    def visualize_data_distribution(self, data, save_path=None):
        """
        Create visualizations for data distribution analysis.
        
        Args:
            data (pd.DataFrame): The dataset to visualize
            save_path (str): Path to save the visualization
        """
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Create subplots
        n_numerical = len(numerical_cols)
        n_categorical = len(categorical_cols)
        total_plots = n_numerical + n_categorical
        
        if total_plots == 0:
            return
        
        fig, axes = plt.subplots(total_plots, 1, figsize=(12, 4 * total_plots))
        if total_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot numerical distributions
        for col in numerical_cols:
            axes[plot_idx].hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[plot_idx].set_title(f'Distribution of {col}')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel('Frequency')
            plot_idx += 1
        
        # Plot categorical distributions
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            axes[plot_idx].bar(range(len(value_counts)), value_counts.values)
            axes[plot_idx].set_title(f'Distribution of {col}')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel('Count')
            axes[plot_idx].set_xticks(range(len(value_counts)))
            axes[plot_idx].set_xticklabels(value_counts.index, rotation=45)
            plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ModelAgnosticTester:
    """
    Module for model-agnostic testing including metamorphic testing and adversarial testing.
    """
    
    def __init__(self):
        self.test_results = {}
    
    def metamorphic_test(self, model, X_test, transformation_func, relation_func, test_name):
        """
        Perform metamorphic testing on a model.
        
        Args:
            model: Trained model with predict method
            X_test: Test data
            transformation_func: Function to transform input data
            relation_func: Function to check metamorphic relation
            test_name: Name of the test
        
        Returns:
            dict: Test results
        """
        # Get original predictions
        original_predictions = model.predict(X_test)
        
        # Transform input data
        transformed_X = transformation_func(X_test)
        
        # Get predictions on transformed data
        transformed_predictions = model.predict(transformed_X)
        
        # Check metamorphic relation
        violations = []
        for i in range(len(original_predictions)):
            if not relation_func(original_predictions[i], transformed_predictions[i]):
                violations.append(i)
        
        violation_rate = len(violations) / len(original_predictions)
        
        result = {
            'test_name': test_name,
            'total_cases': len(original_predictions),
            'violations': len(violations),
            'violation_rate': violation_rate,
            'violation_indices': violations
        }
        
        self.test_results[test_name] = result
        return result
    
    def adversarial_test(self, model, X_test, y_test, epsilon=0.1):
        """
        Perform basic adversarial testing using FGSM-like perturbations.
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels
            epsilon: Perturbation magnitude
        
        Returns:
            dict: Adversarial test results
        """
        # Generate random perturbations (simplified adversarial examples)
        perturbations = np.random.normal(0, epsilon, X_test.shape)
        adversarial_X = X_test + perturbations
        
        # Get predictions on original and adversarial examples
        original_predictions = model.predict(X_test)
        adversarial_predictions = model.predict(adversarial_X)
        
        # Calculate robustness metrics
        original_accuracy = accuracy_score(y_test, original_predictions)
        adversarial_accuracy = accuracy_score(y_test, adversarial_predictions)
        
        # Count prediction changes
        prediction_changes = np.sum(original_predictions != adversarial_predictions)
        change_rate = prediction_changes / len(original_predictions)
        
        result = {
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'accuracy_drop': original_accuracy - adversarial_accuracy,
            'prediction_changes': prediction_changes,
            'change_rate': change_rate,
            'epsilon': epsilon
        }
        
        self.test_results['adversarial_test'] = result
        return result


class FairnessAssessor:
    """
    Module for assessing fairness and detecting bias in model predictions.
    """
    
    def __init__(self):
        self.fairness_results = {}
    
    def calculate_fairness_metrics(self, y_true, y_pred, protected_attribute):
        """
        Calculate various fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute values
        
        Returns:
            dict: Fairness metrics
        """
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'protected': protected_attribute
        })
        
        # Calculate metrics for each group
        groups = df['protected'].unique()
        group_metrics = {}
        
        for group in groups:
            group_data = df[df['protected'] == group]
            
            # True Positive Rate (Sensitivity)
            tp = np.sum((group_data['y_true'] == 1) & (group_data['y_pred'] == 1))
            fn = np.sum((group_data['y_true'] == 1) & (group_data['y_pred'] == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # False Positive Rate
            fp = np.sum((group_data['y_true'] == 0) & (group_data['y_pred'] == 1))
            tn = np.sum((group_data['y_true'] == 0) & (group_data['y_pred'] == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Positive Prediction Rate
            ppr = np.sum(group_data['y_pred'] == 1) / len(group_data)
            
            # Accuracy
            accuracy = np.sum(group_data['y_true'] == group_data['y_pred']) / len(group_data)
            
            group_metrics[group] = {
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'positive_prediction_rate': ppr,
                'accuracy': accuracy,
                'sample_size': len(group_data)
            }
        
        # Calculate fairness metrics
        group_values = list(group_metrics.values())
        
        # Demographic Parity (Statistical Parity)
        ppr_values = [metrics['positive_prediction_rate'] for metrics in group_values]
        demographic_parity_diff = max(ppr_values) - min(ppr_values)
        
        # Equalized Odds
        tpr_values = [metrics['true_positive_rate'] for metrics in group_values]
        fpr_values = [metrics['false_positive_rate'] for metrics in group_values]
        equalized_odds_diff = max(abs(max(tpr_values) - min(tpr_values)), 
                                 abs(max(fpr_values) - min(fpr_values)))
        
        # Equal Opportunity
        equal_opportunity_diff = max(tpr_values) - min(tpr_values)
        
        fairness_metrics = {
            'group_metrics': group_metrics,
            'demographic_parity_difference': demographic_parity_diff,
            'equalized_odds_difference': equalized_odds_diff,
            'equal_opportunity_difference': equal_opportunity_diff,
            'fairness_satisfied': {
                'demographic_parity': demographic_parity_diff < 0.1,
                'equalized_odds': equalized_odds_diff < 0.1,
                'equal_opportunity': equal_opportunity_diff < 0.1
            }
        }
        
        self.fairness_results['fairness_metrics'] = fairness_metrics
        return fairness_metrics


class DriftDetector:
    """
    Module for detecting data drift and model performance degradation.
    """
    
    def __init__(self):
        self.drift_results = {}
        self.baseline_stats = {}
    
    def set_baseline(self, data, model=None):
        """
        Set baseline statistics for drift detection.
        
        Args:
            data: Baseline dataset
            model: Trained model (optional)
        """
        self.baseline_stats = {
            'mean': data.mean().to_dict(),
            'std': data.std().to_dict(),
            'min': data.min().to_dict(),
            'max': data.max().to_dict(),
            'shape': data.shape
        }
        
        if model is not None:
            # Store baseline model performance if labels are available
            pass
    
    def detect_data_drift(self, new_data, threshold=0.1):
        """
        Detect data drift using statistical measures.
        
        Args:
            new_data: New dataset to compare against baseline
            threshold: Threshold for drift detection
        
        Returns:
            dict: Drift detection results
        """
        if not self.baseline_stats:
            raise ValueError("Baseline statistics not set. Call set_baseline() first.")
        
        drift_detected = {}
        drift_scores = {}
        
        for column in new_data.columns:
            if column in self.baseline_stats['mean']:
                # Calculate drift score using normalized difference in means
                baseline_mean = self.baseline_stats['mean'][column]
                baseline_std = self.baseline_stats['std'][column]
                new_mean = new_data[column].mean()
                
                if baseline_std > 0:
                    drift_score = abs(new_mean - baseline_mean) / baseline_std
                else:
                    drift_score = abs(new_mean - baseline_mean)
                
                drift_scores[column] = drift_score
                drift_detected[column] = drift_score > threshold
        
        overall_drift = any(drift_detected.values())
        
        result = {
            'overall_drift_detected': overall_drift,
            'column_drift_scores': drift_scores,
            'column_drift_detected': drift_detected,
            'threshold': threshold,
            'timestamp': pd.Timestamp.now()
        }
        
        self.drift_results['data_drift'] = result
        return result
    
    def visualize_drift(self, baseline_data, new_data, save_path=None):
        """
        Create visualizations for drift analysis.
        
        Args:
            baseline_data: Baseline dataset
            new_data: New dataset
            save_path: Path to save the visualization
        """
        numerical_cols = baseline_data.select_dtypes(include=[np.number]).columns
        n_cols = len(numerical_cols)
        
        if n_cols == 0:
            return
        
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols))
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(numerical_cols):
            axes[i].hist(baseline_data[col].dropna(), bins=30, alpha=0.7, 
                        label='Baseline', color='blue', density=True)
            axes[i].hist(new_data[col].dropna(), bins=30, alpha=0.7, 
                        label='New Data', color='red', density=True)
            axes[i].set_title(f'Distribution Comparison: {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class AIQAFramework:
    """
    Main AI Quality Assurance Framework that integrates all modules.
    """
    
    def __init__(self):
        self.data_auditor = DataQualityAuditor()
        self.model_tester = ModelAgnosticTester()
        self.fairness_assessor = FairnessAssessor()
        self.drift_detector = DriftDetector()
        self.results = {}
    
    def comprehensive_audit(self, data, model, target_column, protected_attributes=None):
        """
        Perform a comprehensive audit of the AI system.
        
        Args:
            data: Dataset to audit
            model: Trained model to test
            target_column: Name of the target variable
            protected_attributes: List of protected attribute columns
        
        Returns:
            dict: Comprehensive audit results
        """
        print("Starting comprehensive AI audit...")
        
        # Data quality audit
        print("1. Performing data quality audit...")
        data_profile = self.data_auditor.profile_data(data, target_column)
        
        if protected_attributes:
            bias_analysis = self.data_auditor.detect_bias(data, protected_attributes, target_column)
        else:
            bias_analysis = {}
        
        # Prepare data for model testing (use only numerical features)
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numerical_columns:
            numerical_columns.remove(target_column)
        X = data[numerical_columns]
        y = data[target_column]
        
        # Model testing
        print("2. Performing model testing...")
        X_test = X.sample(min(1000, len(X)))  # Sample for testing
        y_test = y.loc[X_test.index]
        
        # Metamorphic testing example: adding small noise should not drastically change predictions
        def add_noise(X):
            return X + np.random.normal(0, 0.01, X.shape)
        
        def consistency_relation(pred1, pred2):
            return pred1 == pred2  # Strict consistency for classification
        
        metamorphic_results = self.model_tester.metamorphic_test(
            model, X_test, add_noise, consistency_relation, "noise_consistency"
        )
        
        # Adversarial testing
        adversarial_results = self.model_tester.adversarial_test(model, X_test, y_test)
        
        # Fairness assessment
        print("3. Performing fairness assessment...")
        if protected_attributes and len(protected_attributes) > 0:
            y_pred = model.predict(X_test)
            protected_attr = data.loc[X_test.index, protected_attributes[0]]
            fairness_metrics = self.fairness_assessor.calculate_fairness_metrics(
                y_test, y_pred, protected_attr
            )
        else:
            fairness_metrics = {}
        
        # Compile results
        self.results = {
            'data_profile': data_profile,
            'bias_analysis': bias_analysis,
            'metamorphic_test': metamorphic_results,
            'adversarial_test': adversarial_results,
            'fairness_metrics': fairness_metrics,
            'audit_timestamp': pd.Timestamp.now()
        }
        
        print("Comprehensive audit completed!")
        return self.results
    
    def generate_report(self):
        """
        Generate a summary report of the audit results.
        
        Returns:
            str: Formatted report
        """
        if not self.results:
            return "No audit results available. Run comprehensive_audit() first."
        
        report = "AI QUALITY ASSURANCE AUDIT REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Data Quality Summary
        if 'data_profile' in self.results:
            profile = self.results['data_profile']
            report += f"DATA QUALITY SUMMARY:\n"
            report += f"Dataset Shape: {profile['shape']}\n"
            report += f"Missing Values: {sum(profile['missing_values'].values())} total\n"
            if 'target_analysis' in profile:
                balance_ratio = profile['target_analysis']['balance_ratio']
                report += f"Target Balance Ratio: {balance_ratio:.3f}\n"
            report += "\n"
        
        # Bias Analysis Summary
        if 'bias_analysis' in self.results and self.results['bias_analysis']:
            report += "BIAS ANALYSIS SUMMARY:\n"
            for attr, analysis in self.results['bias_analysis'].items():
                report += f"{attr}: Disparate Impact Ratio = {analysis['disparate_impact_ratio']:.3f}\n"
                report += f"  Bias Detected: {'Yes' if analysis['bias_detected'] else 'No'}\n"
            report += "\n"
        
        # Model Testing Summary
        if 'metamorphic_test' in self.results:
            meta_test = self.results['metamorphic_test']
            report += "METAMORPHIC TESTING SUMMARY:\n"
            report += f"Test: {meta_test['test_name']}\n"
            report += f"Violation Rate: {meta_test['violation_rate']:.3f}\n"
            report += "\n"
        
        if 'adversarial_test' in self.results:
            adv_test = self.results['adversarial_test']
            report += "ADVERSARIAL TESTING SUMMARY:\n"
            report += f"Original Accuracy: {adv_test['original_accuracy']:.3f}\n"
            report += f"Adversarial Accuracy: {adv_test['adversarial_accuracy']:.3f}\n"
            report += f"Accuracy Drop: {adv_test['accuracy_drop']:.3f}\n"
            report += "\n"
        
        # Fairness Summary
        if 'fairness_metrics' in self.results and self.results['fairness_metrics']:
            fairness = self.results['fairness_metrics']
            report += "FAIRNESS ASSESSMENT SUMMARY:\n"
            satisfied = fairness['fairness_satisfied']
            report += f"Demographic Parity: {'PASS' if satisfied['demographic_parity'] else 'FAIL'}\n"
            report += f"Equalized Odds: {'PASS' if satisfied['equalized_odds'] else 'FAIL'}\n"
            report += f"Equal Opportunity: {'PASS' if satisfied['equal_opportunity'] else 'FAIL'}\n"
            report += "\n"
        
        report += f"Report Generated: {pd.Timestamp.now()}\n"
        
        return report


if __name__ == "__main__":
    # Example usage and demonstration
    print("AI Quality Assurance Framework - Demo")
    print("=" * 50)
    
    # This would be replaced with actual data loading in a real scenario
    print("Framework modules initialized successfully!")
    print("Ready for comprehensive AI system auditing.")

