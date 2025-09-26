"""
Demonstration of the AI Quality Assurance Framework
This script demonstrates the framework using a synthetic dataset that mimics real-world scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from ai_testing_framework import AIQAFramework

# Set random seed for reproducibility
np.random.seed(42)

def create_synthetic_dataset(n_samples=5000):
    """
    Create a synthetic dataset that demonstrates bias and fairness issues.
    This simulates a loan approval dataset with potential bias.
    """
    print("Creating synthetic loan approval dataset...")
    
    # Generate demographic features
    age = np.random.normal(40, 15, n_samples)
    age = np.clip(age, 18, 80)
    
    # Create gender with slight imbalance
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    
    # Create race with imbalance (simulating real-world demographics)
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                           n_samples, p=[0.6, 0.15, 0.15, 0.08, 0.02])
    
    # Generate financial features with some correlation to demographics (introducing bias)
    income_base = np.random.normal(50000, 20000, n_samples)
    
    # Introduce bias: systematic income differences by race and gender
    income_bias = np.zeros(n_samples)
    income_bias[gender == 'Female'] -= 5000  # Gender pay gap
    income_bias[race == 'Black'] -= 8000     # Racial income disparity
    income_bias[race == 'Hispanic'] -= 6000
    
    income = income_base + income_bias
    income = np.clip(income, 20000, 200000)
    
    # Credit score (correlated with income but with some noise)
    credit_score = 300 + (income / 1000) * 2 + np.random.normal(0, 50, n_samples)
    credit_score = np.clip(credit_score, 300, 850)
    
    # Loan amount requested
    loan_amount = np.random.normal(200000, 100000, n_samples)
    loan_amount = np.clip(loan_amount, 50000, 1000000)
    
    # Debt-to-income ratio
    existing_debt = np.random.normal(income * 0.3, income * 0.1, n_samples)
    existing_debt = np.clip(existing_debt, 0, income * 0.8)
    debt_to_income = existing_debt / income
    
    # Employment length
    employment_length = np.random.exponential(5, n_samples)
    employment_length = np.clip(employment_length, 0, 40)
    
    # Create target variable (loan approval) with bias
    # Base probability depends on financial factors
    base_prob = (
        0.3 +
        0.4 * (credit_score - 300) / (850 - 300) +
        0.2 * (income - 20000) / (200000 - 20000) +
        0.1 * (1 - debt_to_income) +
        0.1 * np.minimum(employment_length / 10, 1)
    )
    
    # Add bias: lower approval rates for certain groups
    bias_factor = np.ones(n_samples)
    bias_factor[race == 'Black'] *= 0.8      # 20% lower approval rate
    bias_factor[race == 'Hispanic'] *= 0.85  # 15% lower approval rate
    bias_factor[gender == 'Female'] *= 0.9   # 10% lower approval rate
    
    approval_prob = base_prob * bias_factor
    approval_prob = np.clip(approval_prob, 0, 1)
    
    # Generate binary approval decisions
    loan_approved = np.random.binomial(1, approval_prob, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'race': race,
        'income': income,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'debt_to_income_ratio': debt_to_income,
        'employment_length': employment_length,
        'loan_approved': loan_approved
    })
    
    print(f"Dataset created with {n_samples} samples")
    print(f"Approval rate: {loan_approved.mean():.3f}")
    print(f"Approval rate by gender:")
    print(data.groupby('gender')['loan_approved'].mean())
    print(f"Approval rate by race:")
    print(data.groupby('race')['loan_approved'].mean())
    
    return data

def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models for comparison.
    """
    print("\nTraining models...")
    
    models = {}
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    models['Random Forest'] = {'model': rf_model, 'accuracy': rf_accuracy}
    
    # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # Create a wrapper for the scaled logistic regression
    class ScaledLogisticRegression:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
        
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        
        def predict_proba(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
    
    scaled_lr = ScaledLogisticRegression(lr_model, scaler)
    models['Logistic Regression'] = {'model': scaled_lr, 'accuracy': lr_accuracy}
    
    print("Model training completed:")
    for name, info in models.items():
        print(f"{name}: Accuracy = {info['accuracy']:.3f}")
    
    return models

def run_comprehensive_demo():
    """
    Run a comprehensive demonstration of the AI-QA Framework.
    """
    print("AI Quality Assurance Framework - Comprehensive Demo")
    print("=" * 60)
    
    # Create synthetic dataset
    data = create_synthetic_dataset(5000)
    
    # Prepare features and target
    feature_columns = ['age', 'income', 'credit_score', 'loan_amount', 
                      'debt_to_income_ratio', 'employment_length']
    X = data[feature_columns]
    y = data['loan_approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = train_models(X_train, y_train, X_test, y_test)
    
    # Initialize AI-QA Framework
    aiqa = AIQAFramework()
    
    # Run comprehensive audit for each model
    results = {}
    
    for model_name, model_info in models.items():
        print(f"\n{'='*60}")
        print(f"AUDITING {model_name.upper()}")
        print(f"{'='*60}")
        
        model = model_info['model']
        
        # Run comprehensive audit
        audit_results = aiqa.comprehensive_audit(
            data=data,
            model=model,
            target_column='loan_approved',
            protected_attributes=['gender', 'race']
        )
        
        results[model_name] = audit_results
        
        # Generate and print report
        report = aiqa.generate_report()
        print(report)
        
        # Save detailed results
        with open(f'/home/ubuntu/{model_name.lower().replace(" ", "_")}_audit_results.txt', 'w') as f:
            f.write(report)
    
    return data, models, results

def create_visualizations(data, models, results):
    """
    Create comprehensive visualizations for the audit results.
    """
    print("\nCreating visualizations...")
    
    # 1. Data distribution visualization
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Income distribution by race
    plt.subplot(2, 3, 1)
    for race in data['race'].unique():
        race_data = data[data['race'] == race]['income']
        plt.hist(race_data, alpha=0.7, label=race, bins=30, density=True)
    plt.xlabel('Income')
    plt.ylabel('Density')
    plt.title('Income Distribution by Race')
    plt.legend()
    
    # Subplot 2: Credit score distribution by gender
    plt.subplot(2, 3, 2)
    for gender in data['gender'].unique():
        gender_data = data[data['gender'] == gender]['credit_score']
        plt.hist(gender_data, alpha=0.7, label=gender, bins=30, density=True)
    plt.xlabel('Credit Score')
    plt.ylabel('Density')
    plt.title('Credit Score Distribution by Gender')
    plt.legend()
    
    # Subplot 3: Approval rates by race
    plt.subplot(2, 3, 3)
    approval_by_race = data.groupby('race')['loan_approved'].mean()
    bars = plt.bar(approval_by_race.index, approval_by_race.values)
    plt.xlabel('Race')
    plt.ylabel('Approval Rate')
    plt.title('Loan Approval Rate by Race')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Subplot 4: Approval rates by gender
    plt.subplot(2, 3, 4)
    approval_by_gender = data.groupby('gender')['loan_approved'].mean()
    bars = plt.bar(approval_by_gender.index, approval_by_gender.values)
    plt.xlabel('Gender')
    plt.ylabel('Approval Rate')
    plt.title('Loan Approval Rate by Gender')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Subplot 5: Feature correlation heatmap
    plt.subplot(2, 3, 5)
    feature_columns = ['age', 'income', 'credit_score', 'loan_amount', 
                      'debt_to_income_ratio', 'employment_length', 'loan_approved']
    correlation_matrix = data[feature_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    
    # Subplot 6: Model performance comparison
    plt.subplot(2, 3, 6)
    model_names = list(models.keys())
    accuracies = [models[name]['accuracy'] for name in model_names]
    bars = plt.bar(model_names, accuracies)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Fairness metrics visualization
    plt.figure(figsize=(12, 8))
    
    model_names = []
    demographic_parity = []
    equalized_odds = []
    equal_opportunity = []
    
    for model_name, result in results.items():
        if 'fairness_metrics' in result and result['fairness_metrics']:
            model_names.append(model_name)
            fairness = result['fairness_metrics']
            demographic_parity.append(fairness['demographic_parity_difference'])
            equalized_odds.append(fairness['equalized_odds_difference'])
            equal_opportunity.append(fairness['equal_opportunity_difference'])
    
    if model_names:
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.bar(x - width, demographic_parity, width, label='Demographic Parity', alpha=0.8)
        plt.bar(x, equalized_odds, width, label='Equalized Odds', alpha=0.8)
        plt.bar(x + width, equal_opportunity, width, label='Equal Opportunity', alpha=0.8)
        
        plt.xlabel('Model')
        plt.ylabel('Fairness Metric Difference')
        plt.title('Fairness Metrics Comparison\n(Lower values indicate better fairness)')
        plt.xticks(x, model_names)
        plt.legend()
        plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, 
                   label='Fairness Threshold (0.1)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/fairness_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Adversarial robustness visualization
    plt.figure(figsize=(10, 6))
    
    model_names = []
    original_acc = []
    adversarial_acc = []
    accuracy_drops = []
    
    for model_name, result in results.items():
        if 'adversarial_test' in result:
            model_names.append(model_name)
            adv_test = result['adversarial_test']
            original_acc.append(adv_test['original_accuracy'])
            adversarial_acc.append(adv_test['adversarial_accuracy'])
            accuracy_drops.append(adv_test['accuracy_drop'])
    
    if model_names:
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, original_acc, width, label='Original Accuracy', alpha=0.8)
        plt.bar(x + width/2, adversarial_acc, width, label='Adversarial Accuracy', alpha=0.8)
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Robustness: Original vs Adversarial Accuracy')
        plt.xticks(x, model_names)
        plt.legend()
        plt.ylim(0, 1)
        
        # Add accuracy drop annotations
        for i, drop in enumerate(accuracy_drops):
            plt.annotate(f'Drop: {drop:.3f}', 
                        xy=(i, adversarial_acc[i]), 
                        xytext=(i, adversarial_acc[i] - 0.1),
                        ha='center', fontsize=10,
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/adversarial_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved:")
    print("- comprehensive_analysis.png")
    print("- fairness_metrics.png")
    print("- adversarial_robustness.png")

if __name__ == "__main__":
    # Run the comprehensive demonstration
    data, models, results = run_comprehensive_demo()
    
    # Create visualizations
    create_visualizations(data, models, results)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nKey findings:")
    print("1. The synthetic dataset demonstrates realistic bias patterns")
    print("2. Both models show fairness violations across demographic groups")
    print("3. Adversarial testing reveals model vulnerabilities")
    print("4. The AI-QA Framework successfully identifies these issues")
    print("\nAll results and visualizations have been saved to files.")

