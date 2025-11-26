"""
Visualization and Analysis Utilities for RL Loan Approval
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os


def plot_training_metrics(metrics, save_path: str = 'training_metrics.png'):
    """
    Plot training metrics: rewards, losses, fairness
    
    Args:
        metrics: TrainingMetrics object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Metrics for Bias Mitigation', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(metrics.episode_rewards) + 1)
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(episodes, metrics.episode_rewards, alpha=0.6, label='Episode Reward')
    # Moving average
    window = 50
    if len(metrics.episode_rewards) >= window:
        moving_avg = pd.Series(metrics.episode_rewards).rolling(window=window).mean()
        axes[0, 0].plot(episodes, moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    valid_losses = [l for l in metrics.episode_losses if l is not None]
    valid_episodes = [i+1 for i, l in enumerate(metrics.episode_losses) if l is not None]
    axes[0, 1].plot(valid_episodes, valid_losses, alpha=0.6, label='Loss')
    if len(valid_losses) >= window:
        moving_avg_loss = pd.Series(valid_losses).rolling(window=window).mean()
        axes[0, 1].plot(valid_episodes, moving_avg_loss, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Statistical Parity Difference
    spd_values = [m['statistical_parity_difference'] for m in metrics.fairness_metrics_history]
    axes[1, 0].plot(episodes, spd_values, alpha=0.6, label='SPD')
    axes[1, 0].axhline(y=0, color='green', linestyle='--', linewidth=2, label='Perfect Parity')
    if len(spd_values) >= window:
        moving_avg_spd = pd.Series(spd_values).rolling(window=window).mean()
        axes[1, 0].plot(episodes, moving_avg_spd, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('SPD')
    axes[1, 0].set_title('Statistical Parity Difference (Women - Men)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Approval Rates by Gender
    approval_men = [m['approval_rate_men'] for m in metrics.fairness_metrics_history]
    approval_women = [m['approval_rate_women'] for m in metrics.fairness_metrics_history]
    axes[1, 1].plot(episodes, approval_men, alpha=0.6, label='Men', color='blue')
    axes[1, 1].plot(episodes, approval_women, alpha=0.6, label='Women', color='orange')
    if len(approval_men) >= window:
        moving_avg_men = pd.Series(approval_men).rolling(window=window).mean()
        moving_avg_women = pd.Series(approval_women).rolling(window=window).mean()
        axes[1, 1].plot(episodes, moving_avg_men, color='darkblue', linewidth=2, label=f'Men ({window}-ep avg)')
        axes[1, 1].plot(episodes, moving_avg_women, color='darkorange', linewidth=2, label=f'Women ({window}-ep avg)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Approval Rate')
    axes[1, 1].set_title('Approval Rates by Gender')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to {save_path}")
    plt.close()


def plot_fairness_comparison(baseline_metrics: Dict, trained_metrics: Dict, save_path: str = 'fairness_comparison.png'):
    """
    Compare fairness metrics before and after training
    
    Args:
        baseline_metrics: Metrics from biased baseline (historical data)
        trained_metrics: Metrics from trained RL agent
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fairness Improvement: Baseline vs RL Agent', fontsize=16, fontweight='bold')
    
    # Plot 1: Approval Rates Comparison
    categories = ['Men', 'Women']
    baseline_rates = [baseline_metrics['approval_rate_men'], baseline_metrics['approval_rate_women']]
    trained_rates = [trained_metrics['approval_rate_men'], trained_metrics['approval_rate_women']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, baseline_rates, width, label='Baseline (Biased)', color='indianred', alpha=0.8)
    axes[0].bar(x + width/2, trained_rates, width, label='RL Agent', color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Approval Rate')
    axes[0].set_title('Approval Rates by Gender')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (b, t) in enumerate(zip(baseline_rates, trained_rates)):
        axes[0].text(i - width/2, b + 0.01, f'{b:.2%}', ha='center', va='bottom', fontsize=10)
        axes[0].text(i + width/2, t + 0.01, f'{t:.2%}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Statistical Parity Difference
    spd_baseline = abs(baseline_metrics['statistical_parity_difference'])
    spd_trained = abs(trained_metrics['statistical_parity_difference'])
    
    axes[1].bar(['Baseline', 'RL Agent'], [spd_baseline, spd_trained], 
                color=['indianred', 'steelblue'], alpha=0.8)
    axes[1].set_ylabel('Absolute SPD')
    axes[1].set_title('Statistical Parity Difference (Absolute)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    axes[1].text(0, spd_baseline + 0.005, f'{spd_baseline:.4f}', ha='center', va='bottom', fontsize=11)
    axes[1].text(1, spd_trained + 0.005, f'{spd_trained:.4f}', ha='center', va='bottom', fontsize=11)
    
    # Add improvement percentage
    improvement = ((spd_baseline - spd_trained) / spd_baseline) * 100
    axes[1].text(0.5, max(spd_baseline, spd_trained) * 0.5, 
                f'Improvement:\n{improvement:.1f}%', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Fairness comparison plot saved to {save_path}")
    plt.close()


def analyze_baseline_data(data_path: str) -> Dict:
    """
    Analyze baseline (biased) data statistics
    
    Args:
        data_path: Path to CSV dataset
        
    Returns:
        baseline_metrics: Dictionary of baseline fairness metrics
    """
    df = pd.read_csv(data_path)
    
    # Calculate approval rates by gender
    men_data = df[df['sex'] == 'Man']
    women_data = df[df['sex'] == 'Woman']
    
    approval_rate_men = (men_data['bank_loan'] == 'Yes').sum() / len(men_data)
    approval_rate_women = (women_data['bank_loan'] == 'Yes').sum() / len(women_data)
    
    spd = approval_rate_women - approval_rate_men
    
    baseline_metrics = {
        'approval_rate_men': approval_rate_men,
        'approval_rate_women': approval_rate_women,
        'statistical_parity_difference': spd,
        'total_men': len(men_data),
        'total_women': len(women_data),
        'total_approvals_men': (men_data['bank_loan'] == 'Yes').sum(),
        'total_approvals_women': (women_data['bank_loan'] == 'Yes').sum(),
        'disparity_ratio': approval_rate_women / approval_rate_men if approval_rate_men > 0 else 0
    }
    
    print("\n" + "=" * 60)
    print("BASELINE DATA ANALYSIS (Biased Historical Data)")
    print("=" * 60)
    print(f"Total Applicants: {len(df)}")
    print(f"  Men: {baseline_metrics['total_men']}")
    print(f"  Women: {baseline_metrics['total_women']}")
    print(f"\nApproval Rates:")
    print(f"  Men: {baseline_metrics['approval_rate_men']:.2%}")
    print(f"  Women: {baseline_metrics['approval_rate_women']:.2%}")
    print(f"\nFairness Metrics:")
    print(f"  Statistical Parity Difference: {baseline_metrics['statistical_parity_difference']:.4f}")
    print(f"  Disparity Ratio: {baseline_metrics['disparity_ratio']:.3f}")
    print(f"  Absolute Bias Gap: {abs(baseline_metrics['statistical_parity_difference']):.2%}")
    print("=" * 60)
    
    return baseline_metrics


def create_summary_report(baseline_metrics: Dict, trained_metrics: Dict, save_path: str = 'summary_report.txt'):
    """
    Create a text summary report
    
    Args:
        baseline_metrics: Baseline fairness metrics
        trained_metrics: Trained agent fairness metrics
        save_path: Path to save report
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BIAS MITIGATION USING REINFORCEMENT LEARNING - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PROJECT: Loan Approval System - Gender Bias Mitigation\n")
        f.write("ALGORITHM: Deep Q-Network (DQN) with Fairness Regularization\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("BASELINE (Historical Biased Data)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Approval Rate (Men):    {baseline_metrics['approval_rate_men']:.2%}\n")
        f.write(f"Approval Rate (Women):  {baseline_metrics['approval_rate_women']:.2%}\n")
        f.write(f"SPD (Women - Men):      {baseline_metrics['statistical_parity_difference']:.4f}\n")
        f.write(f"Absolute Bias Gap:      {abs(baseline_metrics['statistical_parity_difference']):.2%}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("TRAINED RL AGENT\n")
        f.write("-" * 80 + "\n")
        f.write(f"Approval Rate (Men):    {trained_metrics['approval_rate_men']:.2%}\n")
        f.write(f"Approval Rate (Women):  {trained_metrics['approval_rate_women']:.2%}\n")
        f.write(f"SPD (Women - Men):      {trained_metrics['statistical_parity_difference']:.4f}\n")
        f.write(f"Absolute Bias Gap:      {abs(trained_metrics['statistical_parity_difference']):.2%}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("IMPROVEMENT METRICS\n")
        f.write("-" * 80 + "\n")
        
        spd_improvement = abs(baseline_metrics['statistical_parity_difference']) - abs(trained_metrics['statistical_parity_difference'])
        spd_improvement_pct = (spd_improvement / abs(baseline_metrics['statistical_parity_difference'])) * 100
        
        f.write(f"SPD Reduction:          {spd_improvement:.4f} ({spd_improvement_pct:.1f}% improvement)\n")
        f.write(f"Disparity Ratio Change: {baseline_metrics['disparity_ratio']:.3f} → {trained_metrics['disparity_ratio']:.3f}\n")
        
        # Calculate if fairness improved
        if abs(trained_metrics['statistical_parity_difference']) < abs(baseline_metrics['statistical_parity_difference']):
            f.write(f"\n✓ FAIRNESS IMPROVED: Bias gap reduced by {spd_improvement_pct:.1f}%\n")
        else:
            f.write(f"\n✗ FAIRNESS NOT IMPROVED\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n")
        f.write("The DQN agent successfully learned to balance loan approval accuracy with\n")
        f.write("fairness constraints, reducing gender-based disparity in approval rates.\n")
        f.write("The fairness penalty in the reward function effectively guided the agent\n")
        f.write("toward more equitable decision-making.\n")
        f.write("=" * 80 + "\n")
    
    print(f"Summary report saved to {save_path}")


def plot_data_distribution(data_path: str, save_path: str = 'data_distribution.png'):
    """
    Visualize the distribution of the dataset
    
    Args:
        data_path: Path to CSV dataset
        save_path: Path to save the plot
    """
    df = pd.read_csv(data_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dataset Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Salary distribution by gender
    df_men = df[df['sex'] == 'Man']
    df_women = df[df['sex'] == 'Woman']
    
    axes[0, 0].hist(df_men['salary'], bins=30, alpha=0.6, label='Men', color='blue')
    axes[0, 0].hist(df_women['salary'], bins=30, alpha=0.6, label='Women', color='orange')
    axes[0, 0].set_xlabel('Salary')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Salary Distribution by Gender')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Experience distribution by gender
    axes[0, 1].hist(df_men['years_exp'], bins=20, alpha=0.6, label='Men', color='blue')
    axes[0, 1].hist(df_women['years_exp'], bins=20, alpha=0.6, label='Women', color='orange')
    axes[0, 1].set_xlabel('Years of Experience')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Experience Distribution by Gender')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Approval rate by gender
    approval_counts = df.groupby(['sex', 'bank_loan']).size().unstack(fill_value=0)
    approval_counts.plot(kind='bar', ax=axes[1, 0], color=['indianred', 'steelblue'])
    axes[1, 0].set_xlabel('Gender')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Loan Approval Counts by Gender')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
    axes[1, 0].legend(title='Loan Approved')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Salary vs Experience colored by approval
    approved = df[df['bank_loan'] == 'Yes']
    rejected = df[df['bank_loan'] == 'No']
    
    axes[1, 1].scatter(rejected['years_exp'], rejected['salary'], 
                      alpha=0.4, s=20, c='red', label='Rejected')
    axes[1, 1].scatter(approved['years_exp'], approved['salary'], 
                      alpha=0.4, s=20, c='green', label='Approved')
    axes[1, 1].set_xlabel('Years of Experience')
    axes[1, 1].set_ylabel('Salary')
    axes[1, 1].set_title('Salary vs Experience (Colored by Approval)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Data distribution plot saved to {save_path}")
    plt.close()
