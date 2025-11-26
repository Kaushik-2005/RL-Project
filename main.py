"""
Main Execution Script for RL Bias Mitigation Project
Run this script to train and evaluate the DQN agent
"""

import os
import sys
from train import train_dqn_agent, evaluate_agent
from visualization import (
    analyze_baseline_data, 
    plot_training_metrics, 
    plot_fairness_comparison,
    create_summary_report,
    plot_data_distribution
)


def main():
    """
    Main execution function for the RL Bias Mitigation project
    
    This implements the complete pipeline:
    1. Analyze baseline biased data
    2. Train DQN agent with fairness regularization
    3. Evaluate trained agent
    4. Generate visualizations and reports
    """
    
    print("\n" + "=" * 80)
    print(" " * 15 + "RL BIAS MITIGATION PROJECT - PHASE 1")
    print(" " * 20 + "Group 3: Aniketh, Jatin, Kaushik")
    print("=" * 80 + "\n")
    
    # Configuration
    DATA_PATH = 'biased_gender_loans.csv'
    MODEL_SAVE_PATH = 'dqn_loan_model.pt'
    
    # Training hyperparameters
    NUM_EPISODES = 1000
    MAX_STEPS = 30000
    EPISODE_LENGTH = 100
    LAMBDA_FAIRNESS = 0.5
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 10
    USE_CUDA = True  # Use GPU for training
    
    # Verify data file exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file '{DATA_PATH}' not found!")
        print("Please ensure 'biased_gender_loans.csv' is in the current directory.")
        sys.exit(1)
    
    # Step 1: Analyze baseline data
    print("STEP 1: Analyzing baseline (biased) data...")
    baseline_metrics = analyze_baseline_data(DATA_PATH)
    
    # Step 2: Visualize data distribution
    print("\nSTEP 2: Visualizing data distribution...")
    plot_data_distribution(DATA_PATH, save_path='data_distribution.png')
    
    # Step 3: Train DQN agent
    print("\nSTEP 3: Training DQN agent with fairness regularization...")
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Max Steps: {MAX_STEPS}")
    print(f"  Episode Length: {EPISODE_LENGTH}")
    print(f"  Lambda (Fairness): {LAMBDA_FAIRNESS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Gamma (Discount): {GAMMA}")
    print(f"  Epsilon: {EPSILON_START} → {EPSILON_END} (decay: {EPSILON_DECAY})")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Target Update Freq: {TARGET_UPDATE_FREQ}")
    print(f"  Use CUDA: {USE_CUDA}")
    print()
    
    agent, metrics, env = train_dqn_agent(
        data_path=DATA_PATH,
        num_episodes=NUM_EPISODES,
        max_steps=MAX_STEPS,
        episode_length=EPISODE_LENGTH,
        lambda_fairness=LAMBDA_FAIRNESS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        save_model_path=MODEL_SAVE_PATH,
        print_freq=50,
        use_cuda=USE_CUDA
    )
    
    # Step 4: Evaluate agent
    print("\nSTEP 4: Evaluating trained agent...")
    eval_metrics = evaluate_agent(agent, env, num_episodes=100, episode_length=EPISODE_LENGTH)
    
    # Step 5: Generate visualizations
    print("\nSTEP 5: Generating visualizations...")
    plot_training_metrics(metrics, save_path='training_metrics.png')
    plot_fairness_comparison(baseline_metrics, eval_metrics, save_path='fairness_comparison.png')
    
    # Step 6: Create summary report
    print("\nSTEP 6: Creating summary report...")
    create_summary_report(baseline_metrics, eval_metrics, save_path='summary_report.txt')
    
    # Final summary
    print("\n" + "=" * 80)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated Files:")
    print(f"  1. {MODEL_SAVE_PATH} - Trained DQN model")
    print(f"  2. training_metrics.png - Training progress visualization")
    print(f"  3. fairness_comparison.png - Baseline vs RL agent comparison")
    print(f"  4. data_distribution.png - Dataset distribution analysis")
    print(f"  5. summary_report.txt - Detailed summary report")
    
    print("\n" + "=" * 80)
    print("KEY RESULTS:")
    print("=" * 80)
    print(f"Baseline Approval Rates:")
    print(f"  Men:   {baseline_metrics['approval_rate_men']:.2%}")
    print(f"  Women: {baseline_metrics['approval_rate_women']:.2%}")
    print(f"  Bias Gap (SPD): {baseline_metrics['statistical_parity_difference']:.4f}")
    
    print(f"\nRL Agent Approval Rates:")
    print(f"  Men:   {eval_metrics['approval_rate_men']:.2%}")
    print(f"  Women: {eval_metrics['approval_rate_women']:.2%}")
    print(f"  Bias Gap (SPD): {eval_metrics['statistical_parity_difference']:.4f}")
    
    spd_reduction = abs(baseline_metrics['statistical_parity_difference']) - abs(eval_metrics['statistical_parity_difference'])
    spd_reduction_pct = (spd_reduction / abs(baseline_metrics['statistical_parity_difference'])) * 100
    
    print(f"\nImprovement:")
    print(f"  SPD Reduction: {spd_reduction:.4f} ({spd_reduction_pct:.1f}%)")
    
    if spd_reduction > 0:
        print(f"\n✓ SUCCESS: The RL agent successfully reduced gender bias!")
    else:
        print(f"\n⚠ Note: Further tuning may be needed to improve fairness.")
    
    print("\n" + "=" * 80)
    print("Thank you for using the RL Bias Mitigation System!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
