"""
Training Script for DQN Agent on Loan Approval Task
Implements training loop with fairness metrics tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from loan_env import LoanApprovalEnv
from dqn_agent import DQNAgent


class TrainingMetrics:
    """Track and manage training metrics"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_lengths = []
        self.fairness_metrics_history = []
        self.epsilon_history = []
        
    def add_episode(self, total_reward, avg_loss, episode_length, fairness_metrics, epsilon):
        """Record metrics for an episode"""
        self.episode_rewards.append(total_reward)
        self.episode_losses.append(avg_loss)
        self.episode_lengths.append(episode_length)
        self.fairness_metrics_history.append(fairness_metrics)
        self.epsilon_history.append(epsilon)
    
    def get_summary(self, last_n: int = 100) -> Dict:
        """Get summary statistics for last N episodes"""
        if len(self.episode_rewards) == 0:
            return {}
        
        n = min(last_n, len(self.episode_rewards))
        return {
            'avg_reward': np.mean(self.episode_rewards[-n:]),
            'avg_loss': np.mean([l for l in self.episode_losses[-n:] if l is not None]),
            'avg_spd': np.mean([m['statistical_parity_difference'] for m in self.fairness_metrics_history[-n:]]),
            'avg_approval_rate_men': np.mean([m['approval_rate_men'] for m in self.fairness_metrics_history[-n:]]),
            'avg_approval_rate_women': np.mean([m['approval_rate_women'] for m in self.fairness_metrics_history[-n:]])
        }


def train_dqn_agent(
    data_path: str,
    num_episodes: int = 1000,
    max_steps: int = 30000,
    episode_length: int = 100,
    lambda_fairness: float = 0.5,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    target_update_freq: int = 10,
    save_model_path: str = 'dqn_loan_model.pt',
    print_freq: int = 50,
    use_cuda: bool = True
) -> tuple:
    """
    Train DQN agent on loan approval task
    
    Args:
        data_path: Path to CSV dataset
        num_episodes: Number of training episodes
        max_steps: Maximum total training steps
        episode_length: Steps per episode
        lambda_fairness: Fairness penalty weight
        learning_rate: Learning rate for DQN
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Epsilon decay rate
        batch_size: Mini-batch size
        target_update_freq: Target network update frequency
        save_model_path: Path to save trained model
        print_freq: Print frequency (episodes)
        use_cuda: Whether to use CUDA GPU for training
        
    Returns:
        agent: Trained DQN agent
        metrics: Training metrics object
    """
    
    # Initialize environment
    env = LoanApprovalEnv(
        data_path=data_path,
        episode_length=episode_length,
        lambda_fairness=lambda_fairness
    )
    
    # Set device
    device = 'cuda' if use_cuda else 'cpu'
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=3,
        action_dim=2,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        device=device
    )
    
    # Initialize metrics tracker
    metrics = TrainingMetrics()
    
    # Training loop
    total_steps = 0
    print(f"Starting training for {num_episodes} episodes (max {max_steps} steps)...")
    print(f"Episode length: {episode_length}, Lambda fairness: {lambda_fairness}")
    print("-" * 80)
    
    for episode in range(num_episodes):
        if total_steps >= max_steps:
            print(f"\nReached maximum steps ({max_steps}). Stopping training.")
            break
        
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        
        # Episode loop
        step = 0
        for step in range(episode_length):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Get fairness metrics
        fairness_metrics = env.get_fairness_metrics()
        
        # Record metrics
        avg_loss = np.mean(episode_losses) if episode_losses else None
        metrics.add_episode(
            episode_reward,
            avg_loss,
            step + 1,
            fairness_metrics,
            agent.epsilon
        )
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            summary = metrics.get_summary(last_n=print_freq)
            print(f"Episode {episode + 1}/{num_episodes} | "
                    f"Steps: {total_steps} | "
                    f"Avg Reward: {summary['avg_reward']:.2f} | "
                    f"Avg Loss: {summary['avg_loss']:.4f} | "
                    f"SPD: {summary['avg_spd']:.4f} | "
                    f"Approval (M/W): {summary['avg_approval_rate_men']:.2%}/{summary['avg_approval_rate_women']:.2%} | "
                    f"ε: {agent.epsilon:.3f}"
                )
    
    # Save model
    agent.save_model(save_model_path)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Final statistics
    final_summary = metrics.get_summary(last_n=100)
    final_fairness = env.get_fairness_metrics()
    
    print("\nFinal Performance (last 100 episodes):")
    print(f"  Average Reward: {final_summary['avg_reward']:.2f}")
    print(f"  Average Loss: {final_summary['avg_loss']:.4f}")
    print(f"  Statistical Parity Difference: {final_summary['avg_spd']:.4f}")
    print(f"  Approval Rate (Men): {final_summary['avg_approval_rate_men']:.2%}")
    print(f"  Approval Rate (Women): {final_summary['avg_approval_rate_women']:.2%}")
    
    print("\nOverall Statistics:")
    print(f"  Total Men: {final_fairness['total_men']}")
    print(f"  Total Women: {final_fairness['total_women']}")
    print(f"  Approvals (Men): {final_fairness['total_approvals_men']}")
    print(f"  Approvals (Women): {final_fairness['total_approvals_women']}")
    print(f"  Disparity Ratio: {final_fairness['disparity_ratio']:.3f}")
    
    return agent, metrics, env


def evaluate_agent(
    agent: DQNAgent,
    env: LoanApprovalEnv,
    num_episodes: int = 100,
    episode_length: int = 100
) -> Dict:
    """
    Evaluate trained agent
    
    Args:
        agent: Trained DQN agent
        env: Loan approval environment
        num_episodes: Number of evaluation episodes
        episode_length: Steps per episode
        
    Returns:
        eval_metrics: Evaluation metrics dictionary
    """
    print(f"\nEvaluating agent over {num_episodes} episodes...")
    
    # Reset environment statistics
    env.total_approvals = {'Man': 0, 'Woman': 0}
    env.total_counts = {'Man': 0, 'Woman': 0}
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        
        for step in range(episode_length):
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
    
    # Get final fairness metrics
    fairness_metrics = env.get_fairness_metrics()
    
    eval_metrics = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'approval_rate_men': fairness_metrics['approval_rate_men'],
        'approval_rate_women': fairness_metrics['approval_rate_women'],
        'statistical_parity_difference': fairness_metrics['statistical_parity_difference'],
        'disparity_ratio': fairness_metrics['disparity_ratio'],
        'total_approvals_men': fairness_metrics['total_approvals_men'],
        'total_approvals_women': fairness_metrics['total_approvals_women'],
        'total_men': fairness_metrics['total_men'],
        'total_women': fairness_metrics['total_women']
    }
    
    print("\nEvaluation Results:")
    print(f"  Average Reward: {eval_metrics['avg_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    print(f"  Approval Rate (Men): {eval_metrics['approval_rate_men']:.2%}")
    print(f"  Approval Rate (Women): {eval_metrics['approval_rate_women']:.2%}")
    print(f"  Statistical Parity Difference: {eval_metrics['statistical_parity_difference']:.4f}")
    print(f"  Disparity Ratio: {eval_metrics['disparity_ratio']:.3f}")
    
    return eval_metrics
