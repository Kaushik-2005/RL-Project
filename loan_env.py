"""
Loan Approval Environment for Bias Mitigation using RL
Implements the MDP as described in the project specification
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import gym
from gym import spaces


class LoanApprovalEnv(gym.Env):
    """
    Custom Environment for Loan Approval with Fairness Constraints
    
    State Space: (salary, years_exp, sex)
    Action Space: {0: Reject, 1: Approve}
    Reward: Classification reward + Fairness penalty
    """
    
    def __init__(self, data_path: str, episode_length: int = 100, lambda_fairness: float = 0.5):
        super(LoanApprovalEnv, self).__init__()
        
        # Load dataset
        self.data = pd.read_csv(data_path)
        self.episode_length = episode_length
        self.lambda_fairness = lambda_fairness
        
        # Preprocess data
        self._preprocess_data()
        
        # Define action and observation space
        # Action: 0 = Reject, 1 = Approve
        self.action_space = spaces.Discrete(2)
        
        # Observation: [salary (normalized), years_exp (normalized), sex (0/1)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.current_state = None
        self.current_label = None
        
        # Fairness tracking (approval rates per episode)
        self.episode_approvals = {'Man': [], 'Woman': []}
        self.episode_total = {'Man': 0, 'Woman': 0}
        
        # Overall statistics for evaluation
        self.total_approvals = {'Man': 0, 'Woman': 0}
        self.total_counts = {'Man': 0, 'Woman': 0}
        
    def _preprocess_data(self):
        """Normalize continuous features and encode categorical variables"""
        # Store original for reference
        self.data_original = self.data.copy()
        
        # Normalize salary and years_exp
        self.salary_min = self.data['salary'].min()
        self.salary_max = self.data['salary'].max()
        self.exp_min = self.data['years_exp'].min()
        self.exp_max = self.data['years_exp'].max()
        
        self.data['salary_norm'] = (self.data['salary'] - self.salary_min) / (self.salary_max - self.salary_min)
        self.data['exp_norm'] = (self.data['years_exp'] - self.exp_min) / (self.exp_max - self.exp_min)
        
        # Encode sex: Man = 1, Woman = 0
        self.data['sex_encoded'] = (self.data['sex'] == 'Man').astype(float)
        
        # Encode label: Yes = 1, No = 0
        self.data['label'] = (self.data['bank_loan'] == 'Yes').astype(int)
        
    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode"""
        self.current_step = 0
        
        # Reset episode fairness tracking
        self.episode_approvals = {'Man': [], 'Woman': []}
        self.episode_total = {'Man': 0, 'Woman': 0}
        
        # Sample first applicant
        return self._get_next_applicant()
    
    def _get_next_applicant(self) -> np.ndarray:
        """Sample a random applicant from the dataset"""
        idx = np.random.randint(0, len(self.data))
        row = self.data.iloc[idx]
        
        # Store current state information
        self.current_state = np.array([
            row['salary_norm'],
            row['exp_norm'],
            row['sex_encoded']
        ], dtype=np.float32)
        
        self.current_label = row['label']
        self.current_sex = row['sex']
        
        # Update episode totals
        self.episode_total[self.current_sex] += 1
        self.total_counts[self.current_sex] += 1
        
        return self.current_state
    
    def _compute_classification_reward(self, action: int) -> float:
        """
        Compute base classification reward
        
        Rules:
        - Approve & label=Yes → +1
        - Approve & label=No → -1
        - Reject → 0
        """
        if action == 1:  # Approve
            return 1.0 if self.current_label == 1 else -1.0
        else:  # Reject
            return 0.0
    
    def _compute_fairness_penalty(self) -> float:
        """
        Compute fairness penalty based on approval rate disparity
        
        Penalty: -λ * |approval_rate(women) - approval_rate(men)|
        """
        # Calculate approval rates for current episode
        approval_rate_man = (
            np.mean(self.episode_approvals['Man']) 
            if len(self.episode_approvals['Man']) > 0 else 0.0
        )
        approval_rate_woman = (
            np.mean(self.episode_approvals['Woman']) 
            if len(self.episode_approvals['Woman']) > 0 else 0.0
        )
        
        gap = abs(approval_rate_woman - approval_rate_man)
        penalty = -self.lambda_fairness * gap
        
        return float(penalty)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: 0 (Reject) or 1 (Approve)
            
        Returns:
            observation: next state
            reward: combined reward
            done: episode termination flag
            info: additional information
        """
        # Record approval decision for fairness tracking
        self.episode_approvals[self.current_sex].append(float(action))
        if action == 1:
            self.total_approvals[self.current_sex] += 1
        
        # Compute rewards
        classification_reward = self._compute_classification_reward(action)
        fairness_penalty = self._compute_fairness_penalty()
        total_reward = classification_reward + fairness_penalty
        
        # Increment step
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        # Get next applicant (if not done)
        if not done:
            next_state = self._get_next_applicant()
        else:
            next_state = self.current_state  # Terminal state
        
        # Info dictionary
        info = {
            'classification_reward': classification_reward,
            'fairness_penalty': fairness_penalty,
            'current_sex': self.current_sex,
            'action': action,
            'label': self.current_label
        }
        
        return next_state, total_reward, done, info
    
    def get_fairness_metrics(self) -> Dict:
        """Calculate fairness metrics across all episodes"""
        approval_rate_man = (
            self.total_approvals['Man'] / self.total_counts['Man']
            if self.total_counts['Man'] > 0 else 0.0
        )
        approval_rate_woman = (
            self.total_approvals['Woman'] / self.total_counts['Woman']
            if self.total_counts['Woman'] > 0 else 0.0
        )
        
        # Statistical Parity Difference
        spd = approval_rate_woman - approval_rate_man
        
        return {
            'approval_rate_men': approval_rate_man,
            'approval_rate_women': approval_rate_woman,
            'statistical_parity_difference': spd,
            'disparity_ratio': approval_rate_woman / approval_rate_man if approval_rate_man > 0 else 0.0,
            'total_approvals_men': self.total_approvals['Man'],
            'total_approvals_women': self.total_approvals['Woman'],
            'total_men': self.total_counts['Man'],
            'total_women': self.total_counts['Woman']
        }
    
    def render(self, mode='human'):
        """Render environment state (optional)"""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.episode_length}")
            print(f"Current applicant: Sex={self.current_sex}, Label={self.current_label}")
            metrics = self.get_fairness_metrics()
            print(f"Overall approval rates - Men: {metrics['approval_rate_men']:.2%}, Women: {metrics['approval_rate_women']:.2%}")
