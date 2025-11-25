"""
Generate training visualization charts for README
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

# Create assets directory
os.makedirs('assets', exist_ok=True)

# Load SFT metrics
sft_metrics = []
with open('training/logs/run_20251124_200256/sft_metrics.jsonl', 'r') as f:
    for line in f:
        sft_metrics.append(json.loads(line))

# Load RL metrics
rl_metrics = []
with open('training/logs/run_20251124_200256/rl_metrics.jsonl', 'r') as f:
    for line in f:
        rl_metrics.append(json.loads(line))

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
colors = {
    'train': '#2563eb',
    'test': '#dc2626',
    'reward': '#059669',
    'f1': '#7c3aed',
    'our_model': '#2563eb',
    'cohere': '#dc2626'
}

# ============ Chart 1: SFT Loss Curve ============
fig, ax = plt.subplots(figsize=(10, 5))

steps = [m['step'] for m in sft_metrics]
train_loss = [m['train_loss'] for m in sft_metrics]

# Get test loss points
test_steps = [m['step'] for m in sft_metrics if 'test_loss' in m]
test_loss = [m['test_loss'] for m in sft_metrics if 'test_loss' in m]

ax.plot(steps, train_loss, color=colors['train'], linewidth=2, label='Train Loss', alpha=0.8)
ax.scatter(test_steps, test_loss, color=colors['test'], s=80, zorder=5, label='Test Loss', marker='o')
ax.plot(test_steps, test_loss, color=colors['test'], linewidth=2, linestyle='--', alpha=0.5)

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('SFT Training: Loss Convergence', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 6)

# Add annotations
ax.annotate(f'Final: {train_loss[-1]:.3f}', xy=(steps[-1], train_loss[-1]), 
            xytext=(steps[-1]-15, train_loss[-1]+0.5),
            fontsize=9, color=colors['train'])
ax.annotate(f'Best Test: {min(test_loss):.3f}', xy=(test_steps[test_loss.index(min(test_loss))], min(test_loss)), 
            xytext=(test_steps[test_loss.index(min(test_loss))]+5, min(test_loss)+0.3),
            fontsize=9, color=colors['test'])

plt.tight_layout()
plt.savefig('assets/sft_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: assets/sft_loss.png")

# ============ Chart 2: RL Reward Progression ============
fig, ax = plt.subplots(figsize=(10, 5))

iterations = [m['iteration'] for m in rl_metrics]
mean_reward = [m['mean_reward'] for m in rl_metrics]
std_reward = [m['std_reward'] for m in rl_metrics]

# Plot with confidence band
ax.fill_between(iterations, 
                [r - s for r, s in zip(mean_reward, std_reward)],
                [r + s for r, s in zip(mean_reward, std_reward)],
                alpha=0.2, color=colors['reward'])
ax.plot(iterations, mean_reward, color=colors['reward'], linewidth=2.5, label='Mean Reward')

ax.set_xlabel('RL Iteration', fontsize=12)
ax.set_ylabel('Reward', fontsize=12)
ax.set_title('RL Training: Reward Progression', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.5, 1.0)

# Add annotations
ax.annotate(f'Start: {mean_reward[0]:.3f}', xy=(0, mean_reward[0]), 
            xytext=(2, mean_reward[0]-0.05), fontsize=9, color=colors['reward'])
ax.annotate(f'Peak: {max(mean_reward):.3f}', xy=(mean_reward.index(max(mean_reward)), max(mean_reward)), 
            xytext=(mean_reward.index(max(mean_reward))+2, max(mean_reward)+0.02),
            fontsize=9, color=colors['reward'])

plt.tight_layout()
plt.savefig('assets/rl_reward.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: assets/rl_reward.png")

# ============ Chart 3: Reward Components ============
fig, ax = plt.subplots(figsize=(10, 5))

r_f1 = [m['mean_r_f1'] for m in rl_metrics]
r_temp = [m['mean_r_temp'] for m in rl_metrics]
r_parity = [m['mean_r_parity'] for m in rl_metrics]
r_eff = [m['mean_r_eff'] for m in rl_metrics]

ax.plot(iterations, r_f1, label='R_F1 (60%)', linewidth=2, color='#2563eb')
ax.plot(iterations, r_temp, label='R_temp (20%)', linewidth=2, color='#7c3aed')
ax.plot(iterations, r_parity, label='R_parity (10%)', linewidth=2, color='#059669')
ax.plot(iterations, r_eff, label='R_eff (10%)', linewidth=2, color='#f59e0b')

ax.set_xlabel('RL Iteration', fontsize=12)
ax.set_ylabel('Reward Component', fontsize=12)
ax.set_title('RL Training: Reward Components', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig('assets/rl_components.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: assets/rl_components.png")

# ============ Chart 4: Model Comparison ============
fig, ax = plt.subplots(figsize=(8, 5))

metrics = ['Avg F1', 'Exact Match', 'Any Match']
our_model = [0.68, 0.60, 0.72]
cohere = [0.61, 0.26, 0.82]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, our_model, width, label='Ours (8B)', color=colors['our_model'])
bars2 = ax.bar(x + width/2, cohere, width, label='Cohere (104B)', color=colors['cohere'])

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison: 50 Marketing Scenarios', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1.0)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.0%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.0%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('assets/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: assets/model_comparison.png")

# ============ Chart 5: Performance by Difficulty ============
fig, ax = plt.subplots(figsize=(8, 5))

difficulties = ['Easy', 'Medium', 'Hard']
our_f1 = [0.86, 0.65, 0.50]
cohere_f1 = [0.48, 0.64, 0.72]

x = np.arange(len(difficulties))
width = 0.35

bars1 = ax.bar(x - width/2, our_f1, width, label='Ours (8B)', color=colors['our_model'])
bars2 = ax.bar(x + width/2, cohere_f1, width, label='Cohere (104B)', color=colors['cohere'])

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score by Difficulty Level', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(difficulties, fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1.0)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('assets/difficulty_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: assets/difficulty_comparison.png")

print("\nAll charts generated successfully!")

