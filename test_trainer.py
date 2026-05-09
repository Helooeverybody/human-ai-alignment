import torch
from trainers import get_tdpo_batch_logps
# ==========================================
# TEST SETUP
# ==========================================
torch.manual_seed(42)

B = 2  # Batch size
S = 4  # Sequence length
V = 5  # Vocab size

# 1. Create dummy model logits
logit = torch.randn(B, S, V)

# 2. Create dummy reference logits
ref_logit = torch.randn(B, S, V)
# TRICK: Make the 2nd sequence perfectly match the model logits
ref_logit[1] = logit[1].clone()

# 3. Create labels with -100 to test masking
# Because of the shift `labels[:, 1:]`, only the last 3 tokens are evaluated.
labels = torch.tensor([
    [0, 1, 3, -100],  # Evaluates: 1, 3 (Ignores -100) -> 2 valid tokens
    [0, -100, 2, 4]   # Evaluates: 2, 4 (Ignores -100) -> 2 valid tokens
])

# ==========================================
# RUN FUNCTION
# ==========================================
avg_margin, avg_kl, avg_logp = get_tdpo_batch_logps(logit, ref_logit, labels, avg_log_probs=True)
sum_margin, sum_kl, sum_logp = get_tdpo_batch_logps(logit, ref_logit, labels, avg_log_probs=False)

# ==========================================
# PRINT RESULTS
# ==========================================
print("--- AVERAGE LOG PROBS (avg_log_probs=True) ---")
print(f"Margin LogPs : {avg_margin}")
print(f"KL Divergence: {avg_kl}")
print(f"Model LogPs  : {avg_logp}\n")

print("--- SUM LOG PROBS (avg_log_probs=False) ---")
print(f"Margin LogPs : {sum_margin}")
print(f"KL Divergence: {sum_kl}")
print(f"Model LogPs  : {sum_logp}")