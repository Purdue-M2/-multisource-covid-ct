"""Loss functions for multi-task training.

Implements Logit-Adjusted Cross-Entropy (Menon et al., 2021) for the
source classification head, which corrects for uneven hospital
contributions to the training set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedLoss(nn.Module):
    """Logit-adjusted cross-entropy for imbalanced source classification.

    Adds log-prior offsets to logits before computing cross-entropy,
    ensuring that underrepresented sources receive proportionate
    gradient signal.

    Args:
        source_frequencies: List of per-source proportions from training set.

    Reference:
        Menon et al., "Long-tail learning via logit adjustment", ICLR 2021.
    """

    def __init__(self, source_frequencies):
        super().__init__()
        freqs = torch.tensor(source_frequencies, dtype=torch.float32)
        self.register_buffer('log_priors', torch.log(freqs))

    def forward(self, logits, targets):
        adjusted = logits + self.log_priors
        return F.cross_entropy(adjusted, targets)


def build_criteria(source_frequencies, device):
    """Build loss functions for both heads.

    Returns:
        covid_criterion: BCEWithLogitsLoss for COVID-19 detection.
        source_criterion: LogitAdjustedLoss for source identification.
    """
    covid_criterion = nn.BCEWithLogitsLoss()
    source_criterion = LogitAdjustedLoss(source_frequencies).to(device)
    return covid_criterion, source_criterion
