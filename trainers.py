import torch 
import transformers 
import torch.nn.functional as F
from typing import Tuple


def get_batch_logps(logit: torch.FloatTensor, labels: torch.LongTensor, avg_log_probs: bool = True):
    """
    logit: (batch_size, seq_len, vocab_size)
    labels: (batch_size, seq_len)
    """

    assert logit.shape[:-1] == labels.shape, "logit and labels must have the same batch_size and seq_len"

    # next token prediction shift 
    logit = logit[:, :-1,:]
    labels = labels[:, 1:].clone() # clone to avoid in-place modification of original labels

    # calculate log probabilities of labels
    mask = (labels != -100).float() # mask for valid tokens
    labels[labels == -100] = 0 # set ignore index to 0, since we will mask them out later
    logps = logit.log_softmax(dim=-1).gather(dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)

    if avg_log_probs:
        return (logps * mask).sum(dim = -1) / mask.sum(dim = -1)
    else:
        return (logps * mask).sum(dim = -1)
    

def get_tdpo_batch_logps(logit: torch.FloatTensor, ref_logit: torch.FloatTensor, 
                         labels: torch.LongTensor, avg_log_probs: bool = True):
    """
    logit: (batch_size, seq_len, vocab_size)
    ref_logit: (batch_size, seq_len, vocab_size)
    labels: (batch_size, seq_len)
    """

    assert logit.shape == ref_logit.shape, "logit and ref_logit must have the same shape"
    assert logit.shape[:-1] == labels.shape, "logit and labels must have the same batch_size and seq_len"


    # next token prediction shift 
    logit = logit[:, :-1,:]
    ref_logit = ref_logit[:, :-1,:]
    labels = labels[:, 1:].clone() # clone to avoid in-place modification of original labels

    # calculate kl divergence between logit and ref_logit
    logit_ps = logit.log_softmax(dim=-1)
    ref_logit_softmax = ref_logit.softmax(dim=-1)
    ref_logit_ps = ref_logit_softmax.log()
    kl_token_logps = (ref_logit_softmax * (ref_logit_ps - logit_ps)).sum(dim = -1)

    # calculate log probabilities of labels
    mask = (labels != -100).float() # mask for valid tokens
    labels[labels == -100] = 0 # set ignore index to 0, since we will mask them out later
    token_logps = logit_ps.gather(dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)
    ref_token_logps = ref_logit_ps.gather(dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)
    margin_logps = token_logps - ref_token_logps

    if avg_log_probs:
        return ((margin_logps * mask).sum(dim = -1) / mask.sum(dim = -1),
                (kl_token_logps * mask).sum(dim = -1) / mask.sum(dim = -1),
                (token_logps * mask).sum(dim = -1) / mask.sum(dim = -1))
    else:
        return ((margin_logps * mask).sum(dim = -1),
                (kl_token_logps * mask).sum(dim = -1),
                (token_logps * mask).sum(dim = -1))


def get_tisdpo_batch_logps(logit: torch.FloatTensor, ref_logit: torch.FloatTensor,
                            labels: torch.LongTensor,weight: torch.FloatTensor = None, 
                            avg_log_probs: bool = True):
    assert logit.shape == ref_logit.shape, "logit and ref_logit must have the same shape"
    assert logit.shape[:-1] == labels.shape, "logit and labels must have the same batch_size and seq_len"


    # next token prediction shift 
    logit = logit[:, :-1,:]
    ref_logit = ref_logit[:, :-1,:]
    labels = labels[:, 1:].clone() # clone to avoid in-place modification of original labels

    # calculate kl divergence between logit and ref_logit
    logit_ps = logit.log_softmax(dim=-1)
    ref_logit_softmax = ref_logit.softmax(dim=-1)
    ref_logit_ps = ref_logit_softmax.log()
    kl_token_logps = (ref_logit_softmax * (ref_logit_ps - logit_ps)).sum(dim = -1)

    # calculate log probabilities of labels
    mask = (labels != -100).float() # mask for valid tokens
    labels[labels == -100] = 0 # set ignore index to 0, since we will mask them out later
    token_logps = logit_ps.gather(dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)
    ref_token_logps = ref_logit_ps.gather(dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)
    margin_logps = token_logps - ref_token_logps

    weight = weight[:, 1:] if weight is not None else torch.ones_like(mask)

    if avg_log_probs:
        return ((margin_logps * mask * weight).sum(dim = -1) / (mask * weight).sum(dim = -1),
                (kl_token_logps * mask * weight).sum(dim = -1) / (mask * weight).sum(dim = -1),
                (token_logps * mask * weight).sum(dim = -1) / (mask * weight).sum(dim = -1))
    else:
        return ((margin_logps * mask * weight).sum(dim = -1),
                (kl_token_logps * mask * weight).sum(dim = -1),
                (token_logps * mask * weight).sum(dim = -1))




def tdpo_loss(chosen_logps_margin: torch.FloatTensor,
              rejected_logps_margin: torch.FloatTensor,
              chosen_position_kl: torch.FloatTensor,
              rejected_position_kl: torch.FloatTensor,
              beta: float, alpha: float = 0.5, if_tdpo2: bool = True):
    """
    chosen_logps_margin: (batch_size,)
    rejected_logps_margin: (batch_size,)
    chosen_position_kl: (batch_size,)
    rejected_position_kl: (batch_size,)
    """

    chosen_values = chosen_logps_margin + chosen_position_kl
    rejected_values = rejected_logps_margin + rejected_position_kl

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if not if_tdpo2:
        logits = chosen_rejected_logps_margin - (rejected_position_kl - chosen_position_kl)    # tdpo1
    else:
        # detach(stop gradient) kl divergence to stabilize training, since kl divergence can be very large at the beginning of training, which will lead to large variance of gradients and make training unstable.
        logits = chosen_rejected_logps_margin - alpha * (rejected_position_kl - chosen_position_kl.detach())  # tdpo2

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return (losses, chosen_rewards, rejected_rewards)


def tisdpo_loss(chosen_logps_margin: torch.FloatTensor,
                rejected_logps_margin: torch.FloatTensor,
                chosen_position_kl: torch.FloatTensor,
                rejected_position_kl: torch.FloatTensor,
                beta: float, alpha: float = 0.5, token_level: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    if token_level:
        chosen_values = chosen_logps_margin - chosen_position_kl
        rejected_values = rejected_logps_margin - rejected_position_kl
    else:
        chosen_values = chosen_logps_margin
        rejected_values = rejected_logps_margin

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if token_level:
        logits = chosen_rejected_logps_margin - alpha * (chosen_position_kl - rejected_position_kl)  
    else:
        logits = chosen_rejected_logps_margin

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards







