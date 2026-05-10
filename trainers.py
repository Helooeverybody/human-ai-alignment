import torch 
import transformers 
import torch.nn.functional as F
from typing import Tuple, Optional 
import torch.nn as nn
from omegaconf import DictConfig
import torch.distributed as dist
from preference_dataset import get_batch_iterator

def rank0_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


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


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


class BaseTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank = 0,
                 world_size = 1, transform_config = None):
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.base_data_dir = config.base_data_dir


        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # set pad token to eos token if pad token is not defined, since we need to pad the input to the same length for batching, 
        # and we don't want to introduce new tokens that are not in the original vocabulary of the model.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.policy = policy
        self.reference_model = reference_model
        
        # Use the passed transform_config if available
        self.transform_config = transform_config
        print(self.transform_config)

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, transform_config=transform_config)
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, transform_config=transform_config)
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

        