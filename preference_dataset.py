import torch
from typing import List, Optional, Dict, Any, Iterator


def get_batch_iterator(names: List[str], tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool  = True,
                       max_legnth: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = True, 
                       n_epochs: Optional[int] = None, 
                       n_examples: Optional[int] = None,
                       seed: int = 0,
                       silent: bool = False,
                       transform_config = None,
                       base_data_dir: Optional[str] = None,
                       cache_dir: Optional[str] = None,
                       reverse_dataset: bool = False) -> Iterator[Dict]:
    pass