"""
data_loader.py — Data loading and preparation for LLMScape experiments.

Handles tokenization, chunking, and DataLoader creation for evaluation
and gradient collection.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class TokenChunkDataset(Dataset):
    """Dataset of fixed-length token chunks."""

    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {
            'input_ids': self.chunks[idx],
            'attention_mask': torch.ones(len(self.chunks[idx]), dtype=torch.long),
        }


def prepare_data(tokenizer, config):
    """
    Load and tokenize dataset, create evaluation and gradient DataLoaders.

    Args:
        tokenizer: HuggingFace tokenizer
        config: dict with data configuration

    Returns:
        dict with keys 'eval_loader', 'grad_loader', 'hvp_loader', 'chunks'
    """
    data_cfg = config['data']
    seq_len = data_cfg['seq_len']

    # Load dataset
    dataset = load_dataset(
        data_cfg['dataset'],
        data_cfg['dataset_config'],
        split=data_cfg['split'],
    )

    # Filter non-empty text and tokenize
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    all_tokens = tokenizer(
        '\n'.join(texts[:200]),
        return_tensors='pt',
        truncation=False,
    )['input_ids'][0]

    # Create fixed-length chunks
    chunks = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        chunks.append(all_tokens[i:i + seq_len])

    # Create DataLoaders
    n_eval = min(data_cfg['n_eval_chunks'], len(chunks))
    eval_dataset = TokenChunkDataset(chunks[:n_eval])
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=data_cfg['eval_batch_size'],
        shuffle=False,
    )

    n_grad = min(data_cfg['n_grad_batches'], len(chunks))
    grad_dataset = TokenChunkDataset(chunks[:n_grad])
    grad_loader = DataLoader(
        grad_dataset,
        batch_size=data_cfg.get('grad_batch_size', 1),
        shuffle=False,
    )

    hvp_cfg = config['direction']['tier3']
    hvp_n = min(hvp_cfg['hvp_max_batches'], len(chunks))
    hvp_dataset = TokenChunkDataset(chunks[:hvp_n])
    hvp_loader = DataLoader(
        hvp_dataset,
        batch_size=hvp_cfg.get('hvp_batch_size', 1),
        shuffle=False,
    )

    return {
        'eval_loader': eval_loader,
        'grad_loader': grad_loader,
        'hvp_loader': hvp_loader,
        'chunks': chunks,
        'n_total_chunks': len(chunks),
    }


def prepare_custom_data(tokenizer, texts, seq_len=256, batch_size=4, max_chunks=50):
    """
    Create DataLoader from custom text list (for dataset sensitivity experiments).

    Args:
        tokenizer: HuggingFace tokenizer
        texts: list of strings
        seq_len: token chunk length
        batch_size: batch size
        max_chunks: max number of chunks

    Returns:
        DataLoader
    """
    all_tokens = tokenizer(
        '\n'.join(texts),
        return_tensors='pt',
        truncation=False,
    )['input_ids'][0]

    chunks = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        chunks.append(all_tokens[i:i + seq_len])
        if len(chunks) >= max_chunks:
            break

    dataset = TokenChunkDataset(chunks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
