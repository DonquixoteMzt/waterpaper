"""Download all required models in parallel."""
from huggingface_hub import snapshot_download
import os
import sys
import multiprocessing

def download_model(model_name):
    try:
        path = snapshot_download(model_name, resume_download=True)
        print(f"OK: {model_name} -> {path}")
        return True
    except Exception as e:
        print(f"FAIL: {model_name}: {e}")
        return False

if __name__ == '__main__':
    models = [
        'TinyLlama/TinyLlama-1.1B-step-50K-105b',
        'TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b',
        'TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T',
        'TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T',
        'TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T',
        'Qwen/Qwen3-0.6B',
    ]

    with multiprocessing.Pool(6) as pool:
        results = pool.map(download_model, models)

    for m, r in zip(models, results):
        print(f"  {m}: {'SUCCESS' if r else 'FAILED'}")
