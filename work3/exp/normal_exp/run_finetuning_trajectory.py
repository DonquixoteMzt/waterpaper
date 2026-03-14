"""
Fine-tuning trajectory experiment for MMSP Method A (Trajectory-PCA)
and MMSP Method B (Anchor-Point Projection).

Fine-tunes Qwen3-0.6B-Base on WikiText-2 for ~100 steps, saving checkpoints
to create a genuine training trajectory. Then:
1. Runs trajectory_pca on the checkpoint sequence (Method A)
2. Runs anchor_point_projection between step-0 and step-100 (Method B)
3. Evaluates loss surfaces and computes metrics
"""

import os
import json
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Local imports
from multi_model import trajectory_pca, anchor_point_projection, compute_model_distance
from normalization import apply_tadn, get_normalization_units
from grid_evaluation import evaluate_2d_surface, evaluate_loss, evaluate_1d_curve
from data_loader import TokenChunkDataset
from metrics import compute_surface_metrics


def load_chunks(tokenizer, split="train", seq_len=256, max_chunks=200):
    """Load WikiText-2 and return list of token chunks."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    all_tokens = tokenizer(
        '\n'.join(texts[:200]),
        return_tensors='pt',
        truncation=False,
    )['input_ids'][0]
    chunks = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        chunks.append(all_tokens[i:i + seq_len])
        if len(chunks) >= max_chunks:
            break
    return chunks


def fine_tune_and_save_checkpoints(model, train_chunks, device,
                                    n_steps=100, checkpoint_every=20, lr=1e-5):
    """Fine-tune model and save parameter checkpoints at regular intervals."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    checkpoints = []
    checkpoint_steps = []
    losses = []

    # Save step-0 checkpoint
    step0_params = {n: p.data.clone().cpu() for n, p in model.named_parameters()}
    checkpoints.append(step0_params)
    checkpoint_steps.append(0)

    step = 0
    epoch = 0

    while step < n_steps:
        epoch += 1
        for chunk in train_chunks:
            if step >= n_steps:
                break

            input_ids = chunk.unsqueeze(0).to(device)
            if input_ids.shape[1] < 10:
                continue

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=input_ids.clone())
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            losses.append(loss.item())

            if step % checkpoint_every == 0:
                ckpt_params = {n: p.data.clone().cpu() for n, p in model.named_parameters()}
                checkpoints.append(ckpt_params)
                checkpoint_steps.append(step)
                print(f"  Step {step}: loss={loss.item():.4f} (checkpoint saved)")
            elif step % 10 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")

    model.eval()
    return checkpoints, checkpoint_steps, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n-steps', type=int, default=100)
    parser.add_argument('--checkpoint-every', type=int, default=20)
    parser.add_argument('--grid-size', type=int, default=21)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    output_dir = 'results/finetuning_trajectory'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Fine-Tuning Trajectory Analysis (MMSP Methods A & B)")
    print("=" * 70)

    # Load model
    model_name = "Qwen/Qwen3-0.6B-Base"
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True
    ).to(device)

    # Prepare data
    print("Preparing training data...")
    train_chunks = load_chunks(tokenizer, split="train", seq_len=256, max_chunks=200)
    print(f"  {len(train_chunks)} training chunks")

    print("Preparing evaluation data...")
    eval_chunks = load_chunks(tokenizer, split="test", seq_len=256, max_chunks=50)
    eval_dataset = TokenChunkDataset(eval_chunks)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)
    print(f"  {len(eval_chunks)} eval chunks")

    # Compute baseline loss before fine-tuning
    model.eval()
    baseline_loss = evaluate_loss(model, eval_loader, device)
    print(f"Baseline loss (before fine-tuning): {baseline_loss:.4f}")

    # ============================================================
    # Phase 1: Fine-tune and collect checkpoints
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: Fine-tuning for {args.n_steps} steps")
    print(f"{'='*60}")

    t0 = time.time()
    checkpoints, checkpoint_steps, training_losses = fine_tune_and_save_checkpoints(
        model, train_chunks, device,
        n_steps=args.n_steps,
        checkpoint_every=args.checkpoint_every,
        lr=1e-5
    )
    ft_time = time.time() - t0
    print(f"Fine-tuning complete in {ft_time:.1f}s")
    print(f"Checkpoints saved at steps: {checkpoint_steps}")

    # Compute post-fine-tuning loss
    model.eval()
    post_ft_loss = evaluate_loss(model, eval_loader, device)
    print(f"Post-fine-tuning loss: {post_ft_loss:.4f}")

    # Compute distances between consecutive checkpoints
    distances = []
    for i in range(1, len(checkpoints)):
        dist = compute_model_distance(checkpoints[i-1], checkpoints[i])
        distances.append(dist)
        print(f"  Distance step {checkpoint_steps[i-1]} -> {checkpoint_steps[i]}: {dist:.4f}")

    # ============================================================
    # Phase 2: MMSP Method A — Trajectory-PCA
    # ============================================================
    print(f"\n{'='*60}")
    print("Phase 2: MMSP Method A — Trajectory-PCA")
    print(f"{'='*60}")

    t0 = time.time()
    pca_directions, projected_coords, centroid, explained_var = trajectory_pca(
        checkpoints, k=2
    )
    traj_pca_time = time.time() - t0
    print(f"Trajectory PCA complete in {traj_pca_time:.1f}s")
    print(f"Explained variance: [{explained_var[0]:.4f}, {explained_var[1]:.4f}]")
    print(f"Projected coordinates:")
    for i, (x, y) in enumerate(projected_coords):
        print(f"  Step {checkpoint_steps[i]}: ({x:.4f}, {y:.4f})")

    # Apply TADN to PCA directions
    norm_units = get_normalization_units(model)
    tadn_d1 = apply_tadn(pca_directions[0], model, norm_units)
    tadn_d2 = apply_tadn(pca_directions[1], model, norm_units)

    # Evaluate 2D loss surface centered at centroid
    print("\nEvaluating 2D loss surface centered at trajectory centroid...")

    # Load centroid into model
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in centroid:
                param.data.copy_(centroid[name].to(param.dtype).to(param.device))

    alphas_t, betas_t, surface_t = evaluate_2d_surface(
        model, tadn_d1, tadn_d2, eval_loader, device,
        grid_range=1.0, grid_size=args.grid_size
    )

    traj_metrics = compute_surface_metrics(alphas_t, betas_t, surface_t)
    print(f"Surface metrics: loss_range={traj_metrics['loss_range']:.2f}, roughness={traj_metrics['roughness']:.4f}")

    # Evaluate loss at each checkpoint position
    checkpoint_losses = []
    for i, ckpt in enumerate(checkpoints):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ckpt:
                    param.data.copy_(ckpt[name].to(param.dtype).to(param.device))
        loss = evaluate_loss(model, eval_loader, device)
        checkpoint_losses.append(loss)
        print(f"  Step {checkpoint_steps[i]}: loss = {loss:.4f}")

    # Save trajectory surface
    np.savez(os.path.join(output_dir, 'trajectory_surface.npz'),
             Z=surface_t, alphas=alphas_t, betas=betas_t,
             coords=projected_coords, steps=checkpoint_steps)

    # ============================================================
    # Phase 3: MMSP Method B — Anchor-Point Projection
    # ============================================================
    print(f"\n{'='*60}")
    print("Phase 3: MMSP Method B — Anchor-Point Projection")
    print(f"{'='*60}")

    # Use step-0 (pre-training) and final step (post-fine-tuning)
    params_pre = checkpoints[0]
    params_post = checkpoints[-1]

    t0 = time.time()
    d1_ap, d2_ap, midpoint, model_dist = anchor_point_projection(params_pre, params_post)
    ap_time = time.time() - t0
    print(f"Anchor-Point Projection complete in {ap_time:.1f}s")
    print(f"Model distance (pre -> post): {model_dist:.4f}")

    # Apply TADN
    tadn_d1_ap = apply_tadn(d1_ap, model, norm_units)
    tadn_d2_ap = apply_tadn(d2_ap, model, norm_units)

    # Load midpoint into model
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in midpoint:
                param.data.copy_(midpoint[name].to(param.dtype).to(param.device))

    # Evaluate surface centered at midpoint
    print("Evaluating 2D surface centered at midpoint...")
    alphas_ap, betas_ap, surface_ap = evaluate_2d_surface(
        model, tadn_d1_ap, tadn_d2_ap, eval_loader, device,
        grid_range=1.0, grid_size=args.grid_size
    )

    ap_metrics = compute_surface_metrics(alphas_ap, betas_ap, surface_ap)
    print(f"Anchor-point surface metrics: loss_range={ap_metrics['loss_range']:.2f}, roughness={ap_metrics['roughness']:.4f}")

    # Evaluate loss at pre and post positions
    for label, ckpt in [("Pre (step 0)", params_pre), ("Post (final)", params_post)]:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ckpt:
                    param.data.copy_(ckpt[name].to(param.dtype).to(param.device))
        loss = evaluate_loss(model, eval_loader, device)
        print(f"  {label}: loss = {loss:.4f}")

    # Evaluate 1D cross-section along d1 (pre -> post direction)
    print("Computing 1D cross-section along pre->post direction...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in midpoint:
                param.data.copy_(midpoint[name].to(param.dtype).to(param.device))

    cross_alphas, cross_losses = evaluate_1d_curve(
        model, tadn_d1_ap, eval_loader, device,
        alpha_range=(-2.0, 2.0), n_points=41
    )

    np.savez(os.path.join(output_dir, 'anchor_point_surface.npz'),
             Z=surface_ap, alphas=alphas_ap, betas=betas_ap,
             cross_section=cross_losses, cross_alphas=cross_alphas)

    # ============================================================
    # Save all results
    # ============================================================
    results = {
        "experiment": "Fine-Tuning Trajectory & Anchor-Point Analysis",
        "model": model_name,
        "fine_tuning": {
            "n_steps": args.n_steps,
            "checkpoint_every": args.checkpoint_every,
            "lr": 1e-5,
            "baseline_loss": baseline_loss,
            "post_ft_loss": post_ft_loss,
            "training_losses": training_losses,
            "checkpoint_steps": checkpoint_steps,
            "checkpoint_losses": checkpoint_losses,
            "inter_checkpoint_distances": distances,
            "fine_tuning_time_seconds": ft_time
        },
        "trajectory_pca": {
            "explained_variance": explained_var,
            "projected_coords": projected_coords,
            "surface_metrics": traj_metrics,
            "time_seconds": traj_pca_time
        },
        "anchor_point": {
            "model_distance": model_dist,
            "surface_metrics": ap_metrics,
            "cross_section_losses": cross_losses.tolist(),
            "cross_section_alphas": cross_alphas.tolist(),
            "time_seconds": ap_time
        }
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x))

    print(f"\nResults saved to {output_dir}/")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline loss:         {baseline_loss:.4f}")
    print(f"Post-FT loss:          {post_ft_loss:.4f}")
    print(f"Model distance:        {model_dist:.4f}")
    print(f"Trajectory PCA EV:     [{explained_var[0]:.4f}, {explained_var[1]:.4f}]")
    print(f"Trajectory roughness:  {traj_metrics['roughness']:.4f}")
    print(f"Anchor-point roughness:{ap_metrics['roughness']:.4f}")
    print(f"Cross-section range:   {max(cross_losses) - min(cross_losses):.2f}")

    print("\nTrajectory coordinates:")
    for i in range(len(checkpoint_steps)):
        x, y = projected_coords[i]
        print(f"  Step {checkpoint_steps[i]:4d}: ({x:8.4f}, {y:8.4f}) loss={checkpoint_losses[i]:.4f}")


if __name__ == '__main__':
    main()
