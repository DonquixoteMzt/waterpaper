#!/bin/bash
# Download models via wget from hf-mirror.com
# Much more reliable than the xethub CDN

MODEL_DIR="/home/leo/waterpaper/work3/exp/normal_exp/models"
MIRROR="https://hf-mirror.com"

download_model() {
    local repo=$1
    local filename=$2
    local out_dir="$MODEL_DIR/$(echo $repo | tr '/' '_')"
    mkdir -p "$out_dir"

    local out_file="$out_dir/$filename"
    if [ -f "$out_file" ]; then
        echo "SKIP: $repo/$filename (already exists)"
        return 0
    fi

    echo "Downloading: $repo/$filename ..."
    wget --timeout=120 --tries=5 -q --show-progress \
        -O "$out_file" \
        "$MIRROR/$repo/resolve/main/$filename" 2>&1

    if [ $? -eq 0 ]; then
        echo "OK: $repo/$filename ($(du -sh "$out_file" | cut -f1))"
    else
        echo "FAIL: $repo/$filename"
        rm -f "$out_file"
    fi
}

# First download all config files (small)
for repo in \
    "Qwen/Qwen3-0.6B" \
    "TinyLlama/TinyLlama-1.1B-step-50K-105b" \
    "TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b" \
    "TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T" \
    "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T" \
    "TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T" \
    ; do
    out_dir="$MODEL_DIR/$(echo $repo | tr '/' '_')"
    mkdir -p "$out_dir"
    for f in config.json tokenizer.json tokenizer_config.json generation_config.json special_tokens_map.json tokenizer.model; do
        if [ ! -f "$out_dir/$f" ]; then
            wget --timeout=30 --tries=3 -q -O "$out_dir/$f" "$MIRROR/$repo/resolve/main/$f" 2>/dev/null
        fi
    done
    echo "Config files for $repo: done"
done

# Download model weights in parallel
# Qwen3-0.6B (1.2GB, ~5 min)
download_model "Qwen/Qwen3-0.6B" "model.safetensors" &
P1=$!

# TinyLlama models (~4.4GB each, some use pytorch_model.bin, some use model.safetensors)
download_model "TinyLlama/TinyLlama-1.1B-step-50K-105b" "pytorch_model.bin" &
P2=$!

download_model "TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b" "pytorch_model.bin" &
P3=$!

download_model "TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T" "model.safetensors" &
P4=$!

download_model "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T" "pytorch_model.bin" &
P5=$!

download_model "TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T" "pytorch_model.bin" &
P6=$!

echo "Waiting for all downloads..."
wait $P1 && echo "Qwen3-0.6B done" &
wait $P2 && echo "TinyLlama 50K done" &
wait $P3 && echo "TinyLlama 240K done" &
wait $P4 && echo "TinyLlama 480K done" &
wait $P5 && echo "TinyLlama 715K done" &
wait $P6 && echo "TinyLlama 955K done" &

wait
echo "All downloads complete!"
ls -la $MODEL_DIR/*/ 2>/dev/null
