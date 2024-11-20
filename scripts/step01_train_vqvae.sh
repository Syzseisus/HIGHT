#!/bin/bash

# 프로젝트 루트 디렉토리로 이동
cd "$(dirname "$0")/.."

# PYTHONPATH 설정하여 모듈을 찾을 수 있도록 함
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 기본 인자값 설정
ARGS=(
    --save_root save
    --project tokenizer
    --version atom_128_each__motif_graph_64
)

# 외부에서 전달된 인자들을 기본 인자에 추가
# 동일한 인자가 전달되면 기본값을 덮어씁니다
ARGS+=("$@")

# vqvae.py 실행
python train/vqvae.py "${ARGS[@]}"