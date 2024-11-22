#!/bin/bash

# 프로젝트 루트 디렉토리로 이동
cd "$(dirname "$0")/.."

# PYTHONPATH 설정하여 모듈을 찾을 수 있도록 함
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 기본 인자값 설정
ARGS=(
    --save_root save
    --project molebert_ft
    --version atom_128_each__motif_graph_64
)

# 외부에서 전달된 인자들을 기본 인자에 추가
# 동일한 인자가 전달되면 기본값을 덮어씁니다
ARGS+=("$@")

# 모든 데이터셋에서 molebert_ft.py 실행
echo "Running for dataset: tox21 (1/8)"
ALL_ARGS=("${ARGS[@]}" --dataset "tox21")
python train/molebert_ft.py "${ALL_ARGS[@]}"
# 71.90/71.75/71.90

echo "Running for dataset: toxcast (2/8)"
ALL_ARGS=("${ARGS[@]}" --dataset "toxcast")
python train/molebert_ft.py "${ALL_ARGS[@]}"
# 60.10/57.53/60.10

echo "Running for dataset: sider (3/8)"
ALL_ARGS=("${ARGS[@]}" --dataset "sider")
python train/molebert_ft.py "${ALL_ARGS[@]}"
# 59.04/60.27/58.47

echo "Running for dataset: clintox (4/8)"
ALL_ARGS=("${ARGS[@]}" --dataset "clintox")
python train/molebert_ft.py "${ALL_ARGS[@]}"
# 83.52/83.19/83.47

echo "Running for dataset: muv (5/8)"
ALL_ARGS=("${ARGS[@]}" --dataset "muv")
python train/molebert_ft.py "${ALL_ARGS[@]}"
# 72.38/51.84/73.52

echo "Running for dataset: hiv (6/8)"
ALL_ARGS=("${ARGS[@]}" --dataset "hiv")
python train/molebert_ft.py "${ALL_ARGS[@]}"
# 71.71/72.93/73.33

echo "Running for dataset: bbbp (7/8)"
ALL_ARGS=("${ARGS[@]}" --dataset "bbbp")
python train/molebert_ft.py "${ALL_ARGS[@]}"
# 70.59/58.55/63.69

echo "Running for dataset: bace (8/8)"
ALL_ARGS=("${ARGS[@]}" --dataset "bace")
python train/molebert_ft.py "${ALL_ARGS[@]}"
# 75.43/76.26/77.37