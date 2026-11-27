#!/bin/bash
# BEIR OpenAI 임베딩 (text-embedding-3-small 1536d / -large 3072d), 63 GB.
# ⚠ pull 방향 — dskbd084 는 내부 IP 뿐이라 팟에서 push 가 안 된다.
set -u
D=/home/smrc/samsung-nvme/sieun/beir_emb
H=213.192.6.100; P=40080; K=$HOME/.ssh/id_ed25519_runpod
E="ssh -p $P -i $K -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=30"
S=root@$H:/root/beir_emb/out_full
mkdir -p $D/logs
# 작은 것(README/SHA256SUMS/스크립트/jsonl/tsv) 먼저 — 체크섬을 일찍 확보
rsync -a --partial -e "$E" --exclude='*.npy' $S/ $D/ > $D/logs/small.log 2>&1
# npy 는 데이터셋별 4스트림 + 모델별로 쪼개 8병렬
for ds in nq quora msmarco dbpedia-entity; do
  for m in small large; do
    rsync -a --partial -e "$E" --include="*/" --include="*text-embedding-3-${m}*.npy" --exclude='*' \
       $S/$ds/ $D/$ds/ > $D/logs/${ds}_${m}.log 2>&1 &
  done
  wait   # 데이터셋 단위로 2개씩, 전체는 순차 → 동시 8개가 아니라 2개
done
echo "ALL DONE $(date -Is)"
