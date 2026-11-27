#!/bin/bash
set -u
D=/home/smrc/samsung-nvme/sieun/beir_emb
H=213.192.6.100; P=40080; K=$HOME/.ssh/id_ed25519_runpod
E="ssh -p $P -i $K -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=30"
S=root@$H:/root/beir_emb/out_full
mkdir -p $D/logs
rsync -a --partial -e "$E" --exclude='*.npy' $S/ $D/ > $D/logs/small.log 2>&1
for ds in nq quora msmarco dbpedia-entity; do
  for m in small large; do
    ( mkdir -p $D/$ds
      rsync -a --partial -e "$E" --include="*text-embedding-3-${m}*.npy" --exclude='*' \
         $S/$ds/ $D/$ds/ > $D/logs/${ds}_${m}.log 2>&1 ) &
  done
done
wait
echo "ALL DONE $(date -Is)"
