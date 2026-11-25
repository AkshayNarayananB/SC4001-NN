#!/bin/bash
# Local <-> ASPIRE sync for the journal extension (uses the `aspire` ssh alias / master connection).
#   bash scripts/journal/aspire/sync.sh code      # push code (no data/results/.git)
#   bash scripts/journal/aspire/sync.sh data      # push the fresh TEST set and the DAE checkpoint
#   bash scripts/journal/aspire/sync.sh pull      # pull finished journal outputs to /mnt/Data/akshay/journal_raw
set -euo pipefail
R=/scratch/users/ntu/akshayna
ssh() { command ssh -o BatchMode=yes -o ConnectTimeout=20 "$@"; }   # never hang on a password prompt
case "$1" in
  code) rsync -az --exclude .git --exclude data --exclude 'results/raw' --exclude checkpoints --exclude '__pycache__' \
          ./ aspire:~/SC4001-NN/ ;;
  data) # source JPEGs + questions only: run_readout degrades on the fly (bit-identical to the stored PNGs)
        ssh aspire "mkdir -p $R/data/pope_fresh $R/data/pope $R/checkpoints/dae/dae_v1 $R/logs $R/hf_cache"
        tar -C /mnt/Data/akshay/pope_fresh -cf - . | ssh aspire "tar -C $R/data/pope_fresh -xf -"
        tar -C data/pope -cf - questions coco_val2014 | ssh aspire "tar -C $R/data/pope -xf -"
        rsync -az checkpoints/dae/dae_v1/best.pt aspire:$R/checkpoints/dae/dae_v1/ ;;
  pull) # only finished run dirs (DONE marker); tar stream is faster than rsync for big trees
        for d in $(ssh aspire "cd $R/results/raw/journal && find . -name DONE -printf '%h\n'"); do
          [ -e "/mnt/Data/akshay/journal_raw/$d/DONE" ] && continue
          dst="/mnt/Data/akshay/journal_raw/$d"; tmp="$dst.part"
          rm -rf "$tmp"; mkdir -p "$tmp"
          ssh aspire "tar -C $R/results/raw/journal/$d -cf - ." | tar -C "$tmp" -xf -
          rm -rf "$dst"; mv "$tmp" "$dst"     # appears (with DONE) only once complete
          echo "pulled $d"
        done ;;
  *) echo "usage: $0 code|data|pull"; exit 1 ;;
esac
