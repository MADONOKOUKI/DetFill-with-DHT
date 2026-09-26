#!/bin/bash
# Run from ANY host with ssh access. Instantly vacate reaper3 when someone asks for it:
# kills our eval workers, pushes all finished outputs to reaper2 (nothing lost; reaper2's
# full-72 run + reaper6 sweeper will cover the rest). Pass --purge to also delete the
# staged data (~4GB) from reaper3's /scratch.
#   usage: bash D_reaper3_vacate.sh [--purge]
set -u
SR=/scratch/madono/seg_retrain
RSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"
[ -f ~/.ssh/id_ed25519 ] && RSH="$RSH -i $HOME/.ssh/id_ed25519"

echo "== stop eval workers on reaper3 =="
$RSH reaper3 "for g in \$(seq 0 9); do tmux kill-session -t em_danbooregion_g\$g 2>/dev/null; done; pkill -9 -f '[e]val_on_' 2>/dev/null; sleep 4
  echo -n 'GPU procs left (ours should be 0): '; nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l"

echo "== push reaper3 outputs -> reaper2 (keep all finished work) =="
$RSH reaper3 "n=\$(find $SR/results_eval/danbooregion_model -path '*/200/*.png' 2>/dev/null | wc -l); echo \"  pushing \$n imgs\";
  rsync -a --ignore-existing -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=no' $SR/results_eval/danbooregion_model/ reaper2:$SR/results_eval/danbooregion_model/ && echo '  push OK'"

$RSH reaper2 "tmux kill-session -t r3pull 2>/dev/null" && echo "reaper2 r3pull syncer stopped" || true

if [ "${1:-}" = "--purge" ]; then
  echo "== purge staged data on reaper3 =="
  $RSH reaper3 "rm -rf $SR/gt $SR/sketch $SR/felz $SR/slic $SR/danbooregion $SR/handoff_ckpt $SR/results_eval $SR/logs; ls $SR 2>/dev/null || echo '  (empty)'"
fi
echo "reaper3 vacated. (reaper2 full-72 run + reaper6 sweeper will finish the matrix)"
