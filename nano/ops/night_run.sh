#!/bin/bash
# nanokimi night run: train (time-cap 9h) → export MKIM0002 → NIGHT_DONE
# automatic resume if a checkpoint exists; export ONLY on a clean finish
# (a kill/crash (exit != 0) lets the watchdog relaunch instead of exporting).
exec 9> ~/night_run.lock
flock -n 9 || { echo "night_run already running - aborting" >> ~/train.log; exit 1; }
export MALLOC_ARENA_MAX=4
cd ~
RESUME=""
if [ -f ~/nano_out/ckpt/ckpt_latest.pt ]; then RESUME="--resume"; fi
echo "[night_run.sh $(date -u '+%F %T') - resume=$RESUME]" >> ~/train.log
~/venv/bin/python3 nano/train.py --data ~/nano_out/tokens.bin --out ~/nano_out/ckpt \
  --layers 8 --batch 32 --seq 256 --steps 560 --lr 3e-4 --warmup 25 \
  --threads 32 --log-every 10 --ckpt-every 100 --ckpt-secs 1800 \
  --trim-every 100 --rss-cap 24.0 --max-hours 9.0 $RESUME >> ~/train.log 2>&1
RC=$?
if [ $RC -eq 0 ]; then
  echo "[train finished cleanly (rc=0) - MKIM0002 export]" >> ~/train.log
  ~/venv/bin/python3 nano/export.py --ckpt ~/nano_out/ckpt/ckpt_latest.pt \
    --out ~/nano_out/nanokimi.bin >> ~/train.log 2>&1
  echo "NIGHT_DONE" >> ~/train.log
else
  echo "[train died abnormally (rc=$RC) - NO export, the watchdog will relaunch]" >> ~/train.log
fi
exit $RC
