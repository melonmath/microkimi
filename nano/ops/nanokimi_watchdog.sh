#!/bin/bash
# nanokimi userland watchdog: every 60 s, if train.py is dead and the
# run is not finished (no NIGHT_DONE), relaunch night_run.sh (auto resume).
while true; do
  if ! pgrep -f "python3 nano/train.py" > /dev/null 2>&1; then
    if ! grep -q "NIGHT_DONE" ~/train.log 2>/dev/null; then
      echo "$(date -u '+%F %T') train.py missing - relaunching night_run.sh" >> ~/watchdog.log
      tmux kill-session -t nanokimi 2>/dev/null
      tmux new-session -d -s nanokimi '~/night_run.sh'
    fi
  fi
  sleep 60
done
