#!/bin/zsh
# One-shot progress report for the error-channel rebuild and the comparison sweep.
#   zsh tools/check_progress.sh            (from the Research folder)
cd "$(dirname "$0")/.." || exit 1

echo "=== $(date '+%a %d %b %H:%M') ==="
echo
echo "--- what is running ---"
pgrep -fl "build-split|rebuild_with_errors|run_err_sweep|sweep.py|cnn_lstm.py" | sed -E 's|/[^ ]*/Python ||' | cut -c1-110 \
  || echo "nothing running"
echo
echo "--- views (detection_views_err) ---"
.venv/bin/python tools/status.py 2>/dev/null | awk '/detection_views_err/{show=1} /^  (Kepler|TESS)/{show=($0 ~ /detection_views_err/)} show' \
  | grep -E "^  (Kepler|TESS)|progress|outcomes|rate|REMAINING"
echo
echo "--- rebuild driver (last 3) ---"
tail -3 rebuild_errors_chain.log 2>/dev/null | cut -c1-130
echo
echo "--- comparison sweep (last 5) ---"
tail -5 err_sweep.log 2>/dev/null | cut -c1-130
if [[ -f stage_a_transit_model/runs/error_channel_comparison.csv ]]; then
  echo
  echo "RESULT READY: .venv/bin/python stage_a_transit_model/sweep.py --errors --table"
fi
echo
echo "--- power ---"
pmset -g batt | sed -n 2p | sed 's/^ *-InternalBattery-0 ([^)]*)[[:space:]]*//'
