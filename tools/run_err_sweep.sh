#!/bin/zsh
# Wait until kepler/tess train+val error-channel views are complete, then run the val-only comparison.
cd /Users/ayushmondal/Desktop/Ideapad/VS/Research
done_all() {
  .venv/bin/python -c "
import sys; sys.path.insert(0, 'tools')
from rebuild_with_errors import split_complete
sys.exit(0 if all(split_complete(m, s) for m in ('Kepler','TESS') for s in ('train','val')) else 1)"
}
echo "[$(date '+%F %T')] waiting for train+val (both missions) in detection_views_err/" >> err_sweep.log
until done_all; do sleep 300; done
echo "[$(date '+%F %T')] views ready; starting sweep.py --errors" >> err_sweep.log
caffeinate -i -s .venv/bin/python stage_a_transit_model/sweep.py --errors >> err_sweep.log 2>&1
echo "[$(date '+%F %T')] sweep finished (exit $?)" >> err_sweep.log
