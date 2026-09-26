# Sample views

252 real views (2.4 MB) taken from the full build, label-stratified, so the code in this repository can be run straight after cloning.

**These are for smoke-testing, not for results.** About 40 rows per split cannot reproduce the reported metrics; they exist so a reviewer can confirm the pipeline runs and produces sensible output without rebuilding the full set, which means re-downloading roughly 120 GB of photometry over about 30 hours.

```bash
python stage_a_transit_model/baselines.py \
    --train-views detection_views_sample/kepler_train \
    --val-views   detection_views_sample/kepler_val

python stage_a_transit_model/cnn_lstm.py --epochs 3 \
    --train-views detection_views_sample/kepler_train \
    --val-views   detection_views_sample/kepler_val
```

The full view set is linked from the top-level README.
