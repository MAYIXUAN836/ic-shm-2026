# Evaluation utilities

All commands take user-supplied files; no evaluation image, organizer label, or
prediction is included in this code repository.

## Stage 1 multi-label scoring

For the 240-image regression record, use the prediction field
`predicted_categories` and reference field `target_categories`:

```bash
python evaluation/score_multilabel.py \
  --predictions <stage1-predictions.jsonl> \
  --references <damage240-references.jsonl> \
  --output <metrics.json>
```

For references storing categories under `target.damage_categories`, pass
`--reference-field target.damage_categories`.

## Integrated description scoring

For the project Q2 JSON reference file, use `q2-json`:

```bash
python evaluation/score_description.py \
  --predictions <final-predictions.jsonl> \
  --references <description.json> --reference-format q2-json \
  --output <summary.json> --per-image-output <scores.csv>
```

For the controlled 116-image holdout, first rebuild the frozen expert routes,
then score against the JSONL `description_label` field:

```bash
python evaluation/rebuild_controlled116.py \
  --holdout <holdout.jsonl> --expert-a <a.jsonl> \
  --expert-b-v07 <b-v07.jsonl> --expert-b-v08 <b-v08.jsonl> \
  --output <combined.jsonl>
python evaluation/score_description.py \
  --predictions <combined.jsonl> --references <holdout.jsonl> \
  --reference-format jsonl --output <summary.json> --per-image-output <scores.csv>
```

`score_description.py` implements METEOR with synonym matching disabled and
the token-F1 definition used for the reported description evaluations.
