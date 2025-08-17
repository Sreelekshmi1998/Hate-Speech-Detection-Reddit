# Data Layout

This folder contains **anonymized sample data** for demonstration. The full raw dataset is **not** included.

## Files
- `samples/sample_data.csv` — small sample (up to 50 rows), with **heavily anonymized** `tweet` text and label column `class`.

## Columns
- `tweet`: anonymized text (URLs -> `[URL]`, mentions -> `[USER]`, hashtags -> `[TAG]`, digits -> `0`, words masked like `h***o`).
- `class`: ground-truth label for classification.

## Privacy & Safety
- The sample is produced from the original data via heavy masking to avoid exposing sensitive or toxic content.

