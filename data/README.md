# Synthetic demo data

These four JSONL files were authored for a runnable local demo. They are not customer data, a public benchmark, or evidence of real-world accuracy.

They cover refunds, urgency, department routing, ordinal frustration, location facts, Unicode, and delivery negation. Training has 20 records and 56 decisions. Each other split has 6 records and 18 decisions. States and group IDs are disjoint, but wording and question templates are deliberately similar across splits. Metrics therefore measure this toy task, not generalization to new domains.

Refund and urgency answers require explicit evidence in the message. Frustration labels use explicit emotional language; urgency alone is not anger. Score labels are zero-based indices, and choice labels are option keys.

`./train.sh` uses these files by default. Default-path checkpoints are labeled as demos in their model ID and metadata. For actual evaluation, supply independently labeled train, validation, calibration, and test files at other paths, plus a new output directory. There is no claim that JEPA beats the pretrained baseline on this demo.
