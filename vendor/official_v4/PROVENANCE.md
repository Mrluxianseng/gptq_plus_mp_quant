This directory contains the byte-verified official REAL-Q baseline and the
registered v4 storage derivative used in the archived 2026-10-19 comparison.

Upstream: https://github.com/QianerPKU/REAL-Q
Commit: dbf35ae27b5a2a35d6823cf234b7959fc747c02d

`manifest.json` records all 239 original byte identities and the archived parent
manifest SHA256. Original pending/acceptance fields were NOT rewritten. This
portable export makes no new GPU acceptance claim.

`ablations.json` contains only the exact changed source members from the five
previously registered ablations. Each records its parent and replacement SHA256.
The default matrix uses three of them. Source contents are not synthesized at
runtime and the archived local source trees remain untouched.

Model weights, datasets, experiment payloads, credentials, private PDFs and
server-specific launch receipts are not included. `utils/had_mat.pt` is the
small upstream mathematical Hadamard constant table, not a model checkpoint.
