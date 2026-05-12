---
name: source-ingest
description: Use when adding a paper, PDF, source package, or AI-readable derived source artifact to this ICLR 2027 repo.
---

# Source Ingest

## Required behavior

- Preserve originals.
- Create derived text or Markdown instead of editing raw files.
- Keep source boundaries explicit.
- Update `소스/INDEX.md` after adding a durable source.

## Steps

1. Identify the source title, origin path, and source type.
2. Preserve or copy the original without modifying it.
3. Create AI-readable derived text with page or section anchors when possible.
4. Create a source card or guide that states supported and unsupported claims.
5. Create or update a manifest with provenance and checksum information when available.
6. Verify that the derived artifact is readable and linked from the index.

## Output

Report original path, derived files, manifest path, source boundary, and what was not verified.
