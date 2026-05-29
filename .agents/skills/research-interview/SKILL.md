---
name: research-interview
description: Use when an ICLR 2027 research request is ambiguous or high-risk before server experiment planning, result import, result interpretation, report drafting, paper claim/evidence-boundary work, source work, or durable task design. Do not use for simple file lookup, clearly specified commands, or requests that already include the needed config, run ID, manifest, scope, and completion criteria.
---

# Research Interview

## Purpose

Turn a vague research-operation request into a decision-ready brief before using execution skills such as `server-run-prep`, `result-import`, `experiment-report`, or `source-ingest`.

This is a gate, not a replacement for those skills. Use it only when missing information could cause a wrong experiment, unsupported claim, unsafe result interpretation, or vague long-running task.

## First Move

1. Read the smallest relevant project context first.
2. Resolve facts from the repo before asking the user.
3. Pick the single highest-risk unknown.
4. Ask one question only.

Use this question format:

```text
현재 이해: <요청을 한 문장으로 요약>
막힌 결정: <지금 결정해야 하는 가장 중요한 불확실성>
추천 답안: <근거가 있으면 기본값 제시>
질문: <하나의 질문>
```

If useful, offer 2-3 concrete options and allow free-form input. After each answer, update the decided facts briefly and ask another question only if a high-risk unknown remains.

## Mode Guide

| Mode | Use for | Highest-risk unknowns |
|---|---|---|
| `experiment-design/server-run` | Server experiment prompts, config/run queues, GPU planning | evidence layer, protocol, config path, seed set, server/GPU, output path, completion check |
| `result-import` | Copying/registering server outputs | run ID, server path, local destination, manifest status, copied file set, checkpoint policy |
| `result-interpretation` | Reading metrics, comparing runs, explaining outcomes | manifest path, exact metrics files, seed/run set, metric direction, confirmed vs interpretation boundary |
| `paper-claim/source` | Paper claims, literature/source use, evidence boundaries | claim status, evidence source, unsupported leap, local experiment requirement |
| `reporting/task` | Reports, daily logs, durable task notes, handoffs | audience, source files read, desired artifact, scope, completion criteria |

## Project Guardrails

- Preserve the boundary between NeurIPS-confirmed results, ICLR hypotheses, external-paper evidence, and new local experiment results.
- Do not call a metric confirmed without a readable result file and a manifest or explicit provenance.
- Do not choose optimizer or detector hyperparameters using OOD AUROC, AUPR, or FPR95.
- Do not turn a smoke run into paper-level evidence.
- Do not launch or imply expensive local training unless the user explicitly asked for it.
- For server work, require recoverability from server, branch/commit or snapshot, config, command, run ID, output path, and expected files.

## Stop Conditions

Stop interviewing when these are clear enough for the next skill or action:

- goal and audience
- included scope and excluded scope
- key constraints and evidence boundaries
- required files, run IDs, configs, manifests, or sources
- completion and verification criteria
- remaining open questions, if any

Then summarize only:

```text
결정사항:
- ...

남은 열린 질문:
- ...

다음에 사용할 기존 스킬:
- ...

검증/완료 기준:
- ...
```

## Common Mistakes

- Asking where a file, config, manifest, or metric is before searching the repo.
- Asking several questions at once because the request feels broad.
- Continuing to interview after the next action is already decision-ready.
- Treating external literature as direct evidence for this repository's OOD detector behavior.
- Interpreting result files before checking manifest/provenance.
