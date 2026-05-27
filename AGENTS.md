# ARM-demo Agent Operating Guide

This repository is worked on directly from the current Codex workspace unless the user explicitly requests a different machine or workflow.

## User Context

- User is a BYU ML undergraduate, aiming for robotics/AI PhD.
- User prefers step-by-step, intuitive explanations tied to runnable code.
- Current priority is a stable, runnable world model demo rather than SOTA.

## Non-Negotiable Execution Rules

1. Do not launch expensive experiment workloads unless the user has requested or approved them.
2. Before any run command, state:
   - where the command runs,
   - purpose,
   - expected duration,
   - timeout/cancel policy.
3. For long-running commands:
   - set explicit timeout,
   - for training/experiment phases, use relaxed timeout budgets (do not use aggressive short limits),
   - if command exceeds threshold or stalls, stop safely and report.
4. Provide frequent progress updates during execution.
5. Avoid destructive operations and never touch unrelated files.

## Workflow

1. Edit code in this repository.
2. Run lightweight checks locally in the current workspace when appropriate.
3. Use remote machines only when the user explicitly asks for remote execution or when the task clearly requires it.
4. Save relevant artifacts such as plots, videos, and logs in the project when they are part of the requested work.

## Logging Policy

When actions go beyond code edits (environment changes, system-level operations, server-side setup), record them in `changes-log.txt` with timestamp and concise details.

## World Model Project Goal

Build an action-conditioned world model:

- Input: `frame_t`, `action_t`
- Output: predicted `frame_{t+1}`

Core modules:

1. Encoder: image -> latent
2. Dynamics: latent + action -> next latent
3. Decoder: latent -> image
