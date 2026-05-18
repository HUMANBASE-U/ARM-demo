# ARM-demo Agent Operating Guide

This repository follows a strict split workflow:

- Local Windows machine: code editing, Git commit/push/pull, docs updates.
- Lab server: all experiment commands (data collection, training, evaluation, video generation).

## User Context

- User is a BYU ML undergraduate, aiming for robotics/AI PhD.
- User prefers step-by-step, intuitive explanations tied to runnable code.
- Current priority is a stable, runnable world model demo rather than SOTA.

## Non-Negotiable Execution Rules

1. Never run experiment workloads on local Windows by default.
2. Before any run command, state:
   - where the command runs (server/local),
   - purpose,
   - expected duration,
   - timeout/cancel policy.
3. For long-running commands:
   - set explicit timeout,
   - for remote training/experiment phases, use relaxed timeout budgets (do not use aggressive short limits),
   - if command exceeds threshold or stalls, stop safely and report.
4. Provide frequent progress updates during execution.
5. Avoid destructive operations and never touch unrelated files.
6. If server password is missing/forgotten, check `C:\Users\yaoda\Desktop\HOW-TO-USE-LAB-MACHINE.txt` first.

## Remote-First Workflow

1. Edit code locally in this repository.
2. Push changes to GitHub.
3. SSH to lab server and pull latest code.
4. Run experiments on server only.
5. Save artifacts (plots/videos/logs), commit, and sync back.

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
