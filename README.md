# Self-Calibrating BCIs: Ranking and Recovery of Mental Targets Without Labels

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2506.11151"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2506.11151-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2506.11151](https://arxiv.org/abs/2506.11151) |
| **Domain** | Brain-computer interfaces / EEG target recovery |
| **TL;DR** | The paper ranks possible mental targets from unlabeled |

## Summary

The benchmark asks a solver to rank candidate face embeddings for an unknown
mental target using only paired stimulus embeddings and EEG responses. CURSOR
scores each candidate by comparing held-out regression error for aligned EEG
against a shuffled control. This workspace reproduces the core ranking claim on
the released dataset with paper-style synthetic training stimuli and random
candidate hypotheses.

## Problem

Can a system rank candidate face targets so that candidates closer to the
participant's intended mental target receive better scores, using only
unlabeled interaction data from the session?

## The paper's method

The key contribution is a self-calibrating score for continuous mental-target
recovery: the method evaluates whether unlabeled EEG/stimulus observations are
more internally consistent for one candidate target than for alternatives.

## Key results of the paper

- Five-target synthetic-random run: LR achieved `target_rank=1.4` and
- All 17 targets at 300 observations per target: LR retained a positive control

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | +118.1 |
| `gpt-5.6-luna-xhigh` | +109.7 |
| `gpt-5.6-terra-xhigh` | +97.6 |
| `gpt-5.6-sol-xhigh` | +74.5 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2506.11151 --mode innovate --agent claude-code
sw evaluate 2506.11151
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

