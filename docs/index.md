# Sky

Sky is a robot built on the [Lumen](https://github.com/fitzyracing1/lumen) gate and a tiny NumPy transformer decoder ("nano LLM"). Behaviour is layered, highest first, and a higher layer inhibits the ones below it (from `robot.py`):

| Layer | Rule |
|---|---|
| air | stay inside policy, and withhold if confidence is low |
| eat | keep a context budget, and stop when resources run out |
| win | finish the mission the operator handed in |
| talk | speak only a continuation the critic did not kill |

The weights (`nano_llm.npz`) are never updated by the kernel. The revision budget is 2.

## Getting started

Requires Python 3 and NumPy.

```bash
python3 robot.py [mission words...]                    # writes sky_trace.json
python3 lumen_llm.py                                   # Lumen-gated decoding, writes lumen_run.json
python3 nano_llm.py sample --prompt "Sky " --n 120     # options: --temperature 0.8 --seed 1
python3 nano_llm.py train --steps 600                  # retrain on corpus.txt (default --steps 400, --lr 3e-3)
```

## Files

| File | Role |
|---|---|
| `robot.py` | layered robot (air, eat, win, talk) |
| `lumen_llm.py` | `LumenGate`: the decoder proposes continuations, Lumen attacks and simulates them, then commits or withholds |
| `nano_llm.py` | tiny transformer: train and sample CLI, `NanoLM` |
| `corpus.txt`, `nano_llm.npz` | training text and checkpoint |
| `sky_trace.json`, `lumen_run.json` | sample outputs |

The repo has no tests, no CI and no LICENSE file.

Copyright (c) 2026 Joshua Almeida
