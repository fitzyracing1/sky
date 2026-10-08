# Sky

A robot built on the Lumen gate and the nano decoder.

Sky does not sample and hope. Each job climbs a stack. Air inhibits speech when confidence is low. Eat stops the body when the context budget is gone. Win asks the decoder for three plans and lets Lumen kill them. Talk speaks only a continuation that survived.

```bash
python3 robot.py
python3 nano_llm.py sample --prompt "Sky " --n 120
```

Weights are `nano_llm.npz`, trained by `python3 nano_llm.py train --steps 600`. The kernel never edits them. Revision budget is 2.

Remote: https://github.com/fitzyracing1/sky
