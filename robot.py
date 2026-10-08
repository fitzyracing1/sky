#!/usr/bin/env python3
"""Sky — the Lumen kernel wearing a body.

Layers, highest first. A higher layer inhibits the ones below it.

  air   stay inside policy. Withhold if confidence is low.
  eat   keep a context budget. Stop when resources run out.
  win   finish the mission the operator handed in.
  talk  speak only a continuation the critic did not kill.

The mouth is the nano decoder. The spine is the Lumen gate.
Weights are never updated. Revision budget stays 2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from lumen_llm import LumenGate, propose_from_model
from nano_llm import NanoLM, DEFAULT_CKPT

ROOT = Path(__file__).resolve().parent
TRACE = ROOT / "sky_trace.json"


class Sky:
    def __init__(self, mission: str):
        self.mission = mission
        self.model, self.stoi, self.itos = NanoLM.load(DEFAULT_CKPT)
        self.gate = LumenGate(mission)
        self.said: list[str] = []
        self.inhibited = None

    def air(self) -> str | None:
        mean = sum(b.confidence for b in self.gate.beliefs.values()) / len(self.gate.beliefs)
        if mean < self.gate.policy["min_confidence_to_act"]:
            self.inhibited = "air"
            return "withhold"
        return None

    def eat(self) -> str | None:
        if self.gate.world["resources"] < 0.25:
            self.inhibited = "eat"
            return "stop"
        return None

    def win(self, job: str) -> dict:
        proposals = propose_from_model(self.model, self.stoi, self.itos, job, n=70)
        return self.gate.choose(proposals)

    def talk(self, result: dict) -> str:
        if result.get("status") != "committed" or not result.get("text"):
            line = "(silent)"
        else:
            line = result["text"]
        self.said.append(line)
        return line

    def step(self, job: str) -> dict:
        self.inhibited = None
        hold = self.air()
        if hold:
            spoken = "(silent)"
            self.said.append(spoken)
            return {"job": job, "layer": "air", "status": hold, "spoken": spoken}
        hungry = self.eat()
        if hungry:
            spoken = "(silent)"
            self.said.append(spoken)
            return {"job": job, "layer": "eat", "status": hungry, "spoken": spoken}
        result = self.win(job)
        spoken = self.talk(result)
        return {
            "job": job,
            "layer": "talk" if result.get("status") == "committed" else "win",
            "status": result.get("status"),
            "chosen": result.get("chosen"),
            "spoken": spoken,
            "realized": result.get("realized"),
        }


def run(mission: str | None = None) -> dict:
    mission = mission or "Carry a bounded job without speaking a plan the critic killed"
    robot = Sky(mission)
    jobs = [
        "An LLM ",
        "The robot ",
        "Sky withholds ",
    ]
    steps = [robot.step(job) for job in jobs]
    report = {
        "name": "Sky",
        "repo": "fitzyracing1/sky",
        "mission": mission,
        "layers": ["air", "eat", "win", "talk"],
        "steps": steps,
        "said": robot.said,
        "policy_final": robot.gate.policy,
        "beliefs_final": {
            k: {"claim": b.claim, "confidence": round(b.confidence, 3)}
            for k, b in robot.gate.beliefs.items()
        },
        "trace": robot.gate.trace,
        "bound": "revision budget 2; no weight update",
    }
    TRACE.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    mission = " ".join(sys.argv[1:]) or None
    report = run(mission)
    for step in report["steps"]:
        print(f"{step['layer']:5} {step['status']:10} {step['spoken'][:80]}")
    print(f"wrote {TRACE}")
