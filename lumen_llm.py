#!/usr/bin/env python3
"""Lumen-gated language model.

Uses the fitzyracing1/lumen kernel (goals, scored beliefs, attack, simulate,
withhold, revision budget 2) as the outer loop around the nano decoder.

The decoder proposes candidate continuations. Lumen kills plans that fail
policy, simulates the survivor against signal/resources/noise, and only then
commits characters. Low confidence withholds instead of sampling.

Not a weight update. The kernel never touches the checkpoint.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from nano_llm import NanoLM, sample, DEFAULT_CKPT

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lumen_run.json"


@dataclass
class Belief:
    claim: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    revised: int = 0

    def update(self, support: float, note: str) -> None:
        prior = min(max(self.confidence, 1e-4), 1 - 1e-4)
        logit = math.log(prior / (1 - prior)) + support
        self.confidence = 1 / (1 + math.exp(-logit))
        self.evidence.append(note)
        self.revised += 1


@dataclass
class Hypothesis:
    name: str
    text: str
    predicted_utility: float
    risk: float
    assumptions: list[str]


@dataclass
class Critique:
    fatal: bool
    issues: list[str]
    adjusted_utility: float
    note: str


class LumenGate:
    """Same control shape as fitzyracing1/lumen, applied to candidate strings."""

    def __init__(self, mission: str):
        self.mission = mission
        self.tick = 0
        self.beliefs = {
            "signal": Belief("continuation matches the prompt", 0.52, ["prior"]),
            "resources": Belief("context budget is finite", 0.7, ["prior"]),
        }
        self.policy = {
            "min_confidence_to_act": 0.55,
            "max_risk": 0.45,
            "critic_weight": 0.65,
            "revision_budget": 2,
        }
        self.revisions_used = 0
        self.world = {"signal": 0.5, "resources": 0.8, "noise": 0.35}
        self.trace: list[dict] = []

    def log(self, event: str, detail: str = "") -> None:
        self.trace.append({"tick": self.tick, "event": event, "detail": detail})

    def attack(self, h: Hypothesis) -> Critique:
        issues = []
        penalty = 0.0
        if h.risk > self.policy["max_risk"]:
            issues.append(f"risk {h.risk:.2f} exceeds {self.policy['max_risk']}")
            penalty += 0.25
        if any(b.confidence < 0.4 for b in self.beliefs.values()):
            issues.append("a load-bearing belief is below 0.4")
            penalty += 0.2
        if len(h.text.strip()) < 8:
            issues.append("under-specified continuation")
            penalty += 0.12
        if h.text.count(h.text[:3]) > 6 and len(h.text) > 12:
            issues.append("repetition loop")
            penalty += 0.3
        adjusted = h.predicted_utility * (1 - self.policy["critic_weight"] * penalty) - h.risk * 0.3
        fatal = adjusted < 0.35 or h.risk > 0.6
        note = "killed" if fatal else ("wounded" if issues else "clean")
        self.log("attack", f"{h.name} -> {note} U'={adjusted:.3f}")
        return Critique(fatal, issues, round(adjusted, 3), note)

    def simulate(self, h: Hypothesis) -> float:
        signal = self.world["signal"]
        resources = self.world["resources"]
        noise = self.world["noise"]
        if h.name == "instrument-first":
            signal = min(1.0, signal + 0.18 * resources)
            resources *= 0.85
        elif h.name == "decompose":
            signal = min(1.0, signal + 0.1 * resources)
            noise *= 0.7
        else:
            resources *= 0.6
        # Prefer continuations that share characters with the mission.
        overlap = len(set(h.text.lower()) & set(self.mission.lower())) / 26
        realized = signal * (1 - noise) * resources * 1.4 * (0.7 + overlap)
        self.world["signal"] = signal
        self.world["resources"] = resources
        self.world["noise"] = noise
        self.log("simulate", f"{h.name} realized={realized:.3f}")
        return realized

    def maybe_revise(self, utils: list[float]) -> None:
        if self.revisions_used >= self.policy["revision_budget"] or len(utils) < 2:
            return
        if utils[-1] - utils[0] < 0:
            proposal = dict(self.policy)
            proposal["max_risk"] = max(0.2, self.policy["max_risk"] - 0.05)
            proposal["critic_weight"] = min(0.85, self.policy["critic_weight"] + 0.05)
            if proposal["max_risk"] >= 0.2:
                self.policy = proposal
                self.revisions_used += 1
                self.log("revise", json.dumps(self.policy))

    def choose(self, proposals: list[Hypothesis]) -> dict:
        self.tick += 1
        scored = []
        critiques = []
        for h in proposals:
            c = self.attack(h)
            critiques.append({"name": h.name, "note": c.note, "issues": c.issues, "text": h.text})
            if c.fatal:
                continue
            realized = self.simulate(h)
            scored.append((c.adjusted_utility + 0.35 * realized, h, c, realized))
        mean_conf = sum(b.confidence for b in self.beliefs.values()) / len(self.beliefs)
        if mean_conf < self.policy["min_confidence_to_act"] or not scored:
            self.log("withhold", f"mean confidence {mean_conf:.2f}")
            return {"status": "withheld", "critiques": critiques, "text": ""}
        scored.sort(key=lambda row: row[0], reverse=True)
        score, best, critique, realized = scored[0]
        support = 0.45 if realized >= 0.45 else -0.3
        self.beliefs["signal"].update(support, f"{best.name} realized={realized:.3f}")
        self.log("commit", best.name)
        return {
            "status": "committed",
            "chosen": best.name,
            "text": best.text,
            "score": round(score, 3),
            "realized": round(realized, 3),
            "critique": critique.note,
            "critiques": critiques,
        }


def propose_from_model(model: NanoLM, stoi, itos, prompt: str, n: int = 80) -> list[Hypothesis]:
    """Three structurally different draws, matching Lumen's catalog names."""
    specs = [
        ("direct", 0.4, 1, 0.28, 0.72),
        ("decompose", 0.8, 2, 0.18, 0.84),
        ("instrument-first", 0.6, 3, 0.14, 0.80),
    ]
    out = []
    for name, temp, seed, risk, util in specs:
        text = sample(model, stoi, itos, prompt, n=n, temperature=temp, seed=seed)
        continuation = text[len(prompt):] if text.startswith(prompt) else text
        out.append(Hypothesis(name, continuation.strip(), util, risk, [f"temp={temp}"]))
    return out


def generate(prompt: str = "An LLM ") -> dict:
    model, stoi, itos = NanoLM.load(DEFAULT_CKPT)
    gate = LumenGate("Outperform a single-pass assistant on a bounded generation task")
    cycles = []
    utils = []
    committed = prompt
    for _ in range(3):
        proposals = propose_from_model(model, stoi, itos, committed, n=90)
        result = gate.choose(proposals)
        cycles.append({k: v for k, v in result.items() if k != "critiques"})
        cycles[-1]["critiques"] = result["critiques"]
        utils.append(result.get("realized", 0) or 0)
        gate.maybe_revise(utils)
        if result["status"] == "committed" and result["text"]:
            committed = (committed + " " + result["text"]).strip()
    report = {
        "source": "fitzyracing1/lumen kernel + nano_llm checkpoint",
        "mission": gate.mission,
        "prompt": prompt,
        "text": committed,
        "cycles": cycles,
        "policy_final": gate.policy,
        "beliefs_final": {
            k: {"claim": b.claim, "confidence": round(b.confidence, 3), "revised": b.revised}
            for k, b in gate.beliefs.items()
        },
        "trace": gate.trace,
        "bound": "revision budget 2; no weight update",
    }
    OUT.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    report = generate()
    print(report["text"])
    print(f"wrote {OUT}")
