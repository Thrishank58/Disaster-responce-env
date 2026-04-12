#!/usr/bin/env python3
"""
Validate stdout from inference.py against OpenEnv hackathon log rules.

Usage:
  python inference.py ... > run.log 2>run.err
  python scripts/check_inference_log_format.py run.log

Or pipe (stderr discarded so only protocol lines go to stdin):
  python inference.py ... 2>nul | python scripts/check_inference_log_format.py -
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

START_RE = re.compile(
    r"^\[START\] task=(?P<task>\S+) env=(?P<env>\S+) model=(?P<model>\S+)$"
)
# One or two spaces after [STEP] (spec template vs example differ slightly)
STEP_RE = re.compile(
    r"^\[STEP\]\s+step=(?P<step>\d+) action=(?P<action>.+) "
    r"reward=(?P<reward>-?\d+\.\d{2}) done=(?P<done>true|false) error=(?P<error>null|.*)$"
)
END_RE = re.compile(
    r"^\[END\] success=(?P<success>true|false) steps=(?P<steps>\d+) "
    r"score=(?P<score>\d+\.\d{2}) rewards=(?P<rewards>.*)$"
)
REWARD_TOKEN = re.compile(r"^-?\d+\.\d{2}$")


def validate_lines(lines: list[str]) -> list[str]:
    errors: list[str] = []
    i = 0
    n = len(lines)
    episode = 0
    expect = "start"  # "start" | "step_or_end"

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if expect == "start":
            if not START_RE.match(line):
                errors.append(f"line {i+1}: expected [START], got: {line[:120]!r}")
                i += 1
                continue
            episode += 1
            expect = "step_or_end"
            i += 1
            continue

        # step_or_end
        if line.startswith("[START]"):
            errors.append(f"line {i+1}: [START] before [END] closed episode {episode}")
            expect = "start"
            continue

        if line.startswith("[END]"):
            em = END_RE.match(line)
            if not em:
                errors.append(f"line {i+1}: bad [END]: {line[:120]!r}")
            else:
                sc = em.group("score")
                if len(sc.split(".", 1)[1]) != 2:
                    errors.append(f"line {i+1}: score {sc!r} must have 2 decimal places")
                rewards_part = em.group("rewards").strip()
                if rewards_part:
                    for j, r in enumerate(rewards_part.split(",")):
                        tok = r.strip()
                        if not REWARD_TOKEN.match(tok):
                            errors.append(
                                f"line {i+1}: rewards part {j+1} {tok!r} should match NN.dd"
                            )
            expect = "start"
            i += 1
            continue

        sm = STEP_RE.match(line)
        if not sm:
            errors.append(f"line {i+1}: expected [STEP] or [END], got: {line[:120]!r}")
            i += 1
            continue

        rd = sm.group("reward")
        frac = rd.split(".", 1)[1]
        if len(frac) != 2:
            errors.append(f"line {i+1}: reward {rd!r} must have exactly 2 decimal places")
        i += 1

    if expect != "start":
        errors.append(f"episode {episode}: file ended before [END]")

    return errors


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "-"
    if path == "-":
        text = sys.stdin.read()
    else:
        text = Path(path).read_text(encoding="utf-8", errors="replace")

    lines = [ln.lstrip("\ufeff") for ln in text.splitlines()]
    errs = validate_lines(lines)
    if errs:
        print("FAIL:", len(errs), "issue(s)", file=sys.stderr)
        for e in errs[:50]:
            print(e, file=sys.stderr)
        if len(errs) > 50:
            print("...", file=sys.stderr)
        return 1
    print("OK: log format checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
