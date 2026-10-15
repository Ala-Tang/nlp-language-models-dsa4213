"""Cross-platform sequential runner for all assignment parts."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_part(script: str, part: str, quick: bool) -> None:
    output_dir = Path("results") / part / ("quick" if quick else "")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", script]
    if quick:
        command.append("--quick")
    print(f"\n===== Running {script} =====")
    with (output_dir / "log.txt").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run tiny one-epoch smoke tests")
    args = parser.parse_args()
    for script, part in (("part1.py", "part1"), ("part2.py", "part2"), ("part3.py", "part3")):
        run_part(script, part, args.quick)
    print("\nAll parts completed successfully.")


if __name__ == "__main__":
    main()
