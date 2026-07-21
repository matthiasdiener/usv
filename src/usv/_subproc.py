"""Fresh-process worker: run a single asv-style benchmark in isolation.

Spawned by :func:`usv.run_benchmarks` when ``fresh_process=True``.  It receives
one JSON argument describing the benchmark to run, executes exactly one
``(method, param-combo)`` on a fresh class instance, and prints the resulting
samples as JSON on stdout::

    python -m usv._subproc '{"target": "mod:Cls", "method": "time_x",
                             "combo_index": 0, "kwargs": {"timer": "wall"}}'

Running each benchmark in its own process is asv's default methodology: it
prevents warmup state, allocator fragmentation, and module-level caches from
leaking between benchmarks.
"""

from __future__ import annotations

import json
import sys


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m usv._subproc <json-payload>", file=sys.stderr)
        return 2

    from usv.asv import run_one_by_index

    spec = json.loads(argv[0])
    m = run_one_by_index(
        spec["target"], spec["method"], spec["combo_index"], spec.get("kwargs", {})
    )
    json.dump(
        {
            "name": m.name,
            "samples": m.samples.tolist(),
            "inner": m.inner,
            "flops": m.flops,
            "bytes": m.bytes,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
