"""Deprecated shim — the trainer lives at ``oracle.scripts.train_head`` now.

Forwards ``python -m oracle.scripts.videomae.train_head ...`` calls to the
generic trainer with ``--head-arch videomae`` injected. Prefer the new path
directly: ``python -m oracle.scripts.train_head --head-arch videomae ...``.
"""
from __future__ import annotations

import sys
import warnings

from oracle.scripts.train_head import main as _generic_main


def main(argv=None) -> int:
    warnings.warn(
        "oracle.scripts.videomae.train_head has moved to oracle.scripts.train_head; "
        "invoke the new path directly and pass --head-arch videomae.",
        DeprecationWarning,
        stacklevel=2,
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--head-arch" not in argv:
        argv = ["--head-arch", "videomae", *argv]
    return _generic_main(argv)


if __name__ == "__main__":
    sys.exit(main())
