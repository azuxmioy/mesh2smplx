"""Strip chumpy arrays out of SMPL / SMPL-H / SMPL-X ``.pkl`` model files.

The original SMPL-family ``.pkl`` files (e.g. ``SMPL_NEUTRAL.pkl``,
``basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl``) store some fields as
``chumpy`` arrays. Loading them therefore requires ``chumpy`` installed, which
breaks on modern numpy. This script loads such a pickle *without* chumpy,
converts every chumpy array to a plain numpy array, and re-saves it. The
numeric values, dtype and shape are preserved exactly; only the container type
changes, so the result loads with no chumpy dependency and is consumed directly
by the ``smplx`` package.

Usage:
    # one file, overwrite in place
    python scripts/dechumpify_models.py body_models/smpl/SMPL_NEUTRAL.pkl

    # one file, write to a new path
    python scripts/dechumpify_models.py IN.pkl OUT.pkl

    # a whole directory: convert every .pkl under it (in place)
    python scripts/dechumpify_models.py body_models/

Pass --dry-run to report which fields are chumpy without writing anything.
"""
import argparse
import pickle
import sys
import types
from pathlib import Path

import numpy as np


def install_fake_chumpy():
    """Register a minimal fake ``chumpy`` so pickle can reconstruct Ch objects.

    chumpy's ``Ch`` is a plain object (NOT an ndarray subclass). A leaf chumpy
    array keeps its real data in the instance attribute ``x``. We only need an
    object that accepts the pickled ``__setstate__`` payload; we read ``.x``
    out of it afterwards.
    """

    class Ch(object):
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__["_state"] = state

        def __reduce_ex__(self, protocol):
            return (object.__new__, (type(self),), self.__dict__)

    class FakeModule(types.ModuleType):
        """Any attribute access returns the Ch stub (covers ch.ch, reordering, ...)."""

        def __getattr__(self, name):
            return Ch

    for name in ("chumpy", "chumpy.ch", "chumpy.reordering", "chumpy.ch_ops"):
        mod = FakeModule(name)
        mod.__path__ = []
        mod.Ch = Ch
        sys.modules[name] = mod

    return Ch


def to_numpy(value, Ch):
    """Resolve a possibly-chumpy value to a plain numpy array."""
    if isinstance(value, Ch):
        d = value.__dict__
        for key in ("x", "_data", "data"):
            if key in d:
                return np.asarray(to_numpy(d[key], Ch))
        for v in d.values():
            if isinstance(v, np.ndarray):
                return np.asarray(v)
        raise ValueError(f"Could not extract array from chumpy obj; keys={list(d)}")
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    return value


def dechumpify_file(in_path, out_path=None, dry_run=False):
    """Convert a single ``.pkl``. Returns the list of de-chumpified field names."""
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path else in_path
    Ch = install_fake_chumpy()

    with open(in_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict in {in_path}, got {type(data)}")

    converted = []
    for k, v in list(data.items()):
        new_v = to_numpy(v, Ch)
        if new_v is not v:
            converted.append(k)
        data[k] = new_v

    if not dry_run:
        with open(out_path, "wb") as f:
            pickle.dump(data, f, protocol=2)

    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="A .pkl file or a directory to scan for .pkl files")
    parser.add_argument("out", nargs="?", default=None,
                        help="Output path (single-file mode only). Default: overwrite in place")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report chumpy fields without writing anything")
    args = parser.parse_args()

    src = Path(args.path)
    if src.is_dir():
        if args.out:
            parser.error("OUT path is not allowed in directory mode (files are converted in place)")
        pkls = sorted(src.rglob("*.pkl"))
        if not pkls:
            print(f"No .pkl files found under {src}")
            return
        for p in pkls:
            converted = dechumpify_file(p, dry_run=args.dry_run)
            verb = "would convert" if args.dry_run else "converted"
            print(f"{p}: {verb} {converted}")
    else:
        converted = dechumpify_file(src, args.out, dry_run=args.dry_run)
        verb = "would convert" if args.dry_run else "converted"
        dest = args.out or args.path
        print(f"{dest}: {verb} {converted}")


if __name__ == "__main__":
    main()
