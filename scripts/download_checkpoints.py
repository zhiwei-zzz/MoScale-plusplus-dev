"""Download checkpoints from a HuggingFace model repo into the local
checkpoints/<dataset>/<run>/ tree expected by SkelVQ / MoScale code paths.

Examples:
    pip install huggingface_hub
    huggingface-cli login          # token from huggingface.co/settings/tokens

    # download just the run we need to feed SkelVQWrapper:
    python scripts/download_checkpoints.py \\
        --repo-id zhiwei-z/skelvq-checkpoints \\
        --runs skelvq_fsq

    # one specific file (returns its cached path on stdout):
    python scripts/download_checkpoints.py \\
        --repo-id zhiwei-z/skelvq-checkpoints \\
        --file skelvq_fsq/model/net_best_fid.tar
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True,
                        help="HF repo id, e.g. zhiwei-z/skelvq-checkpoints")
    parser.add_argument("--runs", nargs="*", default=[],
                        help="Run names to materialize under checkpoints/<dataset>/.")
    parser.add_argument("--file", default=None,
                        help="Single file to fetch (e.g. skelvq_fsq/model/net_best_fid.tar). "
                             "Prints the cached path; does not copy into checkpoints/.")
    parser.add_argument("--dataset", default="t2m",
                        help="Local checkpoints subdir (default: t2m)")
    parser.add_argument("--checkpoints-dir", default="checkpoints",
                        help="Local checkpoints root (default: checkpoints)")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink files from HF cache instead of copying (saves disk).")
    args = parser.parse_args()

    if not args.runs and not args.file:
        print("[error] must specify --runs or --file", file=sys.stderr)
        sys.exit(2)

    from huggingface_hub import hf_hub_download, snapshot_download

    if args.file:
        path = hf_hub_download(repo_id=args.repo_id, filename=args.file, repo_type="model")
        print(path)
        return

    ckpt_root = Path(args.checkpoints_dir) / args.dataset
    ckpt_root.mkdir(parents=True, exist_ok=True)

    for run in args.runs:
        print(f"\n[run] {run}")
        # snapshot_download fetches the repo (or a subset via allow_patterns) into the
        # HF cache and returns the local cache path of the repo root.
        local_cache = snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            allow_patterns=[f"{run}/*"],
        )
        src_run = Path(local_cache) / run
        if not src_run.exists():
            print(f"  [warn] {run} not present in repo {args.repo_id}", file=sys.stderr)
            continue

        dst_run = ckpt_root / run
        dst_run.mkdir(parents=True, exist_ok=True)

        for src_file in src_run.rglob("*"):
            if src_file.is_dir():
                continue
            dst_file = dst_run / src_file.relative_to(src_run)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            if args.symlink:
                # Resolve symlinks because HF cache uses dereferenced blobs.
                dst_file.symlink_to(src_file.resolve())
                print(f"    {dst_file}  ->  {src_file.resolve()}")
            else:
                shutil.copy2(src_file, dst_file)
                print(f"    {dst_file}  ({src_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
