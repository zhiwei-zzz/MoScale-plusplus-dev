"""Upload trained SkelVQ checkpoints to a HuggingFace model repo.

Defaults to uploading only the files actually needed downstream (the best-FID
checkpoint, the run's opt.txt, and the published 20-rep eval log) — skips
`latest.tar`, animation dumps, tensorboard logs, etc., which are bulky and
not load-bearing for stage-2 work on the cluster.

Examples:
    # one-time setup
    pip install huggingface_hub
    huggingface-cli login          # token from huggingface.co/settings/tokens

    # upload only skelvq_fsq (the tokenizer the AR is built around)
    python scripts/upload_checkpoints.py \\
        --repo-id zhiwei-z/skelvq-checkpoints \\
        --runs skelvq_fsq

    # upload all three runs that produced our published comparison
    python scripts/upload_checkpoints.py \\
        --repo-id zhiwei-z/skelvq-checkpoints \\
        --runs vae_repro skelvq_bsq skelvq_fsq
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_ALLOW_PATTERNS = [
    "model/net_best_fid.tar",
    "opt.txt",
    "eval/eval.log",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True,
                        help="HF repo id, e.g. zhiwei-z/skelvq-checkpoints")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run names under checkpoints/<dataset>/, e.g. skelvq_fsq vae_repro")
    parser.add_argument("--dataset", default="t2m",
                        help="Dataset subdir under checkpoints/ (default: t2m)")
    parser.add_argument("--checkpoints-dir", default="checkpoints",
                        help="Local checkpoints root (default: checkpoints)")
    parser.add_argument("--include", nargs="+", default=DEFAULT_ALLOW_PATTERNS,
                        help=f"Glob patterns to include per run (default: {DEFAULT_ALLOW_PATTERNS})")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Create the repo as private (default: True)")
    parser.add_argument("--public", dest="private", action="store_false",
                        help="Create the repo as public (overrides --private)")
    parser.add_argument("--commit-message", default=None,
                        help="Commit message (default auto-generated)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded without uploading")
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    ckpt_root = Path(args.checkpoints_dir) / args.dataset

    # Validate every requested run exists locally before touching the network.
    missing = [r for r in args.runs if not (ckpt_root / r).is_dir()]
    if missing:
        print(f"[error] runs not found under {ckpt_root}: {missing}", file=sys.stderr)
        sys.exit(2)

    # Validate at least one allow_pattern matches per run (avoid silent empty uploads).
    matched_files = {}
    for run in args.runs:
        run_dir = ckpt_root / run
        run_files = []
        for pat in args.include:
            run_files.extend(sorted(run_dir.glob(pat)))
        if not run_files:
            print(f"[error] run '{run}': no files matched include patterns {args.include}", file=sys.stderr)
            print(f"        searched in: {run_dir}", file=sys.stderr)
            sys.exit(2)
        matched_files[run] = run_files

    # Print plan.
    print(f"Repo:    {args.repo_id}  ({'private' if args.private else 'public'})")
    print(f"Dataset: {args.dataset}")
    print(f"Local:   {ckpt_root}")
    print(f"Patterns: {args.include}")
    print()
    total_bytes = 0
    for run, files in matched_files.items():
        print(f"  {run}/")
        for f in files:
            size = f.stat().st_size
            total_bytes += size
            rel = f.relative_to(ckpt_root / run)
            print(f"    {rel}   ({size / 1e6:.1f} MB)")
    print(f"  total: {total_bytes / 1e6:.1f} MB")

    if args.dry_run:
        print("\n[dry-run] no upload performed")
        return

    # Create / verify repo.
    try:
        api.repo_info(args.repo_id, repo_type="model")
        print(f"\nrepo {args.repo_id} exists; uploading new commits")
    except RepositoryNotFoundError:
        print(f"\ncreating repo {args.repo_id} (private={args.private})")
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Upload each run as its own subdirectory in the repo.
    msg = args.commit_message or f"upload {', '.join(args.runs)}"
    for run, files in matched_files.items():
        run_dir = ckpt_root / run
        print(f"\nuploading {run} -> {args.repo_id}:/{run}/")
        api.upload_folder(
            folder_path=str(run_dir),
            path_in_repo=run,
            repo_id=args.repo_id,
            repo_type="model",
            allow_patterns=args.include,
            commit_message=f"{msg} ({run})",
        )
        for f in files:
            rel = f.relative_to(run_dir)
            print(f"    + {rel}")

    print(f"\n[done] view at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
