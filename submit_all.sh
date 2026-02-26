#!/usr/bin/env bash
set -euo pipefail

# Submit generated SLURM scripts in one shot.
# IMPORTANT: run this script directly from a login/interactive shell:
#   ./submit_all.sh [...]
# Do NOT do: sbatch submit_all.sh
#
# Usage:
#   ./submit_all.sh [--account <acct>] [--qos <qos>] [submit_*.slurm ...]

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n "1,20p" "$0"
  exit 0
fi


if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  echo "You must run this on a SLURM login/submit node (e.g. S3DF), not a local shell." >&2
  echo "Try: ssh <your_user>@s3dflogin.slac.stanford.edu" >&2
  exit 127
fi

account=""
qos=""
scripts=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --account)
      account="$2"; shift 2;;
    --qos)
      qos="$2"; shift 2;;
    -h|--help)
      sed -n "1,16p" "$0"; exit 0;;
    *)
      scripts+=("$1"); shift;;
  esac
done

if [ "${#scripts[@]}" -eq 0 ]; then
  mapfile -t scripts < <(ls -1 submit_*bands_10k.slurm submit_2015_10k.slurm 2>/dev/null | sort -u)
fi

if [ "${#scripts[@]}" -eq 0 ]; then
  echo "No SLURM scripts found. Generate them first with hps-gpr slurm-gen." >&2
  exit 1
fi

for f in "${scripts[@]}"; do
  if [ -f "$f" ]; then
    echo "Submitting $f"
    sbatch_args=()
    [ -n "$account" ] && sbatch_args+=("--account=$account")
    [ -n "$qos" ] && sbatch_args+=("--qos=$qos")
    sbatch "${sbatch_args[@]}" "$f"
  else
    echo "Skipping missing file: $f" >&2
  fi
done
