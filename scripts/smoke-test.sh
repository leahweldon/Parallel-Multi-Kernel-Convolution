#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

for config in     "16 16 1 32 32"     "16 16 3 32 32"     "16 16 7 32 32"     "16 32 3 64 32"; do
    echo "Testing configuration: ${config}"
    output="$(./conv-harness ${config})"
    printf '%s\n' "$output"
    if grep -q 'WARNING:' <<< "$output"; then
        echo 'Reference-output comparison reported a warning.' >&2
        exit 1
    fi
    grep -q 'within acceptable range' <<< "$output" || {
        echo 'Expected reference-output check was not reported.' >&2
        exit 1
    }
done

echo 'All small-configuration smoke checks passed.'
