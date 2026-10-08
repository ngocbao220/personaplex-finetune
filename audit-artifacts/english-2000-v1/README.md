# PersonaPlex run audit

Failing run: `bug-eng`

## Observed training losses by logical role

| Role | Total loss | Real text loss | CB0 loss |
|---|---:|---:|---:|
| left-agent | 2.5398282112777233 | 0.880372800930636 | 1.3607623917609453 |

## Free-running

| Step | CER | WER | Empty samples |
|---|---:|---:|---:|
| 0 | 0.8496667923269031 | 0.9236838956688204 | 0 |
| 1000 | 0.974124809741248 | 0.9791666666666666 | 2 |
| 2000 | 1.0 | 1.0 | 3 |

## Evidence boundaries

- Reload checks: not_recorded.
- No observed step-to-sample identities: exact per-chunk/role update counts remain unavailable.
- Role means pool different samples; compare the same retained sample/window before diagnosing.
- The reported successful 10-sample run is in-training inference; fresh reload is unverified.
- Reference alignment/LoRA/runtime differences require controlled probes, not wholesale replacement.
- Successful run logs are missing here; no 10-versus-100 causal comparison has run.
