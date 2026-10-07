# PersonaPlex run audit

Failing run: `artifact`

## Observed training losses by logical role

| Role | Total loss | Real text loss | CB0 loss |
|---|---:|---:|---:|
| left-agent | 2.0426913930019674 | 1.1262877352325134 | 1.185331152167469 |
| right-agent | 1.992950417116682 | 0.9817109219194438 | 1.3607776161030822 |

## Free-running

| Step | CER | WER | Empty samples |
|---|---:|---:|---:|
| 0 | 0.7888040712468194 | 0.9836065573770492 | 0 |
| 100 | 1.0 | 1.0 | 1 |
| 200 | 1.0 | 1.0 | 1 |
| 300 | 1.0 | 1.0 | 1 |
| 400 | 0.7837150127226463 | 0.8934426229508197 | 0 |
| 500 | 0.712468193384224 | 0.8360655737704918 | 0 |
| 600 | 0.7837150127226463 | 0.8852459016393442 | 0 |
| 700 | 0.989821882951654 | 0.9918032786885246 | 0 |
| 800 | 0.7353689567430025 | 0.8688524590163934 | 0 |
| 900 | 0.9211195928753181 | 0.9672131147540983 | 0 |
| 1000 | 0.8956743002544529 | 0.9754098360655737 | 0 |

## Evidence boundaries

- Reload checks: not_recorded.
- No observed step-to-sample identities: exact per-chunk/role update counts remain unavailable.
- Role means pool different samples; compare the same retained sample/window before diagnosing.
- The reported successful 10-sample run is in-training inference; fresh reload is unverified.
- Reference alignment/LoRA/runtime differences require controlled probes, not wholesale replacement.
- Successful run logs are missing here; no 10-versus-100 causal comparison has run.
