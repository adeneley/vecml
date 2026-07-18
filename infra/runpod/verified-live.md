# Live-verified API facts (18 Jul 2026, real key)

Supersedes the UNVERIFIED tags in README.md where they overlap.

## Auth
- REST `rest.runpod.io/v1` Bearer auth works: `GET /pods` and
  `GET /networkvolumes` return 200 (both empty at time of writing).
- The key also has GraphQL access (`api.runpod.io/graphql`).
- Key lives in `infra/runpod/.env` (gitignored, chmod 600). Never commit it.

## Exact `gpuTypeIds` for POST /pods (from `gpuTypes` query)
| id (use verbatim)              | card       | community $/hr | secure $/hr |
|--------------------------------|------------|----------------|-------------|
| `NVIDIA GeForce RTX 5090`      | RTX 5090   | 0.69           | 0.99        |
| `NVIDIA RTX A5000`             | RTX A5000  | 0.16           | 0.27        |
| `NVIDIA A100 80GB PCIe`        | A100 PCIe  | 1.19           | 1.39        |
| `NVIDIA H100 NVL`              | H100 NVL   | 2.59           | 3.19        |

Community (interruptible-friendly) prices are notably below the secure
prices the README's examples used; prefer community + checkpointing per
the plan's spot rule.

## Still unverified
- CPU pod per-vCPU rate (console-only; correct on first paid run).
- `POST /pods/{id}/stop` endpoint path.
