# MEGA CLI Reference


The `mega` CLI is the shortest path for interactive Hub work and shell automation. Its command layout follows the same domain boundaries as the Web API and `MegaHubClient`.

## Install

Install the published Python package with one of these tools:

```bash
uv tool install megatensors
pipx install megatensors
python -m pip install megatensors
```

Confirm the executable and environment:

```bash
mega version
mega env
mega --help
```

Update an existing installation:

```bash
mega update
```

## Authenticate

```bash
mega auth login
mega auth whoami
```

For automation, inject `MEGA_TOKEN` or pass `--token` to an individual command. See [Authentication](/docs/hub/authentication) for device flow, stored tokens, scopes, and endpoint selection.

Browser login requests the scopes needed by the current CLI command surface:
`repo:read`, `repo:write`, `repo:delete`, `community:write`, `jobs:run`,
`inference:run`, `account:keys`, and `webhooks:manage`, plus the identity and
refresh-token scopes. This CLI authorization is independent of the MCP
Read/Write/Full preference. `mega auth whoami` shows the active token kind,
role, and granted scopes so scripts can fail before attempting an unauthorized
operation.

## Command domains

| Domain | Commands | Purpose |
| --- | --- | --- |
| Authentication | `mega auth` | Login, logout, switch tokens, inspect identity, and manage public keys. |
| Repositories | `mega repos`, `models`, `datasets`, `spaces` | Create, inspect, update, move, duplicate, and delete Hub repositories. |
| Files | `mega upload`, `upload-large-folder`, `download`, `snapshot`, `cp` | Transfer selected files or complete repository trees. |
| Revisions | `mega repos branch`, `tag`, `history`, `commit` | Manage branches and tags and inspect immutable commit history. |
| Community | `mega discussions` | Operate discussions, pull requests, comments, reactions, and merge state. |
| Compute | `mega jobs`, `mega sandbox` | Run bounded containers, recurring schedules, and interactive isolated sessions. |
| Buckets | `mega buckets`, `mega sync` | Manage mutable object storage and synchronize local directory trees. |
| Extensibility | `mega extensions`, `mega skills` | Install trusted CLI extensions and AI-assistant skills. |
| Webhooks | `mega webhooks` | Configure signed event delivery and inspect receipts. |
| Local cache | `mega cache` | Inspect, verify, prune, and remove cached Hub artifacts. |
| Artifact runtime | `mega convert`, `mega sign` | Convert Hugging Face folders and sign MEGA artifacts. |

## Repository workflows

```bash
mega repos create mega/demo --type model --exist-ok
mega upload mega/demo ./model --revision main
mega repos tag create mega/demo v1.0 --message "First release"
mega snapshot mega/demo --revision v1.0 --local-dir ./release
```

Typed repository aliases expose the common operations without repeating `--type`:

```bash
mega models list
mega datasets upload research/evals ./data
mega spaces list alice/demo --revision main
```

See [Hub Repositories](/docs/hub/repositories) for file, revision, and community semantics.

## Spaces workflows

Create and publish an application repository under a personal or organization namespace:

```bash
mega repos create mega/openapi --type space --public --exist-ok
mega spaces upload mega/openapi ./spaces/openapi . --sync \
  --commit-message "Publish OpenAPI explorer"
mega spaces info mega/openapi --format json
mega spaces restart mega/openapi
mega spaces wait mega/openapi --timeout 5m
mega spaces logs --follow mega/openapi
```

Runtime management is available directly from the CLI:

```bash
mega spaces hardware
mega spaces runtime mega/openapi --format json
mega spaces settings mega/openapi --hardware cpu-upgrade
mega spaces variables add mega/openapi -e MODE=production
mega spaces secrets add mega/openapi -s API_TOKEN
mega spaces volumes set mega/openapi -v mega://datasets/mega/data:/datasets/data:ro
mega spaces dev-mode mega/openapi
mega spaces hot-reload mega/openapi app.py -f ./app.py
mega spaces pause mega/openapi
```

Use `mega spaces templates`, `search`, and `card` for discovery. Dev Mode keeps
a writable source workspace, `hot-reload` atomically replaces one Python file
without rebuilding the image, and `ssh` enters that same rootless container
after Hub key and write-permission checks. Space volumes are read-only model,
dataset, Space, or Bucket snapshots; use a Space storage tier for
writable `/data`. Secret values are write-only and never appear in list output.
The [Spaces guide](/docs/hub/spaces) documents billing, Dev Mode entitlement,
runtime states, and recovery.

## Jobs workflows

```bash
mega jobs hardware
mega jobs balance
mega jobs run python:3.12-slim python -c 'print("hello")'
mega jobs ps --status RUNNING
mega jobs logs --follow --tail 100 <job-id>
mega jobs wait <job-id>
mega jobs usage
```

Recurring work lives under `mega jobs scheduled`. The [Jobs guide](/docs/hub/jobs) documents every supported command, option, SDK method, and endpoint.

## Sandbox workflows

```bash
mega sandbox create python:3.13
mega sandbox exec <sandbox-id> -- python -V
mega sandbox spawn <sandbox-id> -- python -m http.server 8000
mega sandbox process ls <sandbox-id>
mega sandbox cp ./input.json <sandbox-id>:/workspace/input.json
mega sandbox kill <sandbox-id>
```

Dedicated sessions use the authenticated native Sandbox API. Read-only model,
dataset, and Space snapshots can be attached with repeatable `-v/--volume`
mounts; bucket and writable mounts are rejected. File transfer uses bounded
chunks, and background processes can be inspected and stopped by PID.

`mega sandbox pool create` maintains a reusable template for faster Session
startup. Pool capacity and placement are managed by the service and may vary
with account limits and current availability.

## Extensions and skills

```bash
mega extensions search
mega extensions install ohtensorplay/mega-example
mega example --help
mega extensions update

mega skills preview
mega skills list
mega skills add
mega skills update
```

Extensions execute third-party code, so install only sources you trust. `mega skills add` installs the generated `mega-cli` skill by default; named skills come from the synchronized public MEGA skills marketplace.

## Webhook workflows

```bash
mega webhooks create \
  --name release-verifier \
  --url https://ci.example/hooks/mega \
  --event repo.updated

mega webhooks test <webhook-id>
mega webhooks deliveries <webhook-id> --limit 50
```

The create command reveals a generated signing secret once. See [Webhooks](/docs/hub/webhooks) before configuring a production receiver.

## Git, SSH, and GPG smoke test

MEGA deliberately uses different public hosts: Web/API and HTTPS Git use
`mega.tensorplay.cn`, while SSH Git uses the DNS-only `ssh.tensorplay.cn` host.

```bash
ssh-keygen -t ed25519 -C "$USER@$(hostname)" -f ~/.ssh/id_ed25519_mega
mega auth keys add ~/.ssh/id_ed25519_mega.pub --name workstation
ssh -T -i ~/.ssh/id_ed25519_mega git@ssh.tensorplay.cn
git clone git@ssh.tensorplay.cn:OWNER/REPOSITORY.git
```

Register and exercise a signing key whose email matches the MEGA account:

```bash
gpg --armor --export ACCOUNT_EMAIL > mega-signing-key.asc
mega auth keys add mega-signing-key.asc --type gpg --name release
git config user.email ACCOUNT_EMAIL
git config user.signingkey "$(gpg --list-secret-keys --with-colons ACCOUNT_EMAIL | awk -F: '$1=="sec" {print $5; exit}')"
git config commit.gpgsign true
git commit --allow-empty -m "Verify signed push"
git push origin HEAD
mega repos history OWNER/REPOSITORY --format json
```

The history record reports `signature_status` and `signer_fingerprint`. See [Authentication](/docs/hub/authentication#ssh-git-authentication) for host configuration and identity rules.

## Structured output

Commands that return records support shared formatting options:

| Option | Behavior |
| --- | --- |
| `--format auto` | Selects human or agent output from the terminal environment. |
| `--format human` | Renders readable tables and result messages. |
| `--format agent` | Produces stable, unstyled output for an AI or automation harness. |
| `--format json` or `--json` | Emits machine-readable JSON. |
| `--format quiet` or `-q` | Emits the primary ID, one value per line. |
| `--no-truncate` | Keeps complete scalar values in human tables. |

Prefer JSON for scripts:

```bash
mega repos list --owner research --format json
mega jobs ps --status RUNNING --format json
```

Use quiet output for shell composition:

```bash
mega repos list --type model -q
```

## Shell completion and help

```bash
mega --show-completion
mega --install-completion
mega repos create --help
mega jobs scheduled run --help
```

Leaf-command help is the authoritative option contract for the installed version.

## Environment variables

| Variable | Effect |
| --- | --- |
| `MEGA_TOKEN` | Overrides the active stored token. |
| `MEGA_ENDPOINT` | Changes the Hub endpoint. |
| `MEGA_INFERENCE_ROUTER_ENDPOINT` | Changes the OpenAI-compatible inference data-plane endpoint. |
| `MEGA_INFERENCE_MODELS_ENDPOINT` | Changes the public live Provider-model catalog endpoint. |
| `MEGA_HOME` | Changes the configuration and cache root. |
| `MEGA_DEBUG=1` | Enables full tracebacks. |

## Automation rules

- Use `--format json` instead of parsing human tables.
- Pass `--yes` only when the target ID was validated earlier in the same workflow.
- Inject secrets with environment variables; do not place tokens directly in committed scripts.
- Pin repository automation to an explicit branch, tag, or commit.
- Check the exit code. Foreground `mega jobs run` and `mega jobs wait` fail when the Job does not complete successfully.
