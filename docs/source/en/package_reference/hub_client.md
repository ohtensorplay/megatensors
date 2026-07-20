# Hub Python SDK


`MegaHubClient` is the typed Python client for MEGA Hub service operations. It is separate from the local [Tensor Runtime API](/docs/megatensors/index), which opens and loads `.mega` artifacts.

## Create a client

```python
from megatensors.hub import MegaHubClient

client = MegaHubClient()
print(client.whoami())
```

By default, the client uses `MEGA_ENDPOINT`, `MEGA_TOKEN`, or the token selected by `mega auth login`.

Override configuration explicitly when needed:

```python
import os

from megatensors.hub import MegaHubClient

client = MegaHubClient(
    endpoint="https://mega.tensorplay.cn",
    token=os.environ["MEGA_TOKEN"],
)
```

Pass `token=False` only for public, unauthenticated reads.

## Hugging Face Hub source compatibility

The complete Hub-style surface is available from `megatensors._hub`; the usual
imports are also re-exported by `megatensors.mega_hub`. The conventional names
remain available for migration, but execute the corresponding MEGA
implementation and use `MEGA_ENDPOINT`:

```python
from megatensors.mega_hub import HfApi, hf_hub_download

api = HfApi()
local_path = hf_hub_download("research/demo", "config.json")
```

`HfApi`, `HfFileSystem`, `HfFileMetadata`, `HfUri`, cache, OAuth, URL, and
TensorBoard compatibility names are aliases of their `Mega*` equivalents. For
example, `hf_hub_url(...)` returns a `mega.tensorplay.cn` artifact URL; it never
redirects repository operations to Hugging Face. New MEGA code should use the
`Mega*` spellings. Managed Inference Endpoint lifecycle APIs are outside this
compatibility scope.

## Repository methods

| Workflow | Methods |
| --- | --- |
| Metadata | `create_repo`, `list_repos`, `repo_info`, `update_repo`, `move_repo`, `duplicate_repo`, `delete_repo` |
| Files | `list_files`, `upload_file`, `upload_folder`, `delete_file`, `copy_file`, `copy_files` |
| Downloads | `download_file`, `download_files`, `snapshot_download`, `iter_snapshot_files` |
| Revisions | `list_refs`, `create_branch`, `delete_branch`, `create_tag`, `delete_tag`, `list_commits`, `get_commit`, `create_commit` |
| Community | `list_discussions`, `get_discussion`, `create_discussion`, `reply_to_discussion`, `update_discussion`, `merge_pull_request`, message and reaction methods |

Create and publish a repository:

```python
from megatensors.hub import MegaHubClient

client = MegaHubClient()
client.create_repo(
    "research/demo",
    repo_type="model",
    private=True,
    exist_ok=True,
)
client.upload_file(
    "research/demo",
    "./config.json",
    path_in_repo="config.json",
    revision="main",
    commit_message="Add model configuration",
)
```

## Job methods

```python
job = client.run_job(
    image="python:3.12-slim",
    command=["python", "-c", "print('hello')"],
    timeout="10m",
    labels={"lane": "release"},
)

for line in client.fetch_job_logs(job.id, follow=True, tail=100):
    print(line)

final = client.wait_for_job(job.id, timeout=900)
print(final.status.stage)
```

The Job method surface is:

| Workflow | Methods |
| --- | --- |
| Dispatch | `run_job` / `create_job` |
| Observe | `list_jobs`, `inspect_job` / `get_job`, `fetch_job_logs`, `wait_for_job`, `list_jobs_hardware`, `get_jobs_usage`, `get_compute_billing` |
| Control | `cancel_job` |
| Schedules | `create_scheduled_job`, `list_scheduled_jobs`, `inspect_scheduled_job`, `suspend_scheduled_job`, `resume_scheduled_job`, `trigger_scheduled_job`, `delete_scheduled_job` |

See [Jobs](/docs/hub/jobs) for limits, state transitions, namespaces, and Web API payloads.

## Inference client

`InferenceClient` targets the independent OpenAI-compatible Router rather than the Hub repository API:

```python
from megatensors import InferenceClient

client = InferenceClient(provider="auto")
result = client.chat.completions.create(
    model="mega/gpt-5.4-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Provider values `fastest`, `cheapest`, and `preferred` map to the corresponding model-selection suffix. A concrete Provider slug pins routed requests to that Provider when the client uses a MEGA token. Passing that Provider's own API key calls its public endpoint directly instead. `feature_extraction` uses the Router's OpenAI Embeddings endpoint.

Set `MEGA_INFERENCE_ROUTER_ENDPOINT` only for a staging or self-hosted Router. Production defaults to `https://inference.tensorplay.cn`. See [Your First Inference Provider Call](/docs/inference-providers/guides/first-api-call) for Python, OpenAI SDK, CLI, and curl examples.

## Space methods

The compatibility client exposes repository metadata and runtime actions:

```python
from megatensors._hub import MegaApi

api = MegaApi()
space = api.space_info("mega/openapi")
runtime = api.get_space_runtime("mega/openapi")
api.restart_space("mega/openapi")
```

Use `request_space_hardware`, `set_space_sleep_time`, `pause_space`, `restart_space`, variable and secret methods only with one of MEGA's configured fixed VPS flavors. See [Spaces](/docs/hub/spaces) and the live [OpenAPI Explorer](/spaces/mega/openapi).

## Webhook methods

```python
created = client.create_webhook(
    name="release-verifier",
    url="https://ci.example/hooks/mega",
    events=["repo.updated"],
)

delivery = client.test_webhook(created.webhook.webhook_id)
print(delivery.delivery_id, delivery.state)
```

The lifecycle methods are `list_webhooks`, `get_webhook`, `create_webhook`, `update_webhook`, `test_webhook`, `list_webhook_deliveries`, and `delete_webhook`.

## Account public keys

```python
from pathlib import Path

public_key = Path("~/.ssh/id_ed25519.pub").expanduser().read_text()
key = client.add_account_key(
    key_type="ssh",
    name="Work laptop",
    public_key=public_key,
)
```

Only `ssh` and `gpg` public keys are accepted. The client rejects private-key material locally.

## Errors

Service failures raise `MegaHubError`:

```python
from megatensors.hub import MegaHubError

try:
    client.repo_info("private/missing")
except MegaHubError as error:
    print(error.status_code, error.method, error.url)
    print(str(error))
```

Validation performed before a request usually raises `ValueError`. File methods may also raise normal filesystem exceptions such as `FileNotFoundError`.

## Client selection

| Need | Use |
| --- | --- |
| Shell and CI commands | [MEGA CLI](/docs/megatensors/guides/cli) |
| Typed Python Hub automation | `MegaHubClient` on this page |
| Routed model inference | `InferenceClient` and the Inference Provider guides |
| Framework tensor loading | [Tensor Runtime API](/docs/megatensors/index) |
| Another language or custom transport | [OpenAPI Explorer](/spaces/mega/openapi) |
