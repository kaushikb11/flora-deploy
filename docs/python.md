# Python API

Everything the CLI does for a job is also available from Python, in `flora_deploy.sdk`. Use it
to start jobs from a script, a notebook or a scheduler.

`uv tool install` keeps `flora_deploy` in its own environment. To import it, install the package
into your project's environment instead:

```bash
pip install git+https://github.com/kaushikb11/flora-deploy
```

## Run a job and read its labels

The config's app must already be deployed (`flora deploy`). `submit` then works like
`flora run --detach`.

```python
from flora_deploy import sdk

job = sdk.submit("flora.yaml", budget_usd=50)   # starts the job and returns
view = job.wait(timeout_s=12 * 3600)            # until done, budget stop, failure or timeout
if view.state in ("done", "completed_with_errors"):
    captions = job.results("caption")           # the stage's records
    job.export("./labeled", allow_uncertified=True)
```

- `job.wait()` polls until the job stops or `timeout_s` passes, and returns a `JobView` with
  `state`, `spend_usd`, `projected_usd`, `budget_usd` and `eta_s`. If no orchestrator is alive and
  work remains, it raises `sdk.FloraError`; `job.resume()` restarts it.
- Final states: `done`, `completed_with_errors` (some episodes excluded), `budget_stopped`,
  `failed` and `cancelled`. `view.exit_code` gives the CLI's exit code for the state.
- `job.results(stage)` returns the stage's records (see [`output-format.md`](output-format.md)).
- `job.export(...)` writes the labels into a copy of the LeRobot dataset, with the same options as
  `flora export` (see [`export.md`](export.md)).

## Attach to a running job

From another process (an Airflow sensor, say), open the job by its id:

```python
job = sdk.open_job("my-labels", config="flora.yaml")   # or store="s3://bucket/prefix"
print(job.status().state)
job.resume(budget_usd=100)                             # after a budget stop
job.cancel()
job.report()                                           # what it spent, as a dict
```

For a Volume store, pass `store="modal-volume://name/dir"` or `config=`.
