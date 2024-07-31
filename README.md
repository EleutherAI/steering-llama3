# Steering Llama 3 (and Llama 2)

Requirements:

`pip install -r requirements`

vLLM also requires:

`pip install "outlines[serve]"`

To generate steering vectors and activations:

`python steering/generate_vectors.py <args>`

To do open-ended generation with steering and/or concept editing:

`python steering/steer.py <args>`

To evaluate the resulting responses with Llama 3 70B:

`python steering/eval.py <args>`

To make plots:

`python steering/plots.py` (arguments are optional, will plot everything by default)

Note that these have to be called from `steering-llama3/` (or with it in the Python path) in order for imports to work correctly.

## `gpu_graph` and run files

I usually run experiments using the tool `steering/gpu_graph.py`, which takes a list of GPUs and a directed graph of shell commands and runs commands (after their dependencies) on any available GPU. The original command output is dumped into `logs/`.

`runs/` contains a bunch of chronologically-named files that use `gpu_graph` to do various combinations of runs; these might be useful for getting a sense of the command arguments.

## Command-line arguments

`--help` is available from argparse, but some details that might help:

* Settings for generation, steering, and evaluation are managed via the Settings class in `steering/common.py`. Note that generation takes a smaller subset of these settings.
* `--behavior` selects a dataset derived from Rimsky et al (must be used with `--dataset openr` or `ab`) for vector generation, and also selects a corresponding evaluation dataset.
* Behavior `None` (no command-line arg passed) is evaluated with separate batches of harmful and harmless prompts. If `--dataset openr` or `ab` is passed with behavior None, the `refusal` behavior is used to generate steering vectors, but the harmful/harmless prompts are used for evaluation.
* `--residual` generates steering vectors and applies steering to residual blocks, instead of to the residual stream directly. I usually use this together with `--layer all` but they are independently controlled.
* `--scrub` applies concept scrubbing and should only be used with `--layer all --residual`; I don't know what happens otherwise but it probably doesn't work.
* `--toks` controls where vectors and concept editors are applied (`before` for the prompt, `after` for the completion, or `None` for both).
* `--logit` only applies to datasets `ab` and `abcon` and relies on the A/B completion format.

## Other files

There are some miscellaneous Python files and shell scripts kicking around for safekeeping, which probably shouldn't be on `main`. Some that are actually useful include:

* `server.sh` runs the vLLM server on GPU 0 and port 8005; once it's running, `eval.py` will look on port 8005 by default. As a more flexible alternative, you can run `eval.py --server` to start the vLLM server within the eval script. If you run multiple copies of `eval.py`, you should include `--port` with a distinct value for each.
* `python smartcopy.py foo.tgz parent/foo/` extracts a compressed tarball into the directory provided, overwriting files only if the tarred version is larger. This is the correct way to merge responses across machines, because responses and their evaluation scores are (unfortunately) stored in the same file and you don't want to clobber evaluated ones with unevaluated ones.
* `backtar.sh` is a slightly convenient tool for archiving `artifacts/responses/` to sync between machines. Pass it an arbitrary suffix to append to the tarball (e.g. a date, hostname, and/or timestamp). `backup.sh` also sends the file to `adam-ord`, which is hardcoded; please don't use it.