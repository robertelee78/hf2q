#!/usr/bin/env python3
"""q5 serve entry of the Splash arm (q5_cases/splash-*.json serve.argv).

Checks the launch inputs, the pinned checkout/build/venv and the pinned DFlash2 draft, prepares the run's own
Splash model directory from the hf2q GGUF (its model.json must record the GGUF's content digest equal to
MODEL_SHA256), records {run_dir}/launch.json (pid, exact argv, environment, identity, assembly, the cross-run Neural
Engine calibration state) and then execs the Splash server in this same process, so q5's SERVER_PID is the server.
Every rule lives in native_peer_splash_ready, which the client readiness re-derives; this file only sequences it.
Nothing here falls back: any refusal exits before a server starts.
"""
import argparse
import json
import os
import sys

import native_peer_splash_ready as ready
from qwen38_n1_verdict import require


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ready.LAUNCH_KEYS:
        parser.add_argument('--' + name.replace('_', '-'), dest=name, required=True)
    spec = ready.launch_spec(vars(parser.parse_args(argv)))
    environ = dict(os.environ)
    ready.check_environ(environ)
    identity = ready.verify_identity(spec)
    require(identity['venv']['freeze_sha256'] == spec['venv_freeze_sha256'],
            'Splash venv freeze differs from SPLASH_VENV_FREEZE_SHA256')
    assembly_path, assembly = ready.prepare_run(spec, environ)
    record = ready.launch_record(spec, identity, assembly, environ)
    with open(ready.layout(spec)['launch'], 'x') as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
    sys.stdout.flush()
    sys.stderr.flush()
    command = ready.server_argv(spec, assembly_path)
    os.execve(command[0], command, ready.serve_environ(spec, environ))


if __name__ == '__main__':
    main()
