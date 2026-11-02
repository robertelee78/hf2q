#!/usr/bin/env python3
"""q5 serve entry of the oMLX arm (q5_cases/native-omlx-*.json serve.argv).

Checks the launch inputs and the source/venv/snapshot identity, writes the run's own oMLX base directory
(model_settings.json, settings.json), links the pinned snapshot as the only served model, records
{run_dir}/launch.json (pid, exact argv, environment, identity) and then execs `omlx serve` in this same process,
so q5's SERVER_PID is the server. Every rule lives in native_peer_omlx_ready (prepare_run, server_argv), which
the client readiness re-derives; this file only sequences it. Nothing here falls back: any refusal exits
before a server starts.
"""
import argparse
import json
import os
import sys

import native_peer_omlx_ready as ready


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ready.LAUNCH_KEYS:
        parser.add_argument('--' + name.replace('_', '-'), dest=name, required=True)
    spec = ready.launch_spec(vars(parser.parse_args(argv)))
    environ = dict(os.environ)
    ready.check_environ(environ)
    identity = ready.verify_identity(spec)
    command = ready.prepare_run(spec)
    record = ready.launch_record(spec, identity, environ)
    with open(ready.layout(spec)['launch'], 'x') as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(command[0], command, environ)


if __name__ == '__main__':
    main()
