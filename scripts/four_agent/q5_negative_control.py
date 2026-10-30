#!/usr/bin/env python3
"""ADR-063 Q5 negative control: a bounded synthetic memory-growth child.

The supervisor runs this file as the "server" with a footprint cap below the
child's ceiling. The child allocates and touches memory in fixed steps and
never exceeds its declared ceiling. Passing means the supervisor's footprint
stop terminated it first. The child fails safe on its own: it exits when its
parent changes, when its lifetime ends, and refuses ceilings above HARD_MAX.

Exit codes: 0 never (a control that ends by itself did not prove a stop),
3 parent lost, 4 ceiling reached and held without a stop, 5 lifetime expired,
2 invalid arguments.
"""
import argparse
import os
import sys
import time

MIB = 1 << 20
GIB = 1 << 30
PAGE = 16384
HARD_MAX_BYTES = 32 * GIB
SAFETY_BYTES = 4 * GIB


class ControlError(ValueError):
    pass


def plan_ceiling(baseline_free_cached, free_cached_floor, requested_ceiling, cap):
    """Ceiling the child may reach: requested, but always below the host's live headroom.

    The child's ceiling must leave the free+cached stop floor plus a fixed
    safety margin untouched even if the footprint stop never fires, and the
    footprint cap must sit strictly below the ceiling so the stop can fire first.
    """
    for name, value in (('baseline_free_cached', baseline_free_cached), ('requested_ceiling', requested_ceiling),
                        ('free_cached_floor', free_cached_floor), ('cap', cap)):
        if type(value) is not int or value <= 0:
            raise ControlError(name + ' must be a positive integer')
    headroom = baseline_free_cached - free_cached_floor - SAFETY_BYTES
    if requested_ceiling > HARD_MAX_BYTES:
        raise ControlError(f'ceiling {requested_ceiling} B exceeds hard maximum {HARD_MAX_BYTES} B')
    if requested_ceiling > headroom:
        raise ControlError(f'ceiling {requested_ceiling} B exceeds baseline headroom {headroom} B')
    if not 0 < cap < requested_ceiling:
        raise ControlError('footprint cap must be below the ceiling')
    return {'ceiling_bytes': requested_ceiling, 'cap_bytes': cap, 'headroom_bytes': headroom,
            'safety_bytes': SAFETY_BYTES, 'hard_max_bytes': HARD_MAX_BYTES}


def steps(ceiling, step):
    """Allocation sizes: whole steps, then a final partial step, summing exactly to the ceiling."""
    if step <= 0 or ceiling <= 0:
        raise ControlError('step and ceiling must be positive')
    sizes, total = [], 0
    while total < ceiling:
        size = min(step, ceiling - total)
        sizes.append(size)
        total += size
    return sizes


def touch(size):
    block = bytearray(size)
    count = len(range(0, size, PAGE))
    block[0::PAGE] = b'\x01' * count  # dirty every page so it is resident and charged
    return block


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ceiling-bytes', type=int, required=True)
    parser.add_argument('--step-bytes', type=int, default=256 * MIB)
    parser.add_argument('--step-seconds', type=float, default=0.25)
    parser.add_argument('--hold-seconds', type=float, default=10.0)
    parser.add_argument('--max-seconds', type=float, default=180.0)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.ceiling_bytes <= HARD_MAX_BYTES or not 0 < args.step_bytes <= GIB:
        print('refusing: ceiling or step outside hard bounds', file=sys.stderr)
        return 2
    started = time.monotonic()
    held, allocated = [], 0

    def alive():
        if os.getppid() != args.parent_pid:
            print('parent lost; releasing memory', file=sys.stderr, flush=True)
            raise SystemExit(3)
        if time.monotonic() - started > args.max_seconds:
            print('lifetime expired', file=sys.stderr, flush=True)
            raise SystemExit(5)

    for size in steps(args.ceiling_bytes, args.step_bytes):
        alive()
        if allocated + size > args.ceiling_bytes:  # defensive: steps() never overshoots
            break
        held.append(touch(size))
        allocated += size
        print(f'allocated {allocated}', flush=True)
        time.sleep(args.step_seconds)
    deadline = time.monotonic() + args.hold_seconds
    while time.monotonic() < deadline:
        alive()
        time.sleep(0.1)
    print('ceiling reached and held without a supervisor stop', file=sys.stderr, flush=True)
    return 4


if __name__ == '__main__':
    raise SystemExit(main())
