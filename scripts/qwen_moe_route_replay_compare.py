#!/usr/bin/env python3
"""Compare teacher-forced DFlash2 verify replays of Qwen3.6 MoE expert routes.

usage: qwen_moe_route_replay_compare.py <reference.jsonl> <route.jsonl>...

Each file is the output of the ignored test `qwen36_dflash2_verify_moe_route_replay` for one process setting
(first line a header, then one line per verify row). Rows are matched by (lane, position); every file must hold the
same rows with the same input tokens. For each route against the reference:

  argmax flips            rows whose top-1 token differs from the reference's, and how many of them the reference
                          itself decided by less than 0.5 / 0.1 logits (its top-1 minus top-2)
  logit error             relative RMS and largest absolute difference over the fixed vocabulary sample
  top-1 logit shift       mean and largest absolute change of the reference top-1 token's own logit is not stored;
                          the shift of each file's own top-1 logit is reported instead
  margin                  median top-1 minus top-2 logit of the route, and of the reference on the same rows
  stream agreement        share of rows whose top-1 is the text's actual next token (a route-neutral predictor)
"""
import json
import statistics
import sys


def load(path):
    with open(path) as handle:
        header = json.loads(handle.readline())
        rows = {}
        for line in handle:
            row = json.loads(line)
            rows[(row['lane'], row['position'])] = row
    return header, rows


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref_header, ref = load(sys.argv[1])
    print(f'reference {sys.argv[1]}: route {ref_header["moe_expert_decode_route"]}, {len(ref)} rows, '
          f'{ref_header["lanes"]} lanes, {ref_header["steps"]} steps')
    ref_margin = {key: row['top'][0][1] - row['top'][1][1] for key, row in ref.items()}
    agree = sum(1 for row in ref.values() if row['top'][0][0] == row['next']) / len(ref)
    print(f'  top-1 is the stream\'s next token on {agree:.4f} of rows; median top-2 margin '
          f'{statistics.median(ref_margin.values()):.3f}, rows under 0.5: '
          f'{sum(1 for m in ref_margin.values() if m < 0.5) / len(ref):.4f}, under 0.1: '
          f'{sum(1 for m in ref_margin.values() if m < 0.1) / len(ref):.4f}')
    others = []
    for path in sys.argv[2:]:
        header, rows = load(path)
        if rows.keys() != ref.keys():
            sys.exit(f'{path}: rows differ from the reference')
        if any(rows[key]['input'] != ref[key]['input'] for key in ref):
            sys.exit(f'{path}: input tokens differ from the reference')
        others.append((path, header, rows))
        flips = [key for key in ref if rows[key]['top'][0][0] != ref[key]['top'][0][0]]
        close = lambda bound: sum(1 for key in flips if ref_margin[key] < bound)
        squared, reference_squared, largest = 0.0, 0.0, 0.0
        for key in ref:
            for a, b in zip(rows[key]['sample'], ref[key]['sample']):
                squared += (a - b) ** 2
                reference_squared += b * b
                largest = max(largest, abs(a - b))
        top_shift = [abs(rows[key]['top'][0][1] - ref[key]['top'][0][1]) for key in ref
                     if rows[key]['top'][0][0] == ref[key]['top'][0][0]]
        next_shift = [abs(rows[key]['next_logit'] - ref[key]['next_logit']) for key in ref]
        margin = [rows[key]['top'][0][1] - rows[key]['top'][1][1] for key in ref]
        agree = sum(1 for row in rows.values() if row['top'][0][0] == row['next']) / len(rows)
        print(f'{path}: route {header["moe_expert_decode_route"]}')
        print(f'  argmax flips against the reference: {len(flips)} of {len(ref)} ({len(flips) / len(ref):.4%}); '
              f'reference margin under 0.5 on {close(0.5)} of them, under 0.1 on {close(0.1)}; largest reference '
              f'margin flipped {max((ref_margin[key] for key in flips), default=0.0):.3f}')
        print(f'  sampled logits: relative RMS error {(squared / reference_squared) ** 0.5:.3e}, largest absolute '
              f'difference {largest:.4f}')
        print(f'  logit shift of the agreed top-1 token: mean {statistics.mean(top_shift):.4f}, p99 '
              f'{percentile(top_shift, 0.99):.4f}, max {max(top_shift):.4f}; of the stream\'s next token: mean '
              f'{statistics.mean(next_shift):.4f}, max {max(next_shift):.4f}')
        print(f'  median top-2 margin {statistics.median(margin):.3f}; top-1 is the stream\'s next token on '
              f'{agree:.4f} of rows')
    for index, (path_a, _, rows_a) in enumerate(others):
        for path_b, _, rows_b in others[index + 1:]:
            flips = sum(1 for key in ref if rows_a[key]['top'][0][0] != rows_b[key]['top'][0][0])
            print(f'{path_a} against {path_b}: {flips} argmax differences ({flips / len(ref):.4%})')


if __name__ == '__main__':
    main()
