#!/usr/bin/env python3
"""ADR-063 S3 cancel smoke (S-C1) and joint row-width trace (S-C2).

`run` replays four recorded four-agent requests (client/agent-N/turn-K/request.json)
as concurrent streams against an already running hf2q server. It closes one chosen
stream once that stream has K semantic events (content, reasoning or tool-call
deltas) while the other three are still decoding. It then checks that:
- the three survivors end with valid SSE: one choice, a stop/tool_calls/length
  finish_reason, usage, [DONE], well-formed tool calls and no error frames;
- /metrics hf2q_sse_cancellations rose by exactly one;
- the server log shows a joint decode/MTP turn of width >= 2 before the close;
- the server log has no ERROR or poison lines during the run, and /readyz stays ready.
An optional control pass replays the same four requests without a cancel. Its
completion shows that the cancelled slot was released. It also compares the
survivors' outputs with the cancel pass (exact equality is expected on bitwise
routes; reported, not gated).


Qwen3.6 replay requires completed source receipts bound to repair_turns=1.
`--c2-anchor-depth 1|4` first completes a recorded predecessor cohort, then
requires one actual acknowledged cancellation recovery event, checks idle
pending ownership, and retries only the cancelled request. Recovery disposition
and logical depth/rank come from that event, never global restore-counter deltas;
retry TTFT starts at POST submission and stops at the first semantic SSE delta.
`widths` histograms the per-turn joint decode/MTP debug lines of any server
log (S-C2). The server must run with the debug target enabled, e.g. launcher
LOG_FILTER (see README "S3 cancel smoke").

Never launches a server, model or build. Writes one JSON receipt (summary.json).
"""
import argparse
import collections
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import urllib.request
from urllib.parse import urlsplit
SCHEMA = 'hf2q.s3.cancel-smoke.v2'
WIDTHS_SCHEMA = 'hf2q.s3.joint-widths.v1'
PASS, FAIL, INVALID = 'CancelSmokePass', 'CancelSmokeFail', 'CancelSmokeInvalid'
FAMILIES = {'deepseek4': 'DeepSeek-V4', 'qwen38': 'Qwen', 'qwen36': 'Qwen'}
FINISHES = ('stop', 'tool_calls', 'length')
RAW_LIMIT = 32 * 1024 * 1024
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
JOINT = re.compile(r'\b(?P<engine>DeepSeek-V4|Qwen) joint (?P<kind>decode|MTP) turn(?P<fields>(?:\s+[a-z_]+=\S+)*)')
DFLASH = re.compile(r'\bQwen DFlash2 stage timing(?P<fields>(?:\s+[a-z_]+=\S+)*)')
DFLASH_TURN = re.compile(r'\bQwen DFlash2 decode turn(?P<fields>(?:\s+[a-z_]+=\S+)*)')
FIELD = re.compile(r'([a-z_]+)=(\S+)')
ERROR_LINE = re.compile(r'^\s*ERROR\s')
RECOVERY = re.compile(r'\bQwen anchor cancellation recovery(?P<fields>(?:\s+[a-z_]+=\S+)*)')
# A poison event, not a `cache_poisoned=false` field on an INFO progress line.
POISON = re.compile(r'poison(?!(?:ed)?=false\b)', re.IGNORECASE)
CANCELLATIONS = re.compile(r'^hf2q_sse_cancellations\s+(\d+)\s*$', re.MULTILINE)
LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Invalid(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise Invalid(why)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def loads(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, f'duplicate JSON key {key}')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


# ----------------------------------------------------------------------------- requests

def endpoint(base_url):
    parts = urlsplit(base_url)
    require(parts.scheme == 'http' and parts.hostname == '127.0.0.1' and parts.port and parts.path == '/v1'
            and not parts.query and not parts.fragment and parts.username is None,
            'base URL must be http://127.0.0.1:<port>/v1')
    return parts.hostname, parts.port


def replay_request(raw, model):
    """The recorded request, streamed with usage; everything else verbatim."""
    request = loads(raw)
    require(isinstance(request, dict) and isinstance(request.get('messages'), list) and request['messages'],
            'recorded request needs messages')
    require(request.get('model') == model, f"recorded model {request.get('model')!r} differs from served {model!r}")
    request['stream'] = True
    request['stream_options'] = {'include_usage': True}
    return json.dumps(request, ensure_ascii=False, separators=(',', ':')).encode()


def load_requests(directory, turn, agents, model):
    rows = []
    for agent in agents:
        path = Path(directory) / f'agent-{agent}' / f'turn-{turn}' / 'request.json'
        raw = path.read_bytes()
        body = replay_request(raw, model)
        rows.append({'agent': agent, 'turn': turn, 'path': str(path), 'recorded_sha256': sha256_bytes(raw),
                     'replayed_sha256': sha256_bytes(body), 'body': body})
    return rows


def repair_workload_sources(directory, agents, selected_requests):
    sources = {}
    for agent in agents:
        path = Path(directory) / f'agent-{agent}' / 'summary.json'
        raw = path.read_bytes()
        summary = loads(raw)
        require(isinstance(summary, dict) and type(summary.get('repair_turns')) is int
                and summary['repair_turns'] == 1
                and summary.get('status') == 'ProtocolCompleteAwaitingIndependentPatchReview',
                f'Qwen3.6 source {path} is not a completed declared repair-turn workload')
        turns = summary.get('turns')
        require(isinstance(turns, list), f'Qwen3.6 source {path} lacks retained request receipts')
        selected = {}
        for request in selected_requests:
            if request['agent'] != agent:
                continue
            matches = [row for row in turns if isinstance(row, dict) and type(row.get('turn')) is int
                       and row['turn'] == request['turn']]
            require(len(matches) == 1 and matches[0].get('request_sha256') == request['recorded_sha256'],
                    f'Qwen3.6 source {path} does not bind replayed turn {request["turn"]}')
            selected[str(request['turn'])] = request['recorded_sha256']
        require(selected, f'Qwen3.6 source {path} has no selected requests')
        sources[f'agent-{agent}'] = {'path': str(path), 'sha256': sha256_bytes(raw), 'repair_turns': 1,
                                    'selected_request_sha256': selected}
    return sources


# ----------------------------------------------------------------------------- SSE

class Accumulator:
    """One streamed chat completion; raises Invalid on the first protocol violation."""

    def __init__(self, model):
        self.model = model
        self.identity = None
        self.content, self.reasoning = '', ''
        self.calls = {}
        self.finish = None
        self.usage = None
        self.done = False
        self.frames = 0
        self.events = 0

    def feed(self, data):
        """Returns True when the frame carried a semantic event."""
        require(not self.done, 'data after [DONE]')
        self.frames += 1
        if data == '[DONE]':
            require(self.finish is not None, '[DONE] before finish_reason')
            require(self.usage is not None, '[DONE] before usage')
            self.done = True
            return False
        obj = loads(data)
        require(isinstance(obj, dict) and 'error' not in obj, 'SSE error frame')
        require(obj.get('model') == self.model and isinstance(obj.get('id'), str), 'SSE identity')
        self.identity = self.identity or obj['id']
        require(obj['id'] == self.identity, 'SSE id changed')
        choices = obj.get('choices')
        require(isinstance(choices, list), 'SSE choices')
        if not choices:
            require(obj.get('usage') is not None, 'empty choices without usage')
            self.record_usage(obj['usage'])
            return False
        require(len(choices) == 1 and choices[0].get('index') == 0, 'choice count/index')
        require(self.finish is None, 'choice after finish_reason')
        choice = choices[0]
        delta = choice.get('delta', {})
        require(isinstance(delta, dict) and set(delta) <= {'role', 'content', 'reasoning_content', 'tool_calls'},
                'SSE delta')
        require(delta.get('role', 'assistant') == 'assistant', 'SSE role')
        semantic = False
        for key in ('content', 'reasoning_content'):
            value = delta.get(key)
            if value is not None:
                require(isinstance(value, str), 'non-string delta')
                if key == 'content':
                    self.content += value
                else:
                    self.reasoning += value
                semantic |= bool(value)
        for part in delta.get('tool_calls') or []:
            require(isinstance(part, dict) and type(part.get('index')) is int and part['index'] >= 0, 'tool index')
            call = self.calls.setdefault(part['index'], {'id': None, 'name': '', 'arguments': ''})
            if part.get('id') is not None:
                require(call['id'] is None and isinstance(part['id'], str) and part['id'], 'tool id')
                call['id'] = part['id']
            require(part.get('type', 'function') == 'function', 'tool type')
            function = part.get('function') or {}
            require(isinstance(function, dict) and set(function) <= {'name', 'arguments'}, 'tool delta')
            for key, value in function.items():
                require(isinstance(value, str), 'tool delta not a string')
                call[key] += value
                semantic |= bool(value)
        finish = choice.get('finish_reason')
        if finish is not None:
            require(finish != 'error', 'finish_reason error')
            require(finish in FINISHES, f'unknown finish_reason {finish!r}')
            self.finish = finish
        if obj.get('usage') is not None:
            self.record_usage(obj['usage'])  # hf2q puts usage on the finish chunk
        self.events += semantic
        return semantic

    def record_usage(self, usage):
        require(self.finish is not None and self.usage is None, 'usage before finish or repeated')
        require(isinstance(usage, dict) and all(type(usage.get(k)) is int for k in
                                                ('prompt_tokens', 'completion_tokens', 'total_tokens'))
                and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'usage')
        self.usage = usage

    def output(self):
        """Comparable output; tool-call ids are random per request and excluded."""
        return {'content': self.content, 'reasoning_content': self.reasoning, 'finish_reason': self.finish,
                'tool_calls': [{'name': c['name'], 'arguments': c['arguments']}
                               for _, c in sorted(self.calls.items())]}

    def validate(self):
        require(self.done, 'missing [DONE]')
        indices = sorted(self.calls)
        require(indices == list(range(len(indices))), 'tool call indices not dense')
        for index in indices:
            call = self.calls[index]
            require(call['id'] is not None and call['name'], f'tool call {index} missing id or name')
            try:
                arguments = loads(call['arguments'])
            except ValueError as error:
                raise Invalid(f'tool call {index} arguments are not JSON: {error}') from error
            require(isinstance(arguments, dict), f'tool call {index} arguments are not an object')
        require((self.finish == 'tool_calls') == bool(self.calls), 'finish_reason tool_calls without calls or vice versa')


# ----------------------------------------------------------------------------- server side

def root_url(base_url):
    host, port = endpoint(base_url)
    return f'http://{host}:{port}'


def cancellations(base_url):
    with LOOPBACK.open(root_url(base_url) + '/metrics', timeout=5) as response:
        text = response.read().decode('utf8', 'replace')
    match = CANCELLATIONS.search(text)
    require(match is not None, '/metrics has no hf2q_sse_cancellations')
    return int(match.group(1))


def anchor_idle_state(base_url):
    with LOOPBACK.open(root_url(base_url) + '/metrics', timeout=5) as response:
        text = response.read().decode('utf8', 'strict')
    committed = {}
    pending = {}
    for slot, count in re.findall(r'^hf2q_qwen_anchor_committed\{slot="(\d+)"\} (\d+)$', text, re.MULTILINE):
        require(slot not in committed, 'duplicate anchor committed gauge')
        committed[slot] = int(count)
    for slot, kind, count in re.findall(
            r'^hf2q_qwen_anchor_owned_bytes\{slot="(\d+)",class="([^"]+)",state="pending"\} (\d+)$',
            text, re.MULTILINE):
        key = (slot, kind)
        require(key not in pending, 'duplicate anchor pending gauge')
        pending[key] = int(count)
    classes = {'payload_image', 'logits', 'hidden', 'tokens', 'metadata'}
    require(len(committed) == 4 and set(pending) == {(slot, kind) for slot in committed for kind in classes},
            'C2 requires complete four-slot anchor idle gauges')
    require(not any(pending.values()), 'pending anchor ownership remains after requests settled')
    return {'committed_by_slot': committed, 'pending_owned_bytes': sum(pending.values())}


def ready(base_url):
    try:
        with LOOPBACK.open(root_url(base_url) + '/readyz', timeout=5) as response:
            return loads(response.read()).get('ready') is True
    except (OSError, ValueError):
        return False


def log_size(path):
    return os.stat(path).st_size


def log_lines(path, start, end):
    """Complete lines in [start, end) with their end offsets."""
    with open(path, 'rb') as handle:
        handle.seek(start)
        data = handle.read(max(0, end - start))
    out, offset = [], start
    for raw in data.splitlines(keepends=True):
        offset += len(raw)
        if raw.endswith(b'\n'):
            out.append((offset, ANSI.sub('', raw.decode('utf8', 'replace')).rstrip('\n')))
    return out


def joint_turns(lines, family):
    engine = FAMILIES[family]
    turns = []
    has_dflash_turns = any(DFLASH_TURN.search(line) for _, line in lines)
    for end, line in lines:
        dflash_turn = DFLASH_TURN.search(line) if family.startswith('qwen') else None
        if dflash_turn is not None:
            fields = parse_integer_fields(
                ' '.join(f'{key}={value}' for key, value in FIELD.findall(dflash_turn.group('fields'))
                         if key != 'path'), 'DFlash2 decode turn')
            require({'lanes', 'serial', 'turn_lanes'} <= fields.keys(), 'DFlash2 turn lacks scope')
            turns.append({'end_offset': end, 'kind': 'DFlash2', 'width': fields['lanes'], **fields})
            continue
        dflash = DFLASH.search(line) if family.startswith('qwen') and not has_dflash_turns else None
        if dflash is not None:
            fields = parse_integer_fields(dflash.group('fields'), 'DFlash2 stage')
            require('active_lanes' in fields, f'DFlash2 stage line lacks active_lanes: {line}')
            turns.append({'end_offset': end, 'kind': 'DFlash2',
                          'width': fields['active_lanes'], 'serial': 0, **fields})
            continue
        match = JOINT.search(line)
        if match is None or match.group('engine') != engine:
            continue
        # Closed-refusal lines describe no forward and therefore have no width.
        if 'reason=' in match.group('fields'):
            continue
        fields = parse_integer_fields(
            ' '.join(f'{key}={value}' for key, value in FIELD.findall(match.group('fields'))
                     if key != 'path'), 'joint turn')
        if family == 'deepseek4' or match.group('kind') == 'MTP':
            width_key = 'lanes'
        else:
            width_key = 'rows'
        require(width_key in fields and 'serial' in fields, f'joint turn line lacks {width_key}/serial: {line}')
        turns.append({'end_offset': end, 'kind': match.group('kind'), 'width': fields[width_key], **fields})
    return turns


def parse_integer_fields(raw, context):
    fields = {}
    for key, value in FIELD.findall(raw):
        require(value.isdecimal(), f'non-integer {context} field {key}={value}')
        fields[key] = int(value)
    return fields


def histogram(turns, family):
    """Step-width histogram: one Qwen turn is one joint forward of `rows` rows.

    A DeepSeek-V4 turn runs up to `budget` lockstep steps of `lanes` rows; lanes
    that finish mid-turn shrink later steps, so budget-weighted counts are upper
    bounds on steps at that width.
    """
    by_width = collections.Counter(t['width'] for t in turns)
    result = {'turns': len(turns), 'turns_by_width': {str(w): by_width[w] for w in sorted(by_width)},
              'rows_by_width': {str(w): w * by_width[w] for w in sorted(by_width)},
              'turns_with_serial_lanes': sum(1 for t in turns if t['serial'] > 0),
              'serial_lanes_total': sum(t['serial'] for t in turns)}
    if family == 'deepseek4':
        steps = collections.Counter()
        for t in turns:
            steps[t['width']] += t.get('budget', 1)
        result['budget_steps_by_width_upper_bound'] = {str(w): steps[w] for w in sorted(steps)}
    if turns:
        rows = sum(t['width'] for t in turns)
        result['mean_width'] = rows / len(turns)
        result['joint_share_turns_width_ge_2'] = sum(1 for t in turns if t['width'] >= 2) / len(turns)
    return result


def problem_lines(lines):
    return [line for _, line in lines if ERROR_LINE.match(line) or POISON.search(line)]


def cancellation_recovery(lines, close_offset):
    """One acknowledged recovery in the owned cancellation interval, never counter inference."""
    events = []
    for end, line in lines:
        match = RECOVERY.search(line)
        if match is None or end <= close_offset:
            continue
        fields = dict(FIELD.findall(match.group('fields')))
        required = {'request_id', 'slot', 'disposition', 'recovered_tokens',
                    'restored_rank', 'performed_restore_count'}
        require(required <= fields.keys(), f'incomplete cancellation recovery event: {line}')
        disposition = fields['disposition'].strip('"')
        require(disposition in ('preserved', 'restored', 'reset'), 'unknown cancellation disposition')
        numeric = {}
        for key in required - {'disposition'}:
            require(re.fullmatch(r'-?\d+', fields[key]) is not None, f'invalid recovery {key}')
            numeric[key] = int(fields[key])
        depth, rank, count = (numeric[k] for k in ('recovered_tokens', 'restored_rank', 'performed_restore_count'))
        require(numeric['request_id'] >= 0 and numeric['slot'] >= 0 and depth >= 0, 'negative recovery identity/depth')
        require((disposition == 'restored' and depth > 0 and 0 <= rank < 8 and count == 1)
                or (disposition == 'preserved' and depth > 0 and rank == -1 and count == 0)
                or (disposition == 'reset' and depth == 0 and rank == -1 and count == 0),
                'inconsistent recovery disposition/rank/depth/count')
        events.append({'end_offset': end, 'disposition': disposition, **numeric,
                       'preserved_live_prefix': disposition == 'preserved',
                       'restored_tokens': depth if disposition == 'restored' else 0,
                       'preserved_tokens': depth if disposition == 'preserved' else 0})
    require(len(events) == 1, f'expected one owned cancellation recovery, observed {len(events)}')
    return events[0]


# ----------------------------------------------------------------------------- streams

class Stream(threading.Thread):
    def __init__(self, coordinator, row, directory):
        super().__init__(name=f"agent-{row['agent']}", daemon=True)
        self.coordinator, self.row, self.directory = coordinator, row, directory
        self.agent = row['agent']
        self.accumulator = Accumulator(coordinator.model)
        self.result = {'agent': self.agent, 'request': row['path'], 'replayed_sha256': row['replayed_sha256']}
        self.finished = False

    def run(self):
        c, result = self.coordinator, self.result
        self.directory.mkdir(parents=True)
        (self.directory / 'request.json').write_bytes(self.row['body'])
        connection = http.client.HTTPConnection(c.host, c.port, timeout=c.timeout)
        started = None
        try:
            c.start.wait()
            started = time.monotonic()
            connection.request('POST', '/v1/chat/completions', self.row['body'],
                               {'Content-Type': 'application/json', 'Connection': 'close'})
            sock = connection.sock
            response = connection.getresponse()
            result['http_status'] = response.status
            result['content_type'] = response.getheader('Content-Type', '').split(';', 1)[0].strip()
            require(response.status == 200, f'HTTP {response.status}')
            require(result['content_type'] == 'text/event-stream', 'response media type')
            self.read(response, sock, started)
            if not result.get('cancelled_by_client'):
                self.accumulator.validate()
                result['valid'] = True
        except Exception as error:  # every failure is this stream's recorded result, never a lost thread
            result['valid'] = False
            result['error'] = f'{type(error).__name__}: {error}'
        finally:
            connection.close()
            result['elapsed_seconds'] = time.monotonic() - started if started is not None else None
            acc = self.accumulator
            result.update(frames=acc.frames, semantic_events=acc.events, finish_reason=acc.finish,
                          usage=acc.usage, done=acc.done, output=acc.output())
            save(self.directory / 'assembled.json', result['output'])
            with c.lock:
                self.finished = True

    def read(self, response, sock, started):
        c, acc, result = self.coordinator, self.accumulator, self.result
        frame, pending, total = [], bytearray(), 0
        with (self.directory / 'response.bin').open('xb') as archive:
            while True:
                require(time.monotonic() - started <= c.timeout, 'request deadline')
                block = response.read1(16384)
                if not block:
                    break
                archive.write(block)
                total += len(block)
                require(total <= RAW_LIMIT, 'SSE raw envelope')
                pending.extend(block)
                while b'\n' in pending:
                    raw_line, _, rest = pending.partition(b'\n')
                    pending = bytearray(rest)
                    line = raw_line[:-1] if raw_line.endswith(b'\r') else raw_line
                    if line.startswith(b':'):
                        continue
                    if line:
                        require(line.startswith(b'data: '), 'unexpected SSE field')
                        frame.append(line[6:])
                        continue
                    if not frame:
                        continue
                    require(len(frame) == 1, 'multi-line SSE data')
                    semantic = acc.feed(frame[0].decode('utf8', 'strict'))
                    frame = []
                    if semantic and 'first_semantic_token_seconds' not in result:
                        result['first_semantic_token_seconds'] = time.monotonic() - started
                    with c.lock:
                        if self is c.victim and c.should_cancel():
                            c.cancel(sock, response)
                            result.update(cancelled_by_client=True, events_at_close=acc.events,
                                          close_seconds=time.monotonic() - started)
                            return
            require(not frame and not pending, 'incomplete SSE frame')


class Coordinator:
    def __init__(self, args, rows, directory, victim_agent):
        self.host, self.port = endpoint(args.base_url)
        self.model, self.timeout = args.model, args.request_timeout
        self.after = args.cancel_after_events
        self.server_log = args.server_log
        self.lock = threading.Lock()
        self.start = threading.Event()
        self.streams = [Stream(self, row, directory / f"agent-{row['agent']}") for row in rows]
        self.victim = next((s for s in self.streams if s.agent == victim_agent), None)
        self.close = None

    def survivors(self):
        return [s for s in self.streams if s is not self.victim]

    def should_cancel(self):
        """Victim has K events and every survivor is mid-decode (has events, no finish yet)."""
        if self.close is not None or self.victim.accumulator.events < self.after:
            return False
        return all(s.accumulator.events >= 1 and s.accumulator.finish is None and not s.finished
                   for s in self.survivors())

    def cancel(self, sock, response):
        self.close = {'log_offset': log_size(self.server_log), 'monotonic_ns': time.monotonic_ns(),
                      'survivor_events': {s.agent: s.accumulator.events for s in self.survivors()}}
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        response.close()
        sock.close()

    def run(self):
        for stream in self.streams:
            stream.start()
        self.start.set()
        for stream in self.streams:
            stream.join()
        return {s.agent: s.result for s in self.streams}


def run_pass(args, rows, directory, victim_agent):
    before = cancellations(args.base_url)
    start = log_size(args.server_log)
    coordinator = Coordinator(args, rows, directory, victim_agent)
    agents = coordinator.run()
    after, deadline = cancellations(args.base_url), time.monotonic() + args.counter_wait_seconds
    while victim_agent is not None and coordinator.close is not None and after == before and time.monotonic() < deadline:
        time.sleep(0.25)
        after = cancellations(args.base_url)
    result = {'agents': {f'agent-{a}': r for a, r in agents.items()}, 'close': coordinator.close,
              'cancel_counter': {'before': before, 'after': after, 'delta': after - before},
              'log_offsets': {'start': start, 'end': log_size(args.server_log)}}
    if args.c2_anchor_depth is not None and coordinator.close is not None:
        try:
            while True:
                require(cancellations(args.base_url) == before + 1, 'C2 cancellation interval is not unique')
                end = log_size(args.server_log)
                lines = log_lines(args.server_log, coordinator.close['log_offset'], end)
                if any(RECOVERY.search(line) for _, line in lines):
                    result['recovery'] = cancellation_recovery(lines, coordinator.close['log_offset'])
                    result['log_offsets']['end'] = end
                    break
                require(time.monotonic() < deadline, 'expected one owned cancellation recovery, observed 0 before deadline')
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        except Invalid as error:
            result['recovery_error'] = str(error)
            result['log_offsets']['end'] = log_size(args.server_log)
    return result


# ----------------------------------------------------------------------------- verdict

def prefix_diff(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(cancel_pass, control_pass, victim):
    """Survivor equality and the victim's partial output as a prefix of its control output."""
    survivors = {}
    for name, result in cancel_pass['agents'].items():
        if name == f'agent-{victim}':
            continue
        left, right = result['output'], control_pass['agents'][name]['output']
        survivors[name] = {'exact': left == right,
                           'first_difference': {k: prefix_diff(json.dumps(left[k]), json.dumps(right[k]))
                                                for k in left if left[k] != right[k]}}
    partial = cancel_pass['agents'][f'agent-{victim}']['output']
    full = control_pass['agents'][f'agent-{victim}']['output']
    prefix = (full['content'].startswith(partial['content'])
              and full['reasoning_content'].startswith(partial['reasoning_content'])
              and len(partial['tool_calls']) <= len(full['tool_calls'])
              and all(f['name'].startswith(p['name']) and f['arguments'].startswith(p['arguments'])
                      for p, f in zip(partial['tool_calls'], full['tool_calls'])))
    return {'survivors_exact': all(s['exact'] for s in survivors.values()), 'survivors': survivors,
            'victim_partial_is_prefix_of_control': prefix,
            'expectation': 'exact on bitwise/cohort-invariant routes; reported, not gated'}


def judge(receipt):
    """Fills status and reasons. Invalid = the cancel window never opened (no claim either way)."""
    reasons, invalid = [], []
    cancel_pass = receipt['passes']['cancel']
    close = cancel_pass['close']
    if close is None:
        invalid.append('cancel window missed: the victim never had K events while all survivors were mid-decode')
    for name, result in cancel_pass['agents'].items():
        # An uncancelled victim (missed window) must still have produced a valid stream.
        if not result.get('cancelled_by_client') and not result.get('valid'):
            reasons.append(f"cancel pass {name} invalid: {result.get('error')}")
    if close is not None:
        if cancel_pass['cancel_counter']['delta'] != 1:
            reasons.append(f"hf2q_sse_cancellations delta {cancel_pass['cancel_counter']['delta']} != 1")
        joint = receipt['joint_before_close']
        if joint is None or joint['width'] < 2:
            reasons.append('no joint decode turn of width >= 2 before the close')
    if receipt['widths']['whole_run']['turns'] == 0:
        reasons.append('no joint decode turn lines in the server log (debug target not enabled, or joint decode off)')
    if receipt['problem_log_line_count']:
        reasons.append(f"{receipt['problem_log_line_count']} ERROR/poison server log lines during the run")
    if not receipt['server_ready_after']:
        reasons.append('server not ready after the run')
    control = receipt['passes'].get('control')
    if control is not None:
        for name, result in control['agents'].items():
            if not result.get('valid'):
                reasons.append(f"control pass {name} invalid: {result.get('error')}")
        if control['cancel_counter']['delta'] != 0:
            reasons.append('control pass changed hf2q_sse_cancellations')
    if receipt.get('c2_anchor_depth') is not None and close is not None:
        if 'recovery' not in cancel_pass:
            reasons.append('missing unique acknowledged cancellation recovery')
        retry = receipt['passes'].get('retry')
        if retry is None:
            reasons.append('missing owned cancellation retry')
        else:
            victim = retry['agents'][f"agent-{receipt['cancel']['agent']}"]
            if not victim.get('valid') or victim.get('first_semantic_token_seconds') is None:
                reasons.append('retry lacks valid semantic SSE token time')
            if retry['cancel_counter']['delta'] != 0:
                reasons.append('retry changed hf2q_sse_cancellations')
    receipt['reasons'] = reasons + invalid
    receipt['status'] = FAIL if reasons else INVALID if invalid else PASS


def run(args):
    out = Path(args.output).absolute()
    require(not out.exists(), 'output must be a new directory')
    require(args.cancel_agent in args.agents, '--cancel-agent must be one of --agents')
    require(len(args.agents) == 4 and len(set(args.agents)) == 4, 'four distinct agents')
    require(args.cancel_after_events >= 1, '--cancel-after-events must be positive')
    endpoint(args.base_url)
    if args.c2_anchor_depth is not None:
        require(args.family in ('qwen38', 'qwen36'), 'C2 recovery evidence requires Qwen')
    rows = load_requests(args.requests, args.turn, args.agents, args.model)
    require(ready(args.base_url), 'server not ready')
    warmup_rows = (load_requests(args.requests, args.warmup_turn, args.agents, args.model)
                   if args.c2_anchor_depth is not None else None)
    selected_requests = rows + (warmup_rows or [])
    repair_sources = (repair_workload_sources(args.requests, args.agents, selected_requests)
                      if args.family == 'qwen36' else None)
    out.mkdir(parents=True)
    receipt = {'schema': SCHEMA, 'status': None, 'family': args.family, 'base_url': args.base_url,
               'model': args.model, 'server_pid': args.server_pid, 'server_log': str(args.server_log),
               'c2_recovery_schema': 'hf2q.c2.cancel-recovery.v1' if args.c2_anchor_depth is not None else None,
               'source_sha256': sha256_bytes(Path(__file__).read_bytes()),
               'c2_anchor_depth': args.c2_anchor_depth,
               'source_repair_workload': repair_sources,
               'requests': {'directory': str(args.requests), 'turn': args.turn,
                            'agents': {f"agent-{r['agent']}": {k: r[k] for k in ('path', 'recorded_sha256', 'replayed_sha256')}
                                       for r in rows}},
               'cancel': {'agent': args.cancel_agent, 'after_semantic_events': args.cancel_after_events},
               'passes': {}, 'independent_review_required': True}
    try:
        if warmup_rows is not None:
            configured = [line for _, line in log_lines(args.server_log, 0, log_size(args.server_log))
                          if 'Qwen35 multi-depth anchor store configured' in line]
            require(len(configured) == 1, 'C2 needs exactly one resolved worker anchor configuration')
            fields = dict(FIELD.findall(configured[0]))
            require(fields.get('anchor_depth') == str(args.c2_anchor_depth),
                    'declared C2 depth differs from actual resolved worker configuration')
            warmup = run_pass(args, warmup_rows, out / 'warmup', None)
            receipt['passes']['warmup'] = warmup
            receipt['warmup_turn'] = args.warmup_turn
            require(warmup['cancel_counter']['delta'] == 0
                    and all(row.get('valid') for row in warmup['agents'].values()),
                    'C2 committed-prefix warmup did not complete cleanly')
            warmup['anchor_state_after'] = anchor_idle_state(args.base_url)
            require(all(count > 0 for count in warmup['anchor_state_after']['committed_by_slot'].values()),
                    'warmup did not publish a committed anchor in every slot')
        receipt['passes']['cancel'] = run_pass(args, rows, out / 'cancel', args.cancel_agent)
        cancel_pass = receipt['passes']['cancel']
        if args.c2_anchor_depth is not None and cancel_pass['close'] is not None:
            require(cancel_pass['cancel_counter']['delta'] == 1, 'C2 cancellation interval is not unique')
            require('recovery' in cancel_pass, cancel_pass.get('recovery_error', 'missing recovery completion'))
            cancel_pass['anchor_state_after'] = anchor_idle_state(args.base_url)
            victim_rows = [row for row in rows if row['agent'] == args.cancel_agent]
            receipt['passes']['retry'] = run_pass(args, victim_rows, out / 'retry', None)
            receipt['passes']['retry']['anchor_state_after'] = anchor_idle_state(args.base_url)
        if args.control:
            receipt['passes']['control'] = run_pass(args, rows, out / 'control', None)
        receipt['server_ready_after'] = ready(args.base_url)
        cancel_pass = receipt['passes']['cancel']
        start = min(p['log_offsets']['start'] for p in receipt['passes'].values())
        end = max(p['log_offsets']['end'] for p in receipt['passes'].values())
        lines = log_lines(args.server_log, start, end)
        turns = joint_turns(lines, args.family)
        close = cancel_pass['close']
        cancel_end = cancel_pass['log_offsets']['end']
        before = [t for t in turns if close is not None
                  and cancel_pass['log_offsets']['start'] < t['end_offset'] <= close['log_offset']]
        after = [t for t in turns if close is not None and close['log_offset'] < t['end_offset'] <= cancel_end]
        receipt['joint_before_close'] = before[-1] if before else None
        receipt['joint_after_close_first'] = after[0] if after else None
        receipt['widths'] = {'whole_run': histogram(turns, args.family),
                             'cancel_pass': histogram([t for t in turns if cancel_pass['log_offsets']['start'] < t['end_offset'] <= cancel_end], args.family),
                             'cancel_pass_after_close': histogram(after, args.family)}
        if 'control' in receipt['passes']:
            control = receipt['passes']['control']['log_offsets']
            receipt['widths']['control_pass'] = histogram(
                [t for t in turns if control['start'] < t['end_offset'] <= control['end']], args.family)
            receipt['control_comparison'] = compare(cancel_pass, receipt['passes']['control'], args.cancel_agent)
        if args.family == 'deepseek4':
            receipt['deepseek_cancelled_log_lines'] = sum(
                1 for end_offset, line in lines if 'DeepSeek-V4 request cancelled' in line and end_offset <= cancel_end)
        problems = problem_lines(lines)
        receipt['problem_log_line_count'] = len(problems)
        receipt['problem_log_lines'] = problems[:20]
        judge(receipt)
    except (Invalid, OSError) as error:
        receipt.update(status=FAIL, reasons=[f'{type(error).__name__}: {error}'])
    save(out / 'summary.json', receipt)
    print(json.dumps({'summary': str(out / 'summary.json'), 'status': receipt['status'],
                      'reasons': receipt['reasons']}, indent=2))
    return 0 if receipt['status'] == PASS else 1


def widths(args):
    lines = log_lines(args.server_log, 0, log_size(args.server_log))
    turns = joint_turns(lines, args.family)
    print(json.dumps({'schema': WIDTHS_SCHEMA, 'family': args.family, 'server_log': str(args.server_log),
                      **histogram(turns, args.family)}, indent=2))
    return 0 if turns else 1


def agents(text):
    return [int(part) for part in text.split(',')]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    smoke = sub.add_parser('run', help='cancel smoke against a running server')
    smoke.add_argument('--base-url', required=True, help='http://127.0.0.1:<port>/v1')
    smoke.add_argument('--model', required=True, help='served model id; must equal the recorded requests')
    smoke.add_argument('--family', required=True, choices=sorted(FAMILIES))
    smoke.add_argument('--requests', required=True, type=Path,
                       help='recorded four-agent client directory holding agent-N/turn-K/request.json')
    smoke.add_argument('--turn', type=int, default=2)
    smoke.add_argument('--agents', type=agents, default=[0, 1, 2, 3])
    smoke.add_argument('--cancel-agent', type=int, default=0)
    smoke.add_argument('--cancel-after-events', type=int, default=16,
                       help='semantic SSE events the victim must receive before the close')
    smoke.add_argument('--server-log', required=True, type=Path, help='the running server\'s stderr log')
    smoke.add_argument('--server-pid', type=int, help='recorded only')
    smoke.add_argument('--no-control', dest='control', action='store_false', help='skip the no-cancel control pass')
    smoke.add_argument('--request-timeout', type=float, default=900)
    smoke.add_argument('--counter-wait-seconds', type=float, default=30)
    smoke.add_argument('--c2-anchor-depth', type=int, choices=(1, 4),
                       help='require actual owned recovery and victim-only retry semantic TTFT for this declared C2 arm')
    smoke.add_argument('--warmup-turn', type=int, default=1,
                       help='C2 only: recorded predecessor turn completed before opening the cancellation interval')
    smoke.add_argument('--output', required=True, type=Path, help='new receipt directory')
    trace = sub.add_parser('widths', help='joint decode row-width histogram of a server log')
    trace.add_argument('--family', required=True, choices=sorted(FAMILIES))
    trace.add_argument('--server-log', required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args) if args.command == 'run' else widths(args)
    except (Invalid, OSError) as error:
        print(json.dumps({'status': FAIL, 'error': f'{type(error).__name__}: {error}'}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
