"""ADR-063 Q1 endpoint supervisor, one family case per invocation, T0 first.

Parametrized copy of the reviewed Gemma five-axis endpoint supervisor
(`adr063-q1-gemma-endpoints-supervisor-disabled-v1.py`, plus the witness-on
row validation of `q1-gemma-routing1511-t0-on-actual-v1/supervisor.py`).
Changes:
  1. every constant comes from one sha-pinned bindings.json (BINDINGS and
     BINDINGS_SHA256 below are set in the per-family instance copy); its bytes
     are read once, and the hashed bytes are the ones parsed and copied;
  2. filter, packet env var, completion marker and its stream are per family;
  3. expected logit/cache record counts are derived from the packet, not the
     literal Gemma 5616/576;
  4. `first_case` must be the family's T0 (`T0ColdSerial`) endpoint case;
  5. a `"kind": "localizer"` case runs one single-lane stage localizer packet
     (no trajectory, no 4x144 logits) with its own filter/env/marker;
  6. SIGHUP and SIGQUIT are handled like SIGTERM/SIGINT, so a terminal hangup
     still drains the child's process group;
  7. a `"kind": "multistore"` case runs one packet that writes several stores
     under output/ (e.g. a four-lane stage localizer beside its endpoints).
     Its extent (lanes, host_payload_limit, expected_stores) comes from the
     packet, or from a bound `manifest` file of record (the packet then has
     no expected_stores, and any lanes/host_payload_limit it keeps must
     equal the manifest's); every store must be complete and equal its
     frozen extent; closed registry `checks` own the family-specific
     receipt rules.
The foreign-process denylist and the listener ports are read from the reviewed
Gemma supervisor source (sha-pinned) so the reviewed list is reused verbatim.
No physical-admission, performance, float-policy or whole-Q1 qualification.
"""
from pathlib import Path
import argparse, ast, builtins, csv, errno, hashlib, importlib.util, json, os, re, shutil, signal, subprocess, symtable, sys, time

BINDINGS = None          # absolute path of the family bindings.json (instance copy only)
BINDINGS_SHA256 = None   # its sha256 (instance copy only)
SCHEMA = 'hf2q.q1.endpoint-supervisor-bindings.v1'
REVIEW_SCHEMA = 'hf2q.q1.endpoint-operating-review.v1'
AMBIENT = ('HF2Q_', 'MLX_', 'MTL_', 'METAL_', 'DYLD_', 'RUST_TEST_')
NULLABLE_IDENTITY = ('native_checksum',)  # filled from the composition comparison key
# Qwen3.8 records no active64 restore (dense, boundaries 128/144 only).
RESTORE64_FIELD = {'gemma': 'active64_exact_copy', 'qwen38': None,
                   'qwen36': 'qwen36_restore64_exact', 'deepseek': 'full_active64_exact_restore'}
PLANNED_FIELD = {'gemma': 'planned_tensor_payload', 'qwen38': 'planned_tensor_payload_bytes',
                 'qwen36': 'planned_tensor_payload_bytes', 'deepseek': 'planned_capture_bytes'}
FIRST_TRAJECTORY = 'T0ColdSerial'  # every family's T0
HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)
LOCALIZER_SCHEMA = 'hf2q.adr063.q1.stage-localizer.v1'
# The per-case operating review is passed on the command line (--operating-review), never pinned in the
# bindings: it must name BINDINGS_SHA256 and this copy's sha256, so pinning it there would be a hash cycle.
LOCALIZER_CASE_KEYS = {'kind', 'out', 'packet', 'filter', 'packet_env', 'completion_marker', 'marker_stream'}
MULTISTORE_CASE_KEYS = LOCALIZER_CASE_KEYS | {'stores', 'checks'}
MULTISTORE_OPTIONAL_KEYS = {'manifest', 'completion_line'}  # manifest: bound extent file; completion_line: exact fixed line
MULTISTORE_EXTENT_KEYS = {'lanes', 'host_payload_limit', 'expected_stores'}
# Rust-side localizer token lengths: Gemma 128 prompt + 16 forced; DS/Qwen 64 or 128.
LOCALIZER_TOKENS = {'gemma': (144,), 'qwen38': (64, 128), 'qwen36': (64, 128), 'deepseek': (64, 128)}
GEMMA_DEFAULT_WARM_KERNEL = 'VecHybrid'  # shipped since #544; packets must declare any other
COHORT_SCHEMA = 'hf2q.adr063.q1.cohort-stage-localizer.v1'
COHORT_ROUTES = {'T0ColdSerial': 'SerialAll128PrefillSingleRowDecode', 'T2ActualCohort': 'PackedPrefill512OwnersBatchedDecodeRows'}
INTERRUPTED = 0


def require(v, why):
    if not v: raise ValueError(why)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def save(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')
def bind(p): return dict(path=str(p), sha256=sha(p))
def bound(b, why):
    require(type(b) is dict and set(b) == {'path', 'sha256'} and Path(b['path']).is_absolute() and sha(b['path']) == b['sha256'], why)
    return Path(b['path'])
def checked_bytes(path, pinned, why):
    """Read once; the hashed bytes are the bytes the caller parses."""
    data = Path(path).read_bytes(); require(hashlib.sha256(data).hexdigest() == pinned, why); return data
def load(b, why):
    require(type(b) is dict and set(b) == {'path', 'sha256'} and Path(b['path']).is_absolute(), why)
    return json.loads(checked_bytes(b['path'], b['sha256'], why))
def store_dir(rel):
    """A store directory relative to output/: '.' or plain descending names only."""
    require(type(rel) is str and rel and not Path(rel).is_absolute() and (rel == '.' or all(p not in ('', '.', '..') for p in rel.split('/'))), 'store directory inside output/')
    return rel
def case_plan(b, name):
    """Closed launch plan: an endpoint trajectory, a single-lane localizer or a multistore case."""
    first = b['cases'].get(b['first_case'])
    require(type(first) is dict and first.get('kind', 'endpoint') == 'endpoint' and first.get('trajectory') == FIRST_TRAJECTORY, 'first_case is the family T0 (T0ColdSerial) endpoint case')
    require(name in b['cases'], 'declared case'); c = b['cases'][name]; kind = c.get('kind', 'endpoint')
    if kind == 'endpoint':
        require(type(c.get('trajectory')) is str, 'endpoint trajectory')
        return dict(kind=kind, mode=c['trajectory'], filter=c.get('filter', b['filter']), packet_env=b['packet_env'], completion_marker=b['completion_marker'], marker_stream=b['marker_stream'])
    launch = all(type(c.get(k)) is str and c[k] for k in ('filter', 'packet_env', 'completion_marker')) and c.get('marker_stream') in ('stdout', 'stderr')
    if kind == 'multistore':
        require(MULTISTORE_CASE_KEYS <= set(c) <= MULTISTORE_CASE_KEYS | MULTISTORE_OPTIONAL_KEYS and launch, 'closed multistore case: own filter/env/marker, no trajectory')
        stores = c['stores']; checks = c['checks']
        # `{}` = no trace store (e.g. a preparation phase); the registry checks then own the whole receipt.
        require(type(stores) is dict and all(type(k) is str and k for k in stores) and len({store_dir(v) for v in stores.values()}) == len(stores), 'distinct named stores under output/')
        require(type(checks) is list and checks and len(set(checks)) == len(checks) and all(type(x) is str and x in MULTISTORE_CHECKS for x in checks), 'closed registry checks')
        line = c.get('completion_line')
        require(line is None or (type(line) is str and c['completion_marker'] in line), 'fixed completion line names the marker')
        require('manifest' not in c or (type(c['manifest']) is dict and set(c['manifest']) == {'path', 'sha256'} and type(c['manifest']['path']) is str and Path(c['manifest']['path']).is_absolute()), 'bound extent manifest at an absolute path')
        return dict(kind=kind, mode='multistore', filter=c['filter'], packet_env=c['packet_env'], completion_marker=c['completion_marker'], marker_stream=c['marker_stream'], stores=dict(stores), checks=list(checks), completion_line=line)
    require(kind == 'localizer' and set(c) == LOCALIZER_CASE_KEYS and launch, 'closed localizer case: own filter/env/marker, no trajectory')
    return dict(kind=kind, mode='localizer', filter=c['filter'], packet_env=c['packet_env'], completion_marker=c['completion_marker'], marker_stream=c['marker_stream'])
def packet_scope(plan, packet):
    """The disabled packet matches its case kind: an endpoint carries its trajectory and no multistore lanes/stores
    (its own `host_payload_limit` is an ordinary endpoint field), a localizer no trajectory."""
    if plan['kind'] == 'endpoint': return packet.get('trajectory') == plan['mode'] and not {'lanes', 'expected_stores', 'stage_localizer'} & set(packet)
    if plan['kind'] == 'localizer': return 'trajectory' not in packet
    return True  # multistore: the registry checks own the trajectory
def multistore_extent(case, packet, manifest_bytes):
    """(lanes, host_payload_limit, expected_stores): the packet's, or, in manifest form, the bound manifest's.

    Manifest form: the manifest is the extent of record. The packet may not carry `expected_stores`; a packet
    `lanes` or `host_payload_limit` it keeps for its own runner must equal the manifest's value."""
    require((manifest_bytes is None) == ('manifest' not in case), 'manifest case key and bound bytes agree')
    if manifest_bytes is None:
        require(MULTISTORE_EXTENT_KEYS <= set(packet), 'packet-form extent: lanes, host_payload_limit, expected_stores')
        extent = {k: packet[k] for k in MULTISTORE_EXTENT_KEYS}
    else:
        extent = json.loads(manifest_bytes)
        require(type(extent) is dict and set(extent) == MULTISTORE_EXTENT_KEYS, 'closed manifest: lanes, host_payload_limit, expected_stores')
        require('expected_stores' not in packet, 'manifest form: the packet carries no expected_stores')
        require(all(packet[k] == extent[k] for k in ('lanes', 'host_payload_limit') if k in packet), 'manifest form: packet lanes/host_payload_limit equal the manifest')
    lanes = extent['lanes']
    require(type(lanes) is list and lanes and lanes == sorted(set(lanes)) and all(type(l) is int and 0 <= l < 4 for l in lanes), 'declared distinct workload lanes')
    require(type(extent['host_payload_limit']) is int and extent['host_payload_limit'] > 0 and type(extent['expected_stores']) is dict, 'bounded extent')
    return extent
def interrupt(sig, frame):
    global INTERRUPTED
    INTERRUPTED = sig
def raw(out, label, argv, timeout=2):
    t = time.monotonic_ns(); p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    r = dict(argv=argv, begin_ns=t, end_ns=time.monotonic_ns(), exit=p.returncode, stdout=p.stdout, stderr=p.stderr); save(out / (label + '.json'), r); return r


def reviewed_denylist(source):
    """(names, prefixes, ports) literally as written in the reviewed quiet()."""
    fn = [n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.FunctionDef) and n.name == 'quiet']
    require(len(fn) == 1, 'reviewed quiet() source')
    names = prefixes = ports = None
    for n in ast.walk(fn[0]):
        if isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.NotIn) and isinstance(n.comparators[0], ast.Tuple):
            names = ast.literal_eval(n.comparators[0])
        elif isinstance(n, ast.Call) and getattr(n.func, 'attr', None) == 'startswith' and n.args and isinstance(n.args[0], ast.Tuple):
            prefixes = ast.literal_eval(n.args[0])
        elif isinstance(n, ast.For) and isinstance(n.iter, ast.Tuple):
            ports = ast.literal_eval(n.iter)
    require(names and prefixes and ports and all(type(x) is str for x in names + prefixes) and all(type(x) is int for x in ports), 'reviewed denylist/ports')
    return names, prefixes, ports


def quiet(out, phase, deny):
    names, prefixes, ports = deny
    p = raw(out, phase + '-processes', ['ps', '-axo', 'pid=,ucomm=,args=']); require(p['exit'] == 0 and not p['stderr'], 'process inventory')
    for line in p['stdout'].splitlines():
        row = line.strip().split(None, 2); require(len(row) == 3, 'process row'); name = Path(row[1]).name
        require(name not in names and not name.startswith(prefixes), 'other model/helper/build:' + name)
    for port in ports:
        p = raw(out, phase + '-listener-' + str(port), ['lsof', '-nP', '-iTCP:' + str(port), '-sTCP:LISTEN', '-Fp']); require(p['exit'] == 1 and not p['stdout'] and not p['stderr'], 'listener occupied/unavailable')


def drain(p):
    errors = []; empty = False
    for sig, grace in ((signal.SIGTERM, 3), (signal.SIGKILL, 5)):
        try: os.killpg(p.pid, sig)
        except ProcessLookupError: pass
        except OSError as e: errors.append(repr(e))
        try: p.wait(timeout=grace)
        except subprocess.TimeoutExpired: pass
        except OSError as e: errors.append('wait:' + repr(e))
        try: os.killpg(p.pid, 0)
        except ProcessLookupError: empty = True
        except OSError as e: errors.append('group:' + repr(e))
        if p.poll() is not None and empty: break
    return dict(reaped=p.poll() is not None, group_empty=empty, errors=errors, exit=p.returncode)


def terminal_esrch(ledger, value, pid, child_exit, prior_fresh):
    require(ledger['exit'] == 3 and not ledger['stderr'] and type(value) is dict and set(value) == {'available', 'pid', 'errno'} and value['available'] is False and type(value['pid']) is int and value['pid'] == pid and type(value['errno']) is int and value['errno'] == errno.ESRCH and child_exit is not None and prior_fresh is True, 'only exact recorded terminal ESRCH after fresh validated owned ledger')
    return True


def disk_bytes(root): return sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0


def coordinate(x): return tuple(x['at'][k] for k in ('logical_step', 'layer', 'stage', 'agent', 'logical_row_start', 'segment'))


def checked_store(root, identity):
    """Complete, unique root/trace.json for the composed identity."""
    trace = json.loads((root / 'trace.json').read_text())
    require(not trace['coverage']['errors'] and trace['identity'] == identity, 'complete trace for the composed identity')
    require(trace['acquisition']['observed_model_opens'] >= trace['acquisition']['required_model_opens'] >= 1, 'actual descriptor opens')
    captured, expected = trace['coverage']['captured'], trace['coverage']['expected']
    require(len(captured) == len(expected) and len({coordinate(x) for x in captured}) == len(captured), 'complete unique capture coordinates')
    return captured, expected


def checked_trace(out, identity):
    """execution.json + complete, unique trace.json for the composed identity."""
    execution = json.loads((out / 'output/execution.json').read_text())
    require(execution['whole_q1_qualified'] is False, 'no whole-Q1 claim')
    captured, expected = checked_store(out / 'output', identity)
    return execution, captured, expected


def require_gemma_warm_kernel(family, execution, packet):
    if family == 'gemma':
        require(execution['warm_live_kernel'] == packet.get('warm_live_kernel', GEMMA_DEFAULT_WARM_KERNEL), 'resolved Gemma warm kernel = packet declaration (absent = shipped VecHybrid)')


def validate_localizer(out, b, plan, packet, identity, policy):
    """One real workload lane; the packet's `expected` is the whole extent."""
    family = b['family']; execution, captured, expected = checked_trace(out, identity)
    require(execution['schema'] == LOCALIZER_SCHEMA and execution['packet_sha256'] == sha(out / 'packet.json') and execution['rerun_id'] == packet['rerun_id'] and execution['records'] == len(captured), 'localizer receipt for this exact packet')
    require(len(expected) == len(packet['expected']) and {coordinate(x) for x in expected} == {coordinate(x) for x in packet['expected']}, 'packet manifest is the complete localizer extent')
    agent, tokens, lanes = packet['agent'], packet['tokens'], packet['workload']['lanes']
    require(type(agent) is int and len(lanes) == 4 and 0 <= agent < 4 and len(tokens) in LOCALIZER_TOKENS[family] and tokens == lanes[agent][:len(tokens)], 'one real workload lane prefix')
    require(all(x['at']['agent'] == 'agent-' + str(agent) for x in captured), 'every record on the localized lane')
    total = sum(x['logical_bytes'] for x in captured)
    require(total <= packet['host_payload_limit'] and total <= policy['phase_payload_upper_bytes'], 'captured bytes <= packet and policy bounds')
    line = plan['completion_marker'] + ' records=' + str(len(captured)) + ' whole_q1_qualified=false'
    require(line in (out / plan['marker_stream']).read_text().splitlines(), 'exact localizer completion line')
    require_gemma_warm_kernel(family, execution, packet)
    if family == 'gemma': require(execution['prefill_chunks'] == packet.get('prefill_chunks', [128]), 'executed Gemma prefill route = packet')
    return dict(execution=execution, captured_records=len(captured), captured_bytes=total, whole_q1_qualified=False)


def validate_endpoint(out, b, case, packet, identity, policy):
    """Packet-derived completeness; the family helper owns the numeric checks."""
    execution, captured, expected = checked_trace(out, identity)
    return endpoint_evidence(out, b, b['cases'][case]['trajectory'], packet, execution, captured, expected, identity, policy)


def endpoint_evidence(out, b, trajectory, packet, execution, captured, expected, identity, policy):
    family = b['family']
    require_gemma_warm_kernel(family, execution, packet)
    lanes = packet['workload']['lanes']; rows = len(lanes[0]); require(len(lanes) == 4 and all(len(l) == rows == 144 for l in lanes), 'four144 packet workload')
    logits = [x for x in captured if x['at']['stage'] == 'Logits']
    require({(x['at']['agent'], x['at']['logical_row_start']) for x in logits} == {('agent-' + str(l), p) for l in range(4) for p in range(rows)} and len(logits) == 4 * rows, 'every lane x row logit record')
    if 'expected_cache' in packet:
        cache = {coordinate(x) for x in expected if x['at']['stage'] != 'Logits'}
        require(cache == {coordinate(x) for x in packet['expected_cache']}, 'packet cache manifest is the complete non-logit extent')
    total = sum(x['logical_bytes'] for x in captured)
    require(total == execution[PLANNED_FIELD[family]] <= policy['phase_payload_upper_bytes'], 'actual captured bytes = planned <= policy')
    if trajectory in ('T1Resume64', 'T6CancelRestore') and RESTORE64_FIELD[family]:
        exact = execution[RESTORE64_FIELD[family]]
        require(type(exact) is dict and set(exact) == {'Ok'} and type(exact['Ok']) is int and exact['Ok'] > 0, 'actual complete encoded64 exact restore')
    if family == 'gemma' and trajectory == 'T6CancelRestore':
        require(len(execution['actual_cancellation']) == 4 and {x['slot'] for x in execution['actual_cancellation']} == {0, 1, 2, 3} and all(x['real_receiver_closed_after_checked_advance'] is True and x['checked_prefix'] == 64 and x['actual_unpublished_end'] == 128 and x['recovered_cursor'] == 64 and x['retained_tokens'] == 64 and x['scheduler_in_flight_after'] == 0 and x['supervisor_healthy'] is True for x in execution['actual_cancellation']), 'actual four-owning-cancellation recovery')
    witness = b.get('routing_witness')
    if witness:
        layers = witness['layers']; seen = set(); end = None; path = out / 'output/routing-witness.jsonl'
        with path.open() as file:
            header = json.loads(next(file)); require(header['kind'] == 'header' and header['identity'] == identity and header['qualified'] is False, 'same actual identity routing bank')
            for line in file:
                require(end is None, 'no data after routing end'); r = json.loads(line)
                if r['kind'] == 'end': end = r; continue
                require(r['kind'] == 'routing_row' and r['native_boundary_valid'] is True and r['selected_copy_valid'] is True and r['native_decode_error'] is None and r['qualified'] is False, 'actual native Scored row')
                at = r['at']; key = (at['namespace'], at['row']['lane'], at['row']['position'], at['layer'])
                require(key not in seen and (at['namespace'] == 'Canonical' or at['namespace'] in witness.get('extra_namespaces', ())), 'unique row in a declared namespace'); seen.add(key)
                require(all(type(x) is int and 0 <= x < 2**32 and (x & 0x7f800000) != 0x7f800000 for x in r['scores_f32_bits'] + r['selected_weights_f32_bits']), 'finite raw native values')
        canonical = {k[1:] for k in seen if k[0] == 'Canonical'}
        require(canonical == {(l, p, k) for l in range(4) for p in range(rows) for k in range(layers)} and end is not None and end['records'] == len(seen) and end['all_scored_rows_captured'] is True and end['full_Q1_qualified'] is False, 'complete canonical routing bank')
        require(path.stat().st_size <= witness['metadata_limit_bytes'], 'routing metadata quota')
    return dict(execution=execution, captured_records=len(captured), logit_records=len(logits), captured_bytes=total, whole_q1_qualified=False)


def validate_multistore(out, b, case, plan, packet, extent, identity, policy):
    """Every declared store complete and equal to its frozen extent; then the closed registry checks."""
    execution = json.loads((out / 'output/execution.json').read_text())
    require(execution['packet_sha256'] == sha(out / 'packet.json') and execution['whole_q1_qualified'] is False, 'multistore receipt for this exact packet')
    require(set(extent['expected_stores']) == set(plan['stores']), 'one frozen extent per declared store')
    agents = {'agent-' + str(l) for l in extent['lanes']}; stores = {}
    for name, rel in plan['stores'].items():
        captured, expected = checked_store(out / 'output' / store_dir(rel), identity); frozen = extent['expected_stores'][name]
        require(len(expected) == len(frozen) and {coordinate(x) for x in expected} == {coordinate(x) for x in frozen}, 'store ' + name + ' = frozen extent')
        require(all(x['at']['agent'] in agents for x in captured), 'store ' + name + ': every record on a declared lane')
        stores[name] = dict(captured=captured, expected=expected, bytes=sum(x['logical_bytes'] for x in captured))
    total = sum(s['bytes'] for s in stores.values()); records = sum(len(s['captured']) for s in stores.values())
    require(total <= extent['host_payload_limit'] and total <= policy['phase_payload_upper_bytes'], 'captured bytes <= extent and policy bounds')
    lines = (out / plan['marker_stream']).read_text().splitlines()
    if plan['completion_line']:
        require(lines.count(plan['completion_line']) == 1, 'fixed completion line exactly once')
    else:
        require(plan['completion_marker'] + ' records=' + str(records) + ' whole_q1_qualified=false' in lines, 'exact multistore completion line')
    evidence = dict(execution=execution, stores={n: dict(records=len(s['captured']), bytes=s['bytes']) for n, s in stores.items()}, captured_records=records, captured_bytes=total, whole_q1_qualified=False)
    for check in plan['checks']:
        evidence[check] = MULTISTORE_CHECKS[check](out, b, case, plan, packet, extent, execution, stores, identity, policy)
    return evidence


def check_gemma_loc4(out, b, case, plan, packet, extent, execution, stores, identity, policy):
    """Gemma four-lane stage localizer (#652): the unchanged T0/T2 endpoint store plus a four-lane stage store."""
    trajectory = packet.get('trajectory'); stage = packet.get('stage_localizer')
    require(b['family'] == 'gemma' and plan['stores'] == {'endpoint': '.', 'stages': 'stages'} and trajectory in COHORT_ROUTES and type(stage) is dict, 'Gemma All128/packed localizer stores')
    # The canonical routing bank is part of this route's evidence: never skipped (spec Part 3; only DriftAll128 may omit it).
    require(type(b.get('routing_witness')) is dict and type(packet.get('routing_witness')) is dict, 'cohort localizer requires the routing witness in bindings and packet')
    require(extent['lanes'] == [0, 1, 2, 3] and policy.get('lanes') == 4 and type(policy.get('stage_payload_upper_bytes')) is int, 'four-lane extent and resource policy')
    staged = stores['stages']
    require(execution['schema'] == COHORT_SCHEMA and execution['trajectory'] == trajectory and execution['stage_route'] == COHORT_ROUTES[trajectory] and execution['rerun_id'] == stage['rerun_id'] and execution['stage_store'] == 'stages' and execution['lanes'] == [0, 1, 2, 3] and execution['stage_records'] == len(staged['captured']), 'cohort localizer receipt for this route')
    require({x['at']['agent'] for x in staged['captured']} == {'agent-' + str(l) for l in range(4)}, 'stage records cover all four lanes')
    require(staged['bytes'] == execution['stage_planned_payload'] <= stage['host_payload_limit'] <= policy['stage_payload_upper_bytes'], 'stage bytes = planned <= packet <= four-lane policy')
    endpoint = stores['endpoint']
    evidence = endpoint_evidence(out, b, trajectory, packet, execution, endpoint['captured'], endpoint['expected'], identity, policy)
    return dict(trajectory=trajectory, stage_route=COHORT_ROUTES[trajectory], stage_records=len(staged['captured']), stage_bytes=staged['bytes'], endpoint_records=evidence['captured_records'], logit_records=evidence['logit_records'], endpoint_bytes=evidence['captured_bytes'])


# Closed registry of multistore check functions: (out, b, case, plan, packet, extent, execution, stores, identity, policy) -> evidence dict.
MULTISTORE_CHECKS = {'gemma_loc4': check_gemma_loc4}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--case', required=True)
    ap.add_argument('--operating-review', required=True, help='absolute path of the independent per-case operating review')
    args = ap.parse_args()
    require(sys.flags.isolated and __debug__ and BINDINGS is not None and BINDINGS_SHA256 is not None, 'PENDING sha-pinned bindings in the instance copy')
    bindings_bytes = checked_bytes(BINDINGS, BINDINGS_SHA256, 'sha-pinned bindings'); b = json.loads(bindings_bytes)
    require(b['schema'] == SCHEMA and b['final_approval'] is True and b['family'] in RESTORE64_FIELD and args.case in b['cases'], 'closed approved family bindings')
    require(not any(k.startswith(AMBIENT) for k in os.environ), 'ambient runtime override')
    plan = case_plan(b, args.case); case = b['cases'][args.case]; mode = plan['mode']; out = Path(case['out']); require(out.is_absolute() and not out.exists(), 'one fresh case only')
    outputs = [Path(c['out']) for c in b['cases'].values()]
    composition = load(b['composition'], 'composition binding'); review = load(b['acquisition_review'], 'acquisition review binding')
    require(review == dict(schema='hf2q.q1.acquisition-review.v1', verdict='Accepted', composition_sha256=b['composition']['sha256']), 'closed accepted composition')
    identity = composition['comparison_key']; require(identity['source_commit'] == b['source_commit'] and identity['binary_sha256'] == b['binary_sha256'] and identity['native_version'] == b['native_version'], 'compiled composition')
    binary = Path(b['binary']); require(sha(binary) == b['binary_sha256'], 'binary identity')
    build = load(b['build_receipt'], 'build receipt binding'); require(build['controls_exit'] == 0 and build['native_version'] == b['native_version'] and build['source_commit'] == b['source_commit'] and build['binary_sha256'] == b['binary_sha256'], 'root build identity')
    obs = b['observers']; sampler_source = bound(obs['sampler'], 'sampler'); parser_source = bound(obs['parser'], 'parser'); proc = bound(obs['proc'], 'proc ledger')
    require(sha('/usr/bin/meminfo') == obs['meminfo_sha256'] and sha('/usr/bin/osascript') == obs['thermal_sha256'], 'reviewed observation binaries')
    deny = reviewed_denylist(bound(obs['denylist_source'], 'reviewed denylist source'))
    policy = load(b['resource_policy'], 'resource policy binding'); require(policy['family'] == b['family'] and policy['physical_admission'] is False, 'family resource policy')
    require('operating_review' not in case and Path(args.operating_review).is_absolute(), 'operating review is a launch argument, not a bindings entry')
    operating_bytes = Path(args.operating_review).read_bytes(); operating = json.loads(operating_bytes); operating_sha = hashlib.sha256(operating_bytes).hexdigest()
    require(operating == dict(schema=REVIEW_SCHEMA, verdict='AcceptedForOneAttempt', family=b['family'], case=args.case, composition_sha256=b['composition']['sha256'], policy_sha256=b['resource_policy']['sha256'], bindings_sha256=BINDINGS_SHA256, supervisor_sha256=sha(__file__), physical_admission=False), 'closed per-case operating review; not physical fit')
    packet = load(case['packet'], 'disabled packet binding'); model = Path(packet['model']); expected_stat = composition['model_file']
    def stat_model():
        st = model.stat(); actual = dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size, mtime=st.st_mtime_ns // 10**9, mtime_ns=st.st_mtime_ns % 10**9, ctime=st.st_ctime_ns // 10**9, ctime_ns=st.st_ctime_ns % 10**9); require(actual == expected_stat, 'model stat changed; no new payload hash')
    stat_model()
    if args.case != b['first_case']:
        first = Path(b['cases'][b['first_case']]['out']); require(json.loads((first / 'receipt.json').read_text())['case_complete'] is True, b['first_case'] + ' must complete before another axis')
    require(packet['execution_approved'] is False and packet_scope(plan, packet), 'exact disabled family axis/localizer/multistore packet')
    manifest_bytes = checked_bytes(case['manifest']['path'], case['manifest']['sha256'], 'bound extent manifest') if plan['kind'] == 'multistore' and 'manifest' in case else None
    extent = multistore_extent(case, packet, manifest_bytes) if plan['kind'] == 'multistore' else None
    for k, v in list(packet['identity'].items()):
        if v is None: require(k in NULLABLE_IDENTITY, 'only native_checksum may be filled'); packet['identity'][k] = identity[k]
    require(packet['identity'] == identity, 'packet identity = composition comparison key')
    acquisition = dict(composition=b['composition'], independent_review=b['acquisition_review'])
    require(packet.get('acquisition') in (None, acquisition) or packet['acquisition']['composition'] == b['composition'], 'packet composition binding')
    packet.update(execution_approved=True, acquisition=acquisition, output=str(out / 'output'))
    os.umask(0o077); out.mkdir(mode=0o700); shutil.copy2(__file__, out / 'supervisor.py'); (out / 'bindings.json').write_bytes(bindings_bytes); subprocess.run(['/bin/cp', '-c', str(binary), str(out / 'test-binary')], check=True, timeout=30); shutil.copy2(parser_source, out / 'meminfo_interface.py'); save(out / 'resource-policy.json', policy); save(out / 'acquisition-composition.json', composition); save(out / 'acquisition-review.json', review); (out / 'operating-review.json').write_bytes(operating_bytes)
    save(out / 'packet.json', packet)
    if manifest_bytes is not None: (out / 'manifest.json').write_bytes(manifest_bytes)
    tree = ast.parse(sampler_source.read_text()); nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ('sample', 'proc_fields', 'ledger_ready')]; require(len(nodes) == 3, 'three exact reviewed observation functions')
    assignments = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}; POLICY = ast.literal_eval(assignments['POLICY']); POLICY.update(preload_free_floor_bytes=policy['preload_available_floor_bytes'], runtime_free_floor_bytes=policy['runtime_available_floor_bytes'], reserve_free_bytes=policy['reserve_bytes'], overshoot_margin_bytes=policy['reaction_overshoot_bytes'], process_rss_abort_bytes=policy['process_rss_limit_bytes'], process_footprint_abort_bytes=policy['process_footprint_limit_bytes'])
    spec = importlib.util.spec_from_file_location('q1_frozen_meminfo_parser', out / 'meminfo_interface.py'); parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
    host = b['host']; physical = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True, timeout=2)); require(physical == host['memsize_bytes'] and os.sysconf('SC_PAGE_SIZE') == host['page_bytes'], 'selected host geometry')
    namespace = dict(time=time, Path=Path, subprocess=subprocess, csv=csv, require=require, save=save, sha=sha, POLICY=POLICY, out=out, parser=parser, physical=physical, THERMAL_SHA=obs['thermal_sha256'], re=re); observation_code = ast.unparse(ast.Module(body=nodes, type_ignores=[])); tables = [symtable.symtable(observation_code, str(sampler_source), 'exec')]; globals_used = set()
    while tables:
        table = tables.pop(); tables.extend(table.get_children()); globals_used.update(x.get_name() for x in table.get_symbols() if x.is_referenced() and x.is_global())
    require(globals_used <= set(namespace) | set(dir(builtins)), 'missing observer globals:' + repr(globals_used - set(namespace) - set(dir(builtins))))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(sampler_source), 'exec'), namespace); sample = namespace['sample']
    deadline = policy['deadline_seconds']
    result = dict(family=b['family'], case=args.case, kind=plan['kind'], operating_review_sha256=operating_sha, case_complete=False, mode=mode, model_launch_attempted=False, physical_admission=False, performance_qualified=False, whole_q1_qualified=False); child = None
    for sig in HANDLED_SIGNALS: signal.signal(sig, interrupt)
    def available(x, owned): return x['physical']['free'] + x['physical']['cached_files'] - sum(disk_bytes(o) for o in owned)
    try:
        quiet(out, 'pre', deny); chip = raw(out, 'host-chip', ['sysctl', '-n', 'machdep.cpu.brand_string']); require(chip['exit'] == 0 and chip['stdout'].strip() == host['chip'] and not chip['stderr'], 'mandatory host chip'); epoch = time.monotonic_ns(); baseline = []
        for i in range(12):
            target = epoch + i * 500000000; time.sleep(max(0, (target - time.monotonic_ns()) / 1e9)); x = sample('baseline', i, target); baseline.append(x)
            require(x['pressure_raw'] == '1' and available(x, [o for o in outputs if o != out]) >= policy['preload_available_floor_bytes'], 'preload operating headroom/pressure (nothing else resident)')
        require(not INTERRUPTED, 'interrupted before load'); stat_model(); require(sha(binary) == b['binary_sha256'], 'prelaunch binary')
        selected_filter = plan['filter']
        command = [str(out / 'test-binary'), selected_filter, '--exact', '--ignored', '--nocapture', '--test-threads=1']; save(out / 'command.json', dict(argv=command, binary_sha256=b['binary_sha256'], source_commit=b['source_commit'], packet_sha256=sha(out / 'packet.json'), packet_env=plan['packet_env'], deadline_seconds=deadline, mode=mode, kind=plan['kind']))
        env = {k: v for k, v in os.environ.items() if not k.startswith(AMBIENT)}; env[plan['packet_env']] = str(out / 'packet.json')
        with (out / 'stdout').open('xb') as so, (out / 'stderr').open('xb') as se:
            quiet(out, 'immediate-prelaunch', deny); require(not INTERRUPTED, 'interrupted immediately before Popen'); spawn_ns = time.monotonic_ns(); started = time.monotonic(); child = subprocess.Popen(command, cwd=out, env=env, stdout=so, stderr=se, stdin=subprocess.DEVNULL, start_new_session=True); result.update(model_launch_attempted=True, pid=child.pid); i = 0; next_sample = time.monotonic_ns(); proc_identity = None; last_live_ledger = None; previous_sample_ns = None
            while child.poll() is None:
                require(not INTERRUPTED and time.monotonic() - started < deadline, 'interruption/case watchdog')
                now = time.monotonic_ns()
                if now >= next_sample:
                    x = sample('runtime', i, now); require(previous_sample_ns is None or x['start_ns'] - previous_sample_ns <= POLICY['max_sample_gap_ns'], 'stale runtime sample'); previous_sample_ns = x['start_ns']; require(x['pressure_raw'] == '1', 'runtime pressure')
                    require(available(x, outputs) >= policy['runtime_available_floor_bytes'], 'runtime operating floor'); require(x['swap_bytes'] - baseline[0]['swap_bytes'] <= policy['max_swap_growth_bytes'] and x['physical']['compressor'] - baseline[0]['physical']['compressor'] <= policy['max_compressor_growth_bytes'], 'swap/compressor growth')
                    ledger = raw(out, 'process-' + str(i), [str(proc), str(child.pid)], timeout=.5); value = json.loads(ledger['stdout'])
                    if ledger['exit'] == 0 and value.get('available') is True:
                        proc_identity = namespace['proc_fields'](value, child.pid, proc_identity)
                        live = dict(available=True, parent_start_ns=ledger['begin_ns'], parent_end_ns=ledger['end_ns'], parent_duration_ns=ledger['end_ns'] - ledger['begin_ns'], identity_validated=True)
                        require(namespace['ledger_ready'](live, POLICY, time.monotonic_ns(), after_ns=spawn_ns), 'fresh actual owned ledger required'); last_live_ledger = live
                        require(value['resident_size'] <= policy['process_rss_limit_bytes'] and value['phys_footprint'] <= policy['process_footprint_limit_bytes'], 'RSS/footprint operating limit')
                    else:
                        terminal_esrch(ledger, value, child.pid, child.poll(), namespace['ledger_ready'](last_live_ledger, POLICY, time.monotonic_ns(), after_ns=spawn_ns)); result['terminal_ledger_race'] = ledger
                    require(sum(disk_bytes(o) for o in outputs) <= policy['paired_disk_working_limit_bytes'], 'owned disk quota; no cold-cache refund'); i += 1; next_sample = now + 500000000  # fixed cadence from the sample start: the 1 s gap bound must not absorb the per-sample ledger/disk work
                time.sleep(.02)
        result.update(exit=child.returncode, elapsed_seconds=time.monotonic() - started, runtime_samples=i); require(proc_identity is not None, 'no validated live owned process ledger'); require(sum(disk_bytes(o) for o in outputs) <= policy['paired_disk_working_limit_bytes'], 'final owned disk quota'); require(child.returncode == 0 and result['elapsed_seconds'] <= deadline, 'actual helper failed/deadline')
        stdout = (out / 'stdout').read_text(); marked = (out / plan['marker_stream']).read_text()
        require('test result: ok. 1 passed; 0 failed; 0 ignored;' in stdout and plan['completion_marker'] in marked, 'exact helper pass and family completion marker')
        validate = dict(endpoint=lambda: validate_endpoint(out, b, args.case, packet, identity, policy),
                        localizer=lambda: validate_localizer(out, b, plan, packet, identity, policy),
                        multistore=lambda: validate_multistore(out, b, args.case, plan, packet, extent, identity, policy))
        result['helper_evidence'] = validate[plan['kind']]()
        stat_model(); require(sha(out / 'test-binary') == b['binary_sha256'] and sha(BINDINGS) == BINDINGS_SHA256, 'terminal source/bindings identity')
        for key in ('composition', 'acquisition_review', 'resource_policy', 'build_receipt'): bound(b[key], 'terminal ' + key)
        if manifest_bytes is not None: bound(case['manifest'], 'terminal extent manifest')
    except BaseException as e: result['failure'] = repr(e)
    finally:
        if child is not None: result['cleanup'] = drain(child)
        try:
            quiet(out, 'post', deny)
            if result.get('terminal_ledger_race'):
                post = sample('terminal-exit', 0, time.monotonic_ns())
                require(post['pressure_raw'] == '1' and available(post, outputs) >= policy['runtime_available_floor_bytes'], 'terminal-race fresh global headroom/pressure')
                require(post['swap_bytes'] - baseline[0]['swap_bytes'] <= policy['max_swap_growth_bytes'] and post['physical']['compressor'] - baseline[0]['physical']['compressor'] <= policy['max_compressor_growth_bytes'], 'terminal-race fresh global growth guard')
        except BaseException as e: result['post_failure'] = repr(e)
        result['case_complete'] = not result.get('failure') and not result.get('post_failure') and (child is None or (result.get('cleanup', {}).get('reaped') is True and result['cleanup']['group_empty'] and not result['cleanup']['errors'])) and 'helper_evidence' in result
        save(out / 'receipt.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'helper_evidence'}, indent=2)); raise SystemExit(0 if result['case_complete'] else 1)


if __name__ == '__main__': main()
