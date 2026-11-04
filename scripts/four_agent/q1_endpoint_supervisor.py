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
     receipt rules: gemma_loc4 (#652; legacy, plan-only and DriftAll128 forms),
     qwen_t6_continuation, qwen_t5_phase, qwen_t5_first_fork (#653),
     gemma_t5_ordinary, gemma_t5_ngram, gemma_t5_localizer (#665).
  8. the resource policy may select the ADR-063 Q5 runtime pressure rule
     (`runtime_pressure_rule: "q5-critical-stop"`: warning is recorded, only
     critical or unknown pressure stops; baseline samples stay normal-only), and
     a footprint calibration policy (`footprint_calibration: true`, no process
     cap: Q5 "a calibration run without a prior receipt uses the effective floor
     as its cap"); every run records its measured memory footprint
     (`receipt.json` `memory`, per-sample `memory-series.json`).
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
# Qwen T1 records the source and restored active64 caches (qwen_checkpoint::captures64, both families).
RESTORE64_FIELD = {'gemma': 'active64_exact_copy', 'qwen38': 'qwen38_restore64_exact',
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
# Runtime pressure rules (kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical). Absent = the
# reviewed Q1 rule. `q5-critical-stop` is ADR-063 Q5: warning pressure is recorded but is not a stop.
RUNTIME_PRESSURE = {'normal-every-sample': ('1',), 'q5-critical-stop': ('1', '2')}
# gemma_actual `localizer_mode` -> recipe route label; only DriftAll128 runs with the routing witness off.
LOCALIZER_ROUTES = {'Cohort': 'gemma-four-lane-stage-localizer-native-routing1511-v1', 'DriftAll128': 'gemma-all128-stage-drift-witness-off-v1'}
LOGIT_LAYER = 2**32 - 1  # u32::MAX: the Gemma logits/verifier layout (gemma_endpoint.rs); rows are not a layer
# Qwen writes Logits at layer = cfg.num_hidden_layers (qwen_continuation.rs `at`, q1_speculation `plan::at`): 40 on Qwen3.6.
QWEN_LOGIT_LAYER = {'qwen36': 40}
# #653 source-manifest counts (design §1.4/§1.5; confirmed pre-launch against the exported manifest).
QWEN_T6_RECORDS = {'qwen36': dict(continuation=1440, snapshot=560)}
QWEN_T6_SNAPSHOTS = {'ColdChunkMatched64': ('prefix64',), 'CancelledThenRecovered': ('before', 'recovered')}
QWEN_T5_RECORDS = {'qwen36': {'committed': 1840, 'raw-verifier': 20}}
QWEN_T5_SCHEMA = 'hf2q.q1.qwen-t5-phase.v1'
QWEN_FROZEN_REVIEW_SCHEMA = 'hf2q.q1.qwen-t5-frozen-review.v1'
# #665 Gemma T5 adapter (design §2.3 table; contracts C2/C3).
GEMMA_T5_PHASE_SCHEMA = 'hf2q.q1.gemma-t5-phase.v1'
GEMMA_T5_LOCALIZER_SCHEMA = 'hf2q.q1.gemma-t5-localizer.v1'
GEMMA_T5_STORES = {'OrdinaryGlobal': ('prefork', 'committed'), 'NgramTransaction': ('prefork', 'committed', 'raw-verifier')}
GEMMA_T5_RECORDS = {'prefork': 4204, 'committed': 4264, 'raw-verifier': 4}
GEMMA_INJECTED_ORIGIN = 'InjectedDiagnosticFixtureNotProductionProposalQuality'
GEMMA_ROW127_REFUSAL = 'RouteSensitive: global-route initial decision differs from frozen at row 127'
ALL128_PRECONDITION_SCHEMA = 'hf2q.q1.gemma-all128-precondition.v1'
GEMMA_EOS = (1, 106)
GEMMA_VOCAB = 262144
GEMMA_ROUTING_LAYERS = 30
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
def runtime_pressure(policy):
    """Admitted runtime pressure_raw values; the baseline before launch is always normal-only."""
    rule = policy.get('runtime_pressure_rule', 'normal-every-sample'); require(rule in RUNTIME_PRESSURE, 'closed runtime pressure rule')
    return RUNTIME_PRESSURE[rule]
def process_limits(policy):
    """(footprint_calibration, rss cap, footprint cap). A calibration has no process cap: its cap is the runtime floor."""
    calibration = policy.get('footprint_calibration', False); rss, footprint = policy['process_rss_limit_bytes'], policy['process_footprint_limit_bytes']
    require(type(calibration) is bool, 'footprint_calibration is a boolean')
    if calibration:
        require(rss is None and footprint is None and policy.get('runtime_pressure_rule') == 'q5-critical-stop', 'footprint calibration: no process cap, Q5 pressure rule')
    else:
        require(type(rss) is int and type(footprint) is int and rss > 0 and footprint > 0, 'process RSS/footprint limits')
    return calibration, rss, footprint


class MemoryRecord:
    """Measured case footprint. Host metric = Q5 Free+cached (meminfo free + cached_files = vm_stat free +
    speculative + file-backed); `available` also deducts the owned outputs, as the floors do."""
    def __init__(self, baseline):
        free = [x['physical']['free'] + x['physical']['cached_files'] for x in baseline]
        self.base = dict(min_free_plus_cached_bytes=min(free), max_free_plus_cached_bytes=max(free), samples=len(baseline),
                         max_wired_bytes=max(x['physical']['wired'] for x in baseline), swap_bytes=baseline[0]['swap_bytes'],
                         compressor_bytes=baseline[0]['physical']['compressor'], pressure_raw=sorted({x['pressure_raw'] for x in baseline}))
        self.series = []; self.pressure = {}; self.process = dict(peak_phys_footprint_bytes=None, peak_resident_bytes=None, peak_wired_bytes=None, lifetime_max_phys_footprint_bytes=None)
    def sample(self, x, available, spawn_ns):
        p = x['physical']; self.pressure[x['pressure_raw']] = self.pressure.get(x['pressure_raw'], 0) + 1
        self.series.append(dict(t_ns=x['start_ns'] - spawn_ns, free_plus_cached=p['free'] + p['cached_files'], available=available, wired=p['wired'],
                                compressor=p['compressor'], swap=x['swap_bytes'], pressure_raw=x['pressure_raw'], footprint=None, resident=None))
    def ledger(self, value):
        for key, field in (('peak_phys_footprint_bytes', 'phys_footprint'), ('peak_resident_bytes', 'resident_size'), ('peak_wired_bytes', 'wired_size'), ('lifetime_max_phys_footprint_bytes', 'lifetime_max_phys_footprint')):
            self.process[key] = max(self.process[key] or 0, value[field])
        if self.series: self.series[-1].update(footprint=value['phys_footprint'], resident=value['resident_size'])
    def summary(self):
        s = self.series; low = min((r['free_plus_cached'] for r in s), default=None)
        drops = [a['free_plus_cached'] - b['free_plus_cached'] for a, b in zip(s, s[1:])]
        return dict(metric='free + cached_files (meminfo) = Q5 Free+cached', baseline=self.base, runtime_samples=len(s), pressure_counts=dict(self.pressure),
                    min_free_plus_cached_bytes=low, min_available_after_owned_outputs_bytes=min((r['available'] for r in s), default=None),
                    charged_bytes=None if low is None else max(0, self.base['min_free_plus_cached_bytes'] - low),
                    max_wired_growth_bytes=None if not s else max(r['wired'] for r in s) - self.base['max_wired_bytes'],
                    max_swap_growth_bytes=None if not s else max(r['swap'] for r in s) - self.base['swap_bytes'],
                    max_compressor_growth_bytes=None if not s else max(r['compressor'] for r in s) - self.base['compressor_bytes'],
                    max_sample_drop_bytes=max(drops, default=None), last_sample_t_ns=s[-1]['t_ns'] if s else None, **self.process)
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
    # A storeless extent (Qwen T5 Preparation) plans no tensor payload: its limit is exactly 0.
    require(type(extent['expected_stores']) is dict and type(extent['host_payload_limit']) is int and (extent['host_payload_limit'] > 0 or (extent['host_payload_limit'] == 0 and extent['expected_stores'] == {})), 'bounded extent')
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
    return dict(reaped=p.poll() is not None, group_empty=empty, errors=errors, exit=p.returncode, ended_ns=time.monotonic_ns())


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


def routing_bank(out, identity, witness, namespaces, mode=None):
    """output/routing-witness.jsonl as written by gemma_routing_witness: header for this composed identity (and witness
    mode, when given), unique native-valid Scored rows with finite raw values in `namespaces` only, end.records = rows,
    all scored rows captured, metadata <= the bindings quota. Returns the (namespace, lane, position, layer) keys."""
    seen = set(); end = None; path = out / 'output/routing-witness.jsonl'
    with path.open() as file:
        header = json.loads(next(file)); require(header['kind'] == 'header' and header['identity'] == identity and header['qualified'] is False and (mode is None or header['mode'] == mode), 'same actual identity routing bank')
        for line in file:
            require(end is None, 'no data after routing end'); r = json.loads(line)
            if r['kind'] == 'end': end = r; continue
            require(r['kind'] == 'routing_row' and r['native_boundary_valid'] is True and r['selected_copy_valid'] is True and r['native_decode_error'] is None and r['qualified'] is False, 'actual native Scored row')
            at = r['at']; key = (at['namespace'], at['row']['lane'], at['row']['position'], at['layer'])
            require(key not in seen and at['namespace'] in namespaces, 'unique row in a declared namespace'); seen.add(key)
            require(all(type(x) is int and 0 <= x < 2**32 and (x & 0x7f800000) != 0x7f800000 for x in r['scores_f32_bits'] + r['selected_weights_f32_bits']), 'finite raw native values')
    require(end is not None and end['records'] == len(seen) and end['all_scored_rows_captured'] is True and end['full_Q1_qualified'] is False, 'complete routing bank end')
    require(path.stat().st_size <= witness['metadata_limit_bytes'], 'routing metadata quota')
    return seen


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
        seen = routing_bank(out, identity, witness, {'Canonical', *witness.get('extra_namespaces', ())})
        canonical = {k[1:] for k in seen if k[0] == 'Canonical'}
        require(canonical == {(l, p, k) for l in range(4) for p in range(rows) for k in range(witness['layers'])}, 'complete canonical routing bank')
    return dict(execution=execution, captured_records=len(captured), logit_records=len(logits), captured_bytes=total, whole_q1_qualified=False)


def libtest_prefix(plan):
    """libtest prints `test <filter> ... ` and the test's first stdout line on ONE line, so a runner's first
    println arrives prefixed; a later one arrives bare. Both are the same completion line."""
    return 'test ' + plan['filter'] + ' ... '
def bare_line(plan):
    """The bound completion line without the libtest prefix (a binding may carry either form)."""
    line = plan['completion_line']; prefix = libtest_prefix(plan)
    return line[len(prefix):] if line.startswith(prefix) else line
def completion_count(lines, plan, bare):
    """Occurrences of the completion line in either form (bare, or libtest-prefixed with this case's filter)."""
    return lines.count(bare) + lines.count(libtest_prefix(plan) + bare)
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
    bare = bare_line(plan) if plan['completion_line'] else plan['completion_marker'] + ' records=' + str(records) + ' whole_q1_qualified=false'
    require(completion_count(lines, plan, bare) == 1, 'fixed completion line exactly once' if plan['completion_line'] else 'exact multistore completion line')
    evidence = dict(execution=execution, stores={n: dict(records=len(s['captured']), bytes=s['bytes']) for n, s in stores.items()}, captured_records=records, captured_bytes=total, whole_q1_qualified=False)
    for check in plan['checks']:
        evidence[check] = MULTISTORE_CHECKS[check](out, b, case, plan, packet, extent, execution, stores, identity, policy)
    return evidence


def check_gemma_loc4(out, b, case, plan, packet, extent, execution, stores, identity, policy):
    """Gemma four-lane stage localizer (#652): the unchanged T0/T2 endpoint store plus a four-lane stage store.

    Stage extent: the legacy packet `expected_stores` form, or the plan-only form (packet `stage_manifest` = the case's
    bound `manifest`, no `expected_stores`). `localizer_mode` (absent = Cohort): Cohort keeps the routing bank in bindings,
    packet and receipt; DriftAll128 (#665 §2.6, route DRIFT_ROUTE only) is All128 prefill-only with the witness OFF
    everywhere (packet `routing_witness` null, no bindings witness, no bank file) and `observer_effect_reference` null."""
    trajectory = packet.get('trajectory'); stage = packet.get('stage_localizer'); mode = packet.get('localizer_mode', 'Cohort')
    require(b['family'] == 'gemma' and plan['stores'] == {'endpoint': '.', 'stages': 'stages'} and trajectory in COHORT_ROUTES and type(stage) is dict, 'Gemma All128/packed localizer stores')
    drift = mode == 'DriftAll128'; reference = packet.get('observer_effect_reference')
    require(mode in LOCALIZER_ROUTES and execution['localizer_mode'] == mode and execution['route'] == LOCALIZER_ROUTES[mode] and execution['observer_effect_reference'] == reference and (reference is None or (type(reference) is str and reference)), 'localizer mode, recipe route and observer-effect reference echoed')
    if drift:
        # The only witness-OFF route: no routing bank is asserted, and none may exist.
        require(trajectory == 'T0ColdSerial' and 'routing_witness' in packet and packet['routing_witness'] is None and b.get('routing_witness') is None and reference is None and stage.get('decode_stages') == [] and execution['routing_witness_installed'] is False and not (out / 'output/routing-witness.jsonl').exists(), 'DriftAll128: All128 prefill-only, witness off everywhere, no observer reference')
        # Both drift sides must run the same reviewed plan: the stage extent is the bound manifest, never an inline list.
        require('stage_manifest' in packet, 'DriftAll128: plan-only form with the bound stage manifest')
    else:
        # The canonical routing bank is part of this route's evidence: never skipped (spec Part 3; only DriftAll128 may omit it).
        require(type(b.get('routing_witness')) is dict and type(packet.get('routing_witness')) is dict and execution['routing_witness_installed'] is True, 'cohort localizer requires the routing witness in bindings and packet')
    if 'stage_manifest' in packet:
        require('expected_stores' not in packet and 'manifest' in b['cases'][case] and packet['stage_manifest'] == b['cases'][case]['manifest'] and execution['stage_extent_source'] == 'BoundStageManifest' and execution['stage_manifest'] == packet['stage_manifest'], 'plan-only form: stage extent from the bound case manifest')
    else:
        require('expected_stores' in packet and execution['stage_extent_source'] == 'PacketExpectedStores' and execution['stage_manifest'] is None, 'legacy form: stage extent from the packet')
    require(extent['lanes'] == [0, 1, 2, 3] and policy.get('lanes') == 4 and type(policy.get('stage_payload_upper_bytes')) is int, 'four-lane extent and resource policy')
    staged = stores['stages']
    require(execution['schema'] == COHORT_SCHEMA and execution['trajectory'] == trajectory and execution['stage_route'] == COHORT_ROUTES[trajectory] and execution['rerun_id'] == stage['rerun_id'] and execution['stage_store'] == 'stages' and execution['lanes'] == [0, 1, 2, 3] and execution['stage_records'] == len(staged['captured']), 'cohort localizer receipt for this route')
    require({x['at']['agent'] for x in staged['captured']} == {'agent-' + str(l) for l in range(4)}, 'stage records cover all four lanes')
    require(staged['bytes'] == execution['stage_planned_payload'] <= stage['host_payload_limit'] <= policy['stage_payload_upper_bytes'], 'stage bytes = planned <= packet <= four-lane policy')
    endpoint = stores['endpoint']
    evidence = endpoint_evidence(out, b, trajectory, packet, execution, endpoint['captured'], endpoint['expected'], identity, policy)
    return dict(trajectory=trajectory, stage_route=COHORT_ROUTES[trajectory], localizer_mode=mode, route=LOCALIZER_ROUTES[mode], observer_effect_reference=reference, stage_extent_source=execution['stage_extent_source'], routing_bank_asserted=not drift, stage_records=len(staged['captured']), stage_bytes=staged['bytes'], endpoint_records=evidence['captured_records'], logit_records=evidence['logit_records'], endpoint_bytes=evidence['captured_bytes'])


def u32(x): return type(x) is int and 0 <= x < 2**32
def same_named(plan, names, why): require(plan['stores'] == {n: n for n in names}, why)
def logit_rows(store, layer):
    """(agent, row) of every Logits record at the family's logits layer (u32::MAX for Gemma, num_hidden_layers for Qwen), segment 0."""
    rows = [x['at'] for x in store['captured'] if x['at']['stage'] == 'Logits']
    require(all(at['layer'] == layer and at['segment'] == 0 for at in rows), 'Logits at the family logits layer, segment 0')
    keys = {(at['agent'], at['logical_row_start']) for at in rows}; require(len(keys) == len(rows), 'unique logit rows'); return keys


def qwen_calls(mode):
    """continuation.rs `require_calls`: per lane one prefix call to cursor 128, then single-row decode 128..143."""
    before, rows = (0, 128) if mode == 'ColdBaseline' else (64, 64)
    return [c for l in range(4) for c in [dict(lane=l, before=before, input_rows=rows, after=128, output_rows=rows)] + [dict(lane=l, before=p, input_rows=1, after=p + 1, output_rows=1) for p in range(128, 144)]]


def check_qwen_t6_continuation(out, b, case, plan, packet, extent, execution, stores, identity, policy):
    """Qwen T6 block (#653 §1.4): t3cm (`ColdChunkMatched64`: continuation + prefix64) or t6 (`CancelledThenRecovered`:
    continuation + before + recovered). The fixed source line carries no count, so every store is bound to the receipt
    (`trace_records` = continuation) and to the family's source-manifest count; `planned_tensor_payload` = Σ store bytes."""
    mode = packet.get('mode'); counts = QWEN_T6_RECORDS.get(b['family']); snapshots = QWEN_T6_SNAPSHOTS.get(mode)
    require(counts is not None and snapshots is not None, 'frozen Qwen continuation mode and family counts')
    same_named(plan, ('continuation',) + snapshots, 'continuation stores by mode')
    require(plan['completion_line'] is not None and bare_line(plan) == 'Q1_QWEN_RECOVERY_CONTINUATION_COMPLETE mode=%s lanes=4 rows_per_lane=80 first64_logits=false whole_q1=false' % mode and extent['lanes'] == [0, 1, 2, 3], 'verbatim source completion line; four lanes')
    t6 = mode == 'CancelledThenRecovered'
    require(execution['schema'] == 'hf2q.q1.qwen-recovery-continuation.v1' and execution['mode'] == mode and execution['calls'] == qwen_calls(mode), 'receipt calls = require_calls order')
    require(execution['captured_rows_per_lane'] == 80 and execution['cancelled_unpublished_rows_per_lane'] == (64 if t6 else 0) and execution['first64_logits_captured'] is False and execution['endpoint_scope'] == [64, 144] and execution['active_cache_boundaries'] == [128, 144], 'rows 64..143 per lane, caches at 128 and 144')
    tail = stores['continuation']
    require(execution['trace_records'] == len(tail['captured']) == counts['continuation'], 'continuation = trace_records = source count')
    require(logit_rows(tail, QWEN_LOGIT_LAYER[b['family']]) == {('agent-' + str(l), p) for l in range(4) for p in range(64, 144)}, 'continuation logits rows 64..143 x 4 lanes')
    require(all(len(stores[n]['captured']) == counts['snapshot'] for n in snapshots), 'every active64 snapshot store = source count')
    require(execution['planned_tensor_payload'] == sum(s['bytes'] for s in stores.values()), 'planned payload = captured bytes of every store')
    if t6:
        recovery = execution['recovery']; proof = execution['recovery_proof']
        require(type(recovery) is list and len(recovery) == 4 and all(x['recovered_cursor'] == 64 and x['scheduler_in_flight_after'] == 0 and x['client_closed_after_actual_advance'] is True for x in recovery), 'four actual cancel+recover owners')
        # Comparison has no record count: its compared bytes bind it to the complete 64-snapshot stores.
        require(type(proof) is dict and proof['disposition'] == 'ExactEqual' and proof['compared_bytes'] == stores['before']['bytes'] == stores['recovered']['bytes'] and 'prefix_preparation' not in execution, 'in-process recovery proof ExactEqual over the whole snapshot')
    else:
        prep = execution['prefix_preparation']
        require(type(prep) is list and len(prep) == 4 and all(x['cancellation_performed'] is False and x['retained_cursor'] == 64 and x['scheduler_in_flight_after'] == 0 for x in prep) and execution['recovery'] == [] and execution['recovery_proof'] is None, 'four never-cancelled prefix64 preparations')
    return dict(mode=mode, stores={n: len(s['captured']) for n, s in stores.items()}, trace_records=execution['trace_records'])


def qwen_t5_receipt(plan, packet, execution, first_fork):
    mode = packet.get('mode')
    require(packet.get('target_layers') is None and 'target_layer_localizer' not in execution and packet.get('first_fork') == (True if first_fork else None) and ('first_fork' in execution) == first_fork, 'explicit phase / first-fork packet pairing')
    require(execution['schema'] == QWEN_T5_SCHEMA and execution['mode'] == mode and execution['error'] is None and execution['forced_proposals'] is (mode == 'ForcedProposal'), 'Qwen T5 receipt for the packet mode')
    return mode


def check_qwen_t5_phase(out, b, case, plan, packet, extent, execution, stores, identity, policy):
    """Qwen T5 phases (#653 §1.5). Preparation: no store, the frozen file + preparation-only receipt. OrdinaryBaseline /
    ForcedProposal: `committed` (+ `raw-verifier` forced) bound to `committed_records` / `raw_records` and the source counts;
    0 / 4 K3 transitions; the runner's validate_complete input rows and 17 decisions per lane."""
    mode = qwen_t5_receipt(plan, packet, execution, False); counts = QWEN_T5_RECORDS.get(b['family']); forced = mode == 'ForcedProposal'
    require(counts is not None and mode in ('Preparation', 'OrdinaryBaseline', 'ForcedProposal') and extent['lanes'] == [0, 1, 2, 3], 'Qwen T5 family counts and phase')
    require(plan['completion_line'] is not None and bare_line(plan) == 'Q1_QWEN_T5_PHASE_COMPLETE mode=%s logical_lanes=4 explicit_MTP_selector=true forced_proposal=%s whole_q1=false' % (mode, 'true' if forced else 'false'), 'verbatim source completion line')
    rows = [128] + list(range(131, 144)) if forced else list(range(128, 144))
    require(execution['ordinary_input_positions'] == [rows] * 4 and type(execution['canonical_decisions']) is list and len(execution['canonical_decisions']) == 4 and all(len(d) == 17 for d in execution['canonical_decisions']), 'ordinary input rows and seed+16 decisions per lane')
    if mode == 'Preparation':
        frozen = out / 'output/frozen-workload.json'; prep_path = out / 'output/preparation-only.json'
        require(frozen.is_file() and prep_path.is_file(), 'preparation wrote frozen-workload.json and preparation-only.json')
        prep = json.loads(prep_path.read_text())
        require(execution['status'] == 'CompleteWorkloadPreparationNotComparable' and plan['stores'] == {} and extent['expected_stores'] == {} and execution['committed_records'] is None and execution['raw_records'] is None and execution['actual_transitions'] == [], 'storeless preparation receipt')
        require(prep['frozen_file_sha256'] == sha(frozen) and prep['comparable_acquisition'] is False and prep['requires_independent_review'] is True, 'frozen workload + preparation-only receipt, pending independent review')
        return dict(mode=mode, frozen_file_sha256=prep['frozen_file_sha256'], preparation_sha256=sha(out / 'output/preparation-only.json'))
    names = ('committed', 'raw-verifier') if forced else ('committed',); same_named(plan, names, 'Qwen T5 phase stores by mode')
    require(execution['status'] == 'CompleteScopedAcquisition', 'complete scoped acquisition')
    require(execution['committed_records'] == len(stores['committed']['captured']) == counts['committed'] and execution['committed_payload_plan'] == stores['committed']['bytes'], 'committed = committed_records = source count, bytes = plan')
    if forced:
        require(execution['raw_records'] == len(stores['raw-verifier']['captured']) == counts['raw-verifier'] and execution['raw_payload_plan'] == stores['raw-verifier']['bytes'], 'raw-verifier = raw_records = source count, bytes = plan')
    else:
        require(execution['raw_records'] is None, 'baseline has no raw-verifier store')
    transitions = execution['actual_transitions']
    require(type(transitions) is list and [t['lane'] for t in transitions] == ([0, 1, 2, 3] if forced else []), 'one actual K3 transition per lane (forced) or none')
    return dict(mode=mode, stores={n: len(stores[n]['captured']) for n in names}, transitions=len(transitions))


def check_qwen_t5_first_fork(out, b, case, plan, packet, extent, execution, stores, identity, policy):
    """Qwen T5 first fork (#653 §1.5 T5-A3): agent 0 only, one logical lane; `committed` (+ `raw-verifier`) bound to the
    receipt counts and every side-effect capture store bound to its `first_fork.capture_stores[name].records`;
    `case_payload_plan` = Σ store bytes."""
    mode = qwen_t5_receipt(plan, packet, execution, True); forced = mode == 'ForcedProposal'
    require(b['family'] in ('qwen36', 'qwen38') and mode in ('OrdinaryBaseline', 'ForcedProposal'), 'Qwen first-fork arm')
    captures = ('target-before-mtp', 'target-after-mtp') + (('post-draft-discard129',) if forced else ())
    same_named(plan, ('committed',) + (('raw-verifier',) if forced else ()) + captures, 'first-fork stores by arm')
    require(plan['completion_line'] is not None and bare_line(plan) == 'Q1_QWEN_T5_FIRST_FORK_COMPLETE mode=%s logical_lanes=1 agent=0 cursor=131 float_policy=Unspecified whole_q1=false' % mode, 'verbatim source completion line')
    require(execution['status'] == 'CompleteFirstForkDiagnosticNotQ1Qualified', 'complete first-fork diagnostic')
    ff = execution['first_fork']
    require(ff['logical_lanes_executed'] == 1 and ff['logical_lane'] == 0 and ff['physical_slots'] == 1 and ff['all4lane_execution'] is False and extent['lanes'] == [0] and all(x['at']['agent'] == 'agent-0' for s in stores.values() for x in s['captured']), 'agent 0 only')
    cs = ff['capture_stores']
    require(type(cs) is dict and set(cs) == set(captures) and all(cs[n]['records'] == len(stores[n]['captured']) and cs[n]['errors'] == [] for n in captures), 'capture_stores counts = their stores')
    require(execution['committed_records'] == len(stores['committed']['captured']) and execution['raw_records'] == (len(stores['raw-verifier']['captured']) if forced else None), 'committed/raw-verifier = receipt counts')
    require(ff['case_payload_plan'] == sum(s['bytes'] for s in stores.values()), 'case payload plan = captured bytes of every store')
    inputs = packet.get('first_fork_inputs')
    require(type(inputs) is dict and ff['reviewed_inputs'] == dict(preparation_sha256=inputs['preparation']['sha256'], review_sha256=inputs['review']['sha256'], review_schema=QWEN_FROZEN_REVIEW_SCHEMA), 'first fork ran on the packet-bound reviewed preparation')
    return dict(mode=mode, stores={n: len(s['captured']) for n, s in stores.items()})


def gemma_frozen(packet):
    work = load(packet['greedy_workload'], 'bound reviewed greedy144 workload')
    require(work['family'] == 'Gemma4' and len(work['lanes']) == 4 and all(len(l) == 144 for l in work['lanes']), 'four144 Gemma workload')
    return work['lanes']


def gemma_t5_decisions(arm, decisions, frozen):
    """Per-lane decisions of one complete arm against the frozen lineage; returns the accepted count a per lane
    (None for the ordinary arm). N-gram: accepted/fallback follow accept_prefix_argmax over the injected drafts."""
    require(type(decisions) is list and [d.get('lane') for d in decisions] == [0, 1, 2, 3], 'one decision record per lane, in order')
    accepted = []
    for d in decisions:
        w = frozen[d['lane']]; t = d['transaction']
        require(set(d) == {'lane', 'initial_argmax', 'frozen_128', 'initial_equals_frozen', 'decode_inputs', 'decode_argmax', 'transaction'} and u32(d['initial_argmax']) and d['frozen_128'] == w[128] and d['initial_equals_frozen'] is (d['initial_argmax'] == w[128]), 'initial decision recorded against frozen[128]')
        if arm == 'OrdinaryGlobal':
            require(t is None, 'ordinary arm has no transaction'); a = None; start = 128
        else:
            require(type(t) is dict and set(t) == {'proposer_ran', 'proposed', 'injected', 'injected_origin', 'verifier_argmax', 'accepted', 'rejected', 'fallback', 'rollback_cursor'} and d['initial_equals_frozen'] is True, 'closed transaction from the frozen initial decision')
            inj = t['injected']; v = t['verifier_argmax']; a = t['accepted']
            require(t['proposer_ran'] is True and type(t['proposed']) is list and all(u32(x) for x in t['proposed']) and t['injected_origin'] == GEMMA_INJECTED_ORIGIN, 'production proposer ran; injected diagnostic origin')
            require(type(inj) is list and len(inj) == 3 and inj[:2] == w[129:131] and u32(inj[2]) and inj[2] != w[131] and inj[2] not in GEMMA_EOS, 'injected [f129, f130, non-EOS mut131]')
            require(type(v) is list and len(v) == 4 and all(u32(x) for x in v), 'four verifier argmax rows')
            lead = 0
            while lead < 3 and v[lead] == inj[lead]: lead += 1
            require(type(a) is int and a in (0, 1, 2) and a == lead and t['rejected'] == 3 - a and t['fallback'] == v[a] and t['rollback_cursor'] == 129 + a, 'a in {0,1,2} = leading matches, rejected 3-a, fallback, cursor 129+a')
            start = 129 + a
        require(d['decode_inputs'] == [dict(position=p, token=w[p]) for p in range(start, 144)], 'teacher-forced decode inputs = frozen')
        require(type(d['decode_argmax']) is list and [x.get('position') for x in d['decode_argmax']] == list(range(start, 144)) and all(set(x) == {'position', 'token'} and u32(x['token']) for x in d['decode_argmax']), 'one decision per forward_decode row')
        accepted.append(a)
    return accepted


def gemma_t5_arm(out, b, plan, packet, extent, execution, stores, identity, schema, extra_stores):
    """Shared Gemma T5 arm rules (C2): receipt, per-store counts, logit layouts, decisions and the routing bank."""
    arm = packet.get('arm'); names = GEMMA_T5_STORES.get(arm)
    require(b['family'] == 'gemma' and names is not None and plan['completion_line'] is None, 'Gemma T5 arm with the Σ completion line')
    require(plan['stores'] == {**{n: n for n in names}, **extra_stores}, 'Gemma T5 stores for this arm')
    require(execution['schema'] == schema and execution['arm'] == arm and execution['status'] == 'CompleteScopedAcquisition' and execution['error'] is None and execution['lanes'] == extent['lanes'] == packet['lanes'] == [0, 1, 2, 3] and execution['class_b_refusal'] is None and execution['routing_witness'] == 'routing-witness.jsonl', 'complete Gemma T5 receipt for this arm')
    counts = {n: len(stores[n]['captured']) for n in names}
    require(execution['stores'] == counts == {n: GEMMA_T5_RECORDS[n] for n in names}, 'per-store counts = receipt = source manifest')
    lanes = ['agent-' + str(l) for l in range(4)]
    require(logit_rows(stores['prefork'], LOGIT_LAYER) == {(l, 127) for l in lanes} and logit_rows(stores['committed'], LOGIT_LAYER) == {(l, p) for l in lanes for p in range(128, 144)}, 'prefork row 127, committed rows 128..143, four lanes')
    if 'raw-verifier' in names:
        raw = stores['raw-verifier']['captured']
        require(sorted(x['at']['agent'] for x in raw) == lanes and all(x['at']['stage'] == 'SpecVerifierLogits' and x['at']['logical_step'] == 128 and x['at']['layer'] == LOGIT_LAYER and x['at']['logical_row_start'] == 0 and x['at']['segment'] == 0 and x['shape'] == [4, GEMMA_VOCAB] for x in raw), 'one [4, vocab] verifier record per lane')
    frozen = gemma_frozen(packet); accepted = gemma_t5_decisions(arm, execution['decisions'], frozen)
    witness = b.get('routing_witness')
    require(type(witness) is dict and witness.get('layers') == GEMMA_ROUTING_LAYERS and type(packet.get('routing_witness')) is dict, 'T5 routing witness on in bindings and packet')
    layers = range(GEMMA_ROUTING_LAYERS)
    if arm == 'OrdinaryGlobal':
        expected = {('Canonical', l, p, k) for l in range(4) for p in range(144) for k in layers}
    else:
        expected = {(ns, l, p, k) for l in range(4) for ns, rows in (('T5Initial', range(128)), ('T5Verify', range(132)), ('T5Continuation', range(129 + accepted[l], 144))) for p in rows for k in layers}
    seen = routing_bank(out, identity, witness, {k[0] for k in expected}, 'Ordinary' if arm == 'OrdinaryGlobal' else 'Transaction')
    require(seen == expected, 'exact per-lane T5 routing rows x layers')
    return arm, accepted, dict(arm=arm, stores=counts, accepted=accepted, initial_argmax=[d['initial_argmax'] for d in execution['decisions']], initial_equals_frozen=[d['initial_equals_frozen'] for d in execution['decisions']], routing_rows=len(seen))


def check_gemma_t5_ordinary(out, b, case, plan, packet, extent, execution, stores, identity, policy):
    """Gemma T5 OrdinaryGlobal arm (#665 §2.3): prefork 4204 + committed 4264, teacher-forced inputs 128..143, the initial
    argmax recorded against frozen[128] (a cross-route Class B observation, never a refusal), Canonical routing bank."""
    arm, _, evidence = gemma_t5_arm(out, b, plan, packet, extent, execution, stores, identity, GEMMA_T5_PHASE_SCHEMA, {})
    require(arm == 'OrdinaryGlobal', 'ordinary arm'); return evidence


def sibling_execution(b, case, check):
    """The unique other multistore case of these bindings carrying `check`; it must be complete and its receipt must
    embed exactly its output/execution.json."""
    names = [n for n, c in b['cases'].items() if n != case and c.get('kind') == 'multistore' and check in c.get('checks', ())]
    require(len(names) == 1, 'exactly one sibling ' + check + ' case in these bindings')
    root = Path(b['cases'][names[0]]['out']); receipt = json.loads((root / 'receipt.json').read_text()); path = root / 'output/execution.json'
    execution = json.loads(path.read_text())
    require(receipt['case_complete'] is True and receipt['helper_evidence']['execution'] == execution, names[0] + ' complete; its receipt binds its execution.json')
    return names[0], path, execution


def check_gemma_t5_ngram(out, b, case, plan, packet, extent, execution, stores, identity, policy):
    """Gemma T5 NgramTransaction arm (#665 §2.3): four injected transactions, a/fallback/cursor 129+a, teacher-forced
    inputs 129+a..143, prefork 4204 + committed 4264 + raw-verifier 4, T5Initial/T5Verify/T5Continuation bank.

    Cross-case (G5-A2, initial argmax equal to the ordinary arm): the reference is the same bindings' unique sibling
    `gemma_t5_ordinary` case (same composition), which must be complete; its receipt.json must embed its execution.json
    verbatim, and that receipt's per-lane initial argmax must equal this arm's. No packet field is involved."""
    arm, accepted, evidence = gemma_t5_arm(out, b, plan, packet, extent, execution, stores, identity, GEMMA_T5_PHASE_SCHEMA, {'raw-verifier': 'raw-verifier'})
    name, path, ordinary = sibling_execution(b, case, 'gemma_t5_ordinary')
    require(arm == 'NgramTransaction' and ordinary['schema'] == GEMMA_T5_PHASE_SCHEMA and ordinary['arm'] == 'OrdinaryGlobal' and ordinary['status'] == 'CompleteScopedAcquisition', 'complete ordinary sibling arm')
    require([d['initial_argmax'] for d in ordinary['decisions']] == evidence['initial_argmax'], 'initial argmax equal to the ordinary arm (same call)')
    return dict(evidence, ordinary_case=name, ordinary_execution_sha256=sha(path))


def check_gemma_t5_localizer(out, b, case, plan, packet, extent, execution, stores, identity, policy):
    """Gemma T5 verifier-route localizer arm (#665 §2.6): the matching T5 arm's rules plus a nested `stages` store whose
    extent is the bound case manifest (packet `manifest` = case `manifest`; the Rust admit binds Plan -> manifest), stage
    coordinates per arm (ordinary: prefill rows 0..127 segment 0, decode 128..143; n-gram: verify rows 0..131 segment 1,
    decode 129+a..143 segment 0; step 0), and the packet's `t5_receipts` (both sha-bound T5 execution.json files).
    Branch N: frozen `accepted` = the T5 n-gram a. Branch P: OrdinaryGlobal only, `accepted` null, the bound n-gram
    receipt carries the row-127 Class B refusal."""
    arm, accepted, evidence = gemma_t5_arm(out, b, plan, packet, extent, execution, stores, identity, GEMMA_T5_LOCALIZER_SCHEMA, {'stages': 'stages'})
    stage = packet.get('stage_localizer'); branch = packet.get('branch'); passed = packet.get('pass'); manifest = b['cases'][case].get('manifest')
    require(manifest is not None and packet.get('manifest') == manifest, 'stage extent from the bound case manifest')
    require(type(stage) is dict and set(stage) == {'rerun_id', 'layers', 'prefill_stages', 'decode_stages', 'host_payload_limit', 'accepted'} and branch in ('N', 'P') and passed in (1, 2), 'closed localizer plan, branch and pass')
    staged = stores['stages']
    require(execution['plan'] == stage and execution['branch'] == branch and execution['pass'] == passed and execution['stage_records'] == len(staged['captured']) and staged['bytes'] <= stage['host_payload_limit'], 'receipt echoes plan, branch, pass; stage count and bytes')
    receipts = packet.get('t5_receipts')
    require(type(receipts) is dict and set(receipts) == {'ordinary', 'ngram'}, 'both T5 receipts bound')
    ordinary = load(receipts['ordinary'], 'bound T5 ordinary receipt'); ngram = load(receipts['ngram'], 'bound T5 n-gram receipt')
    require(ordinary['schema'] == ngram['schema'] == GEMMA_T5_PHASE_SCHEMA and ordinary['arm'] == 'OrdinaryGlobal' and ordinary['status'] == 'CompleteScopedAcquisition' and ngram['arm'] == 'NgramTransaction', 'T5 receipts of both arms')
    if branch == 'N':
        a = [d['transaction']['accepted'] for d in ngram['decisions']]
        require(ngram['status'] == 'CompleteScopedAcquisition' and ngram['class_b_refusal'] is None and stage['accepted'] == a and (arm == 'OrdinaryGlobal' or accepted == a), 'branch N: frozen accepted = the T5 n-gram a')
    else:
        refusal = ngram['class_b_refusal']
        require(arm == 'OrdinaryGlobal' and stage['accepted'] is None and ngram['status'] == 'FailedOrIncomplete' and type(refusal) is dict and refusal['row'] == 127 and refusal['reason'] == GEMMA_ROW127_REFUSAL, 'branch P: ordinary arm only, no accepted, bound row-127 refusal')
    precondition = all128_precondition(packet, execution, branch)
    if arm == 'OrdinaryGlobal':
        prefill = {(0, r) for r in range(128)}; decode = {(0, p) for p in range(128, 144)}; per_lane = [(prefill, decode)] * 4
    else:
        per_lane = [({(1, r) for r in range(132)}, {(0, p) for p in range(129 + a, 144)}) for a in accepted]
    layers = set(stage['layers']); roles = (set(stage['prefill_stages']), set(stage['decode_stages'])); seen = set()
    for x in staged['captured']:
        at = x['at']; lane = int(at['agent'][len('agent-'):]); key = (at['segment'], at['logical_row_start'])
        phase = [i for i in (0, 1) if key in per_lane[lane][i]]
        require(at['logical_step'] == 0 and at['layer'] in layers and len(phase) == 1 and at['stage'] in roles[phase[0]], 'stage record in the plan layout for this arm')
        seen.add((lane,) + key)
    want = {(l,) + k for l in range(4) for i in (0, 1) if roles[i] for k in per_lane[l][i]}
    require(seen == want, 'every planned (lane, segment, row) captured')
    refusal = ngram['class_b_refusal']
    return dict(evidence, branch=branch, plan_pass=passed, stage_records=len(staged['captured']), stage_bytes=staged['bytes'], t5_receipts={k: v['sha256'] for k, v in receipts.items()}, ngram_class_b_refusal=refusal, all128_precondition=precondition)


def all128_precondition(packet, execution, branch):
    """Branch P only (§2.6, condition 7): the packet binds the PyTools `all128-precondition` record. The b5g All128
    reference's own row-127 argmax must equal frozen[128] (`holds` true) on this case's greedy workload; the receipt
    echoes {path, sha256, holds}. Branch N carries no precondition (absent or null)."""
    bound_record = packet.get('all128_precondition')
    if branch == 'N':
        require(bound_record is None and execution.get('all128_precondition') is None, 'branch N: no All128 precondition')
        return None
    record = load(bound_record, 'bound All128 precondition record')
    lanes = record.get('lanes') if type(record) is dict else None
    require(type(record) is dict and record.get('schema') == ALL128_PRECONDITION_SCHEMA and record.get('holds') is True and type(record.get('workload')) is dict and record['workload'].get('sha256') == packet['greedy_workload']['sha256'], 'All128 precondition holds on this greedy workload')
    require(type(lanes) is list and [x.get('lane') for x in lanes] == [0, 1, 2, 3] and all(x.get('equal') is True and x.get('argmax') == x.get('frozen_128') for x in lanes), 'All128 row-127 argmax = frozen[128] on every lane')
    require(execution.get('all128_precondition') == dict(bound_record, holds=True), 'receipt echoes the bound All128 precondition')
    return dict(bound_record, holds=True)


# Closed registry of multistore check functions: (out, b, case, plan, packet, extent, execution, stores, identity, policy) -> evidence dict.
MULTISTORE_CHECKS = {'gemma_loc4': check_gemma_loc4, 'qwen_t6_continuation': check_qwen_t6_continuation, 'qwen_t5_phase': check_qwen_t5_phase,
                     'qwen_t5_first_fork': check_qwen_t5_first_fork, 'gemma_t5_ordinary': check_gemma_t5_ordinary,
                     'gemma_t5_ngram': check_gemma_t5_ngram, 'gemma_t5_localizer': check_gemma_t5_localizer}


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
    admitted_pressure = runtime_pressure(policy); calibration, rss_limit, footprint_limit = process_limits(policy)
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
    result = dict(family=b['family'], case=args.case, kind=plan['kind'], operating_review_sha256=operating_sha, case_complete=False, mode=mode, model_launch_attempted=False, physical_admission=False, performance_qualified=False, whole_q1_qualified=False,
                  footprint_calibration=calibration, endpoint_evidence=not calibration, runtime_pressure_admitted=list(admitted_pressure)); child = None; memory = None
    for sig in HANDLED_SIGNALS: signal.signal(sig, interrupt)
    def available(x, owned): return x['physical']['free'] + x['physical']['cached_files'] - sum(disk_bytes(o) for o in owned)
    try:
        quiet(out, 'pre', deny); chip = raw(out, 'host-chip', ['sysctl', '-n', 'machdep.cpu.brand_string']); require(chip['exit'] == 0 and chip['stdout'].strip() == host['chip'] and not chip['stderr'], 'mandatory host chip'); epoch = time.monotonic_ns(); baseline = []
        for i in range(12):
            target = epoch + i * 500000000; time.sleep(max(0, (target - time.monotonic_ns()) / 1e9)); x = sample('baseline', i, target); baseline.append(x)
            require(x['pressure_raw'] == '1' and available(x, [o for o in outputs if o != out]) >= policy['preload_available_floor_bytes'], 'preload operating headroom/pressure (nothing else resident)')
        require(not INTERRUPTED, 'interrupted before load'); stat_model(); require(sha(binary) == b['binary_sha256'], 'prelaunch binary'); memory = MemoryRecord(baseline)
        selected_filter = plan['filter']
        command = [str(out / 'test-binary'), selected_filter, '--exact', '--ignored', '--nocapture', '--test-threads=1']; save(out / 'command.json', dict(argv=command, binary_sha256=b['binary_sha256'], source_commit=b['source_commit'], packet_sha256=sha(out / 'packet.json'), packet_env=plan['packet_env'], deadline_seconds=deadline, mode=mode, kind=plan['kind']))
        env = {k: v for k, v in os.environ.items() if not k.startswith(AMBIENT)}; env[plan['packet_env']] = str(out / 'packet.json')
        with (out / 'stdout').open('xb') as so, (out / 'stderr').open('xb') as se:
            quiet(out, 'immediate-prelaunch', deny); require(not INTERRUPTED, 'interrupted immediately before Popen'); spawn_ns = time.monotonic_ns(); started = time.monotonic(); child = subprocess.Popen(command, cwd=out, env=env, stdout=so, stderr=se, stdin=subprocess.DEVNULL, start_new_session=True); result.update(model_launch_attempted=True, pid=child.pid, spawn_ns=spawn_ns); i = 0; next_sample = time.monotonic_ns(); proc_identity = None; last_live_ledger = None; previous_sample_ns = None
            while child.poll() is None:
                require(not INTERRUPTED and time.monotonic() - started < deadline, 'interruption/case watchdog')
                now = time.monotonic_ns()
                if now >= next_sample:
                    x = sample('runtime', i, now); require(previous_sample_ns is None or x['start_ns'] - previous_sample_ns <= POLICY['max_sample_gap_ns'], 'stale runtime sample'); previous_sample_ns = x['start_ns']
                    avail = available(x, outputs); memory.sample(x, avail, spawn_ns); require(x['pressure_raw'] in admitted_pressure, 'runtime pressure')
                    require(avail >= policy['runtime_available_floor_bytes'], 'runtime operating floor'); require(x['swap_bytes'] - baseline[0]['swap_bytes'] <= policy['max_swap_growth_bytes'] and x['physical']['compressor'] - baseline[0]['physical']['compressor'] <= policy['max_compressor_growth_bytes'], 'swap/compressor growth')
                    ledger = raw(out, 'process-' + str(i), [str(proc), str(child.pid)], timeout=.5); value = json.loads(ledger['stdout'])
                    if ledger['exit'] == 0 and value.get('available') is True:
                        proc_identity = namespace['proc_fields'](value, child.pid, proc_identity)
                        live = dict(available=True, parent_start_ns=ledger['begin_ns'], parent_end_ns=ledger['end_ns'], parent_duration_ns=ledger['end_ns'] - ledger['begin_ns'], identity_validated=True)
                        require(namespace['ledger_ready'](live, POLICY, time.monotonic_ns(), after_ns=spawn_ns), 'fresh actual owned ledger required'); last_live_ledger = live
                        memory.ledger(value)
                        require(calibration or (value['resident_size'] <= rss_limit and value['phys_footprint'] <= footprint_limit), 'RSS/footprint operating limit')
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
                require(post['pressure_raw'] in admitted_pressure and available(post, outputs) >= policy['runtime_available_floor_bytes'], 'terminal-race fresh global headroom/pressure')
                require(post['swap_bytes'] - baseline[0]['swap_bytes'] <= policy['max_swap_growth_bytes'] and post['physical']['compressor'] - baseline[0]['physical']['compressor'] <= policy['max_compressor_growth_bytes'], 'terminal-race fresh global growth guard')
        except BaseException as e: result['post_failure'] = repr(e)
        result['case_complete'] = not result.get('failure') and not result.get('post_failure') and (child is None or (result.get('cleanup', {}).get('reaped') is True and result['cleanup']['group_empty'] and not result['cleanup']['errors'])) and 'helper_evidence' in result
        if memory is not None:
            result['memory'] = memory.summary()
            try: save(out / 'memory-series.json', dict(schema='hf2q.q1.endpoint-memory-series.v1', series=memory.series))
            except BaseException as e: result['memory_series_failure'] = repr(e); result['case_complete'] = False
        save(out / 'receipt.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'helper_evidence'}, indent=2)); raise SystemExit(0 if result['case_complete'] else 1)


if __name__ == '__main__': main()
