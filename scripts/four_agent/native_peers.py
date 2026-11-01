"""Native-engine peer arms (ADR-063 "Native-engine peer arms"): the one registry of engine kinds that
serve their own native artifact, beside 'hf2q' and the same-GGUF 'peer'.

Every consumer (the four-agent client, Q2 certify/run audit, Q3 campaign/identity/report, F2 render,
quality campaign and the process inventories) resolves a native kind here instead of branching on
its name. Kinds are declared in coding_family.NATIVE_KINDS, which every client already freezes; the
arm for kind K is the module native_peer_K.py. The client imports this registry and its own arm only,
so an arm's CLIENT_MODULES (this file, the arm's client modules and the arm's test) are exactly the
extra source-freeze entries of that kind's client runs.
"""
import importlib

from coding_family import NATIVE_KINDS
from qwen38_n1_verdict import require

KINDS = NATIVE_KINDS
# The declared interface of an arm module. Hooks marked "or None" may be None.
INTERFACE = {
    'ENGINE': 'the engine kind (equals the registry key)',
    'FAMILIES': 'harness families (coding_family.FAMILIES) its client runs use',
    'TITLE': 'family title used in selection refusals',
    'Q3_FAMILIES': 'q5/Q3 case families',
    'SLOTS': 'four-slot coding workload only',
    'CLIENT_MODULES': 'client source-freeze additions (registry, arm client modules, arm test)',
    'PROCESSES': 'executable names for the competing-runtime inventories',
    'BASELINES': 'F2 effect baseline id -> label; never a measured column, never a win',
    'AUDIT': 'post-run audit module: audit_run, q3_section, note',
    'SAME_GGUF_UNAVAILABLE': 'verdict reason when the campaign carries --peer-unavailable-evidence',
    'NATIVE_COMPARISON': 'verdict reason when the campaign declares --native-comparison',
    'preflight': '(keys, family) -> approval preflight keys; HF keys replaced, never forged',
    'validate_controls': '(controls, family) -> refuses launch controls other than the declared ones',
    'request_controls': '(canonical request, family) -> exact native request',
    'canonical': '(native request, family) -> canonical request; native fields must be exact',
    'check_unary': '(object) -> closed chat.completion envelope',
    'check_frame': '(object) -> closed chat.completion.chunk envelope',
    'usage': '(usage, turn) -> usage record with cache_observation',
    'readiness': '(base, model, output, approval packet) -> native_readiness record',
    'closing': '(base, coordinator output) -> record once every agent finished (outside the makespan), or None',
    'PEER_CASE_REFUSAL': 'Q3 refusal text when the arm\'s --peer-case lacks its required campaign flag',
    'campaign_identity': '(q5 plan) -> frozen identity additions (peer_build_info, native_runtime)',
    'launch_problems': '(arm name, artifact identity, hf2q artifact sha256) -> refusal strings',
    'label': '(q5 case) -> arm label',
    'q3_role': '(q5 case) -> Q3 plan role',
}


def is_native(kind):
    return kind in KINDS


def adapter(kind):
    """The arm module for a native kind; refuses an unknown kind or an incomplete arm."""
    require(kind in KINDS, f'unknown native engine kind {kind!r}')
    module = importlib.import_module(f'native_peer_{kind}')
    missing = sorted(name for name in INTERFACE if not hasattr(module, name))
    require(not missing and module.ENGINE == kind, f'native arm {kind} lacks ' + ','.join(missing))
    return module


def audit(kind):
    return importlib.import_module(adapter(kind).AUDIT)


def peer_kind(plan):
    """The Q3 plan's native peer kind, or None when its peer arm is the same-GGUF peer (or absent)."""
    kind = plan['arms'].get('peer', {}).get('engine_kind')
    return kind if is_native(kind) else None


def baselines():
    """Every native F2 baseline id -> label."""
    out = {}
    for kind in KINDS:
        out.update(adapter(kind).BASELINES)
    return out


def baseline_of(kind, label):
    """The F2 baseline id an arm label reports under."""
    found = [key for key, value in adapter(kind).BASELINES.items() if value == label]
    require(len(found) == 1, f'no {kind} F2 baseline for label {label!r}')
    return found[0]


def process_names():
    return frozenset(name for kind in KINDS for name in adapter(kind).PROCESSES)
