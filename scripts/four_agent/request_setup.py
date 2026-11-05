"""Opt-in request setups for the four-agent coding client (ADR-063 Q4-QWEN36).

A request setup replaces the canonical greedy omissions with one vendor-documented
sampling configuration. The client sends it explicitly on every request, with the same
values on hf2q and on the peer. Thinking stays on (template default), earlier reasoning is
re-sent in every assistant message (as the canonical client already does), and the
template, tools, task and six-step workflow stay canonical. The server-side thinking budget
is the launch case's concern (q5_cases/qwen36-vendor.json, peer-qwen36-vendor.json).

No process, fixture or host access on import. Canonical clients do not import this module
unless the explicit CLI and approval select a setup.
"""
from qwen38_n1_verdict import require

NAME = "qwen36-vendor-coding-v1"
SETUPS = (NAME,)
EXTRA_MODULES = ("request_setup.py", "test_request_setup.py")
# Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0 README.md "Best Practices",
# "Thinking mode for precise coding tasks (e.g., WebDev)": temperature=0.6, top_p=0.95, top_k=20,
# min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0. hf2q accepts presence_penalty and min_p
# but its sampler does not consume them (src/serve/api/engine.rs, SamplingParams docs), so only
# these zero values have the same meaning on both engines.
SAMPLING = {NAME: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                   "presence_penalty": 0.0, "repetition_penalty": 1.0}}
# The peer's request field for the repetition penalty (tools/server/server-schema.cpp).
PEER_NAMES = {"repetition_penalty": "repeat_penalty"}
# Canonical preflight claim the setup replaces: greedy omissions, penalty 1.05, budgets 2048/512.
GREEDY_PREFLIGHT = "greedy_omissions_rep105_thinking2048_512"
SETUP_PREFLIGHT = {NAME: "vendor_coding_t06_p095_k20_rep10_unbudgeted_thinking"}


def validate(setup):
    require(setup in SETUPS, "unknown request setup")
    return setup


def validate_activation(setup, packet, family, engine_kind, slots):
    validate(setup)
    require(isinstance(packet, dict) and packet.get("request_setup") == setup,
            "explicit request-setup CLI and approval required")
    require(family == "qwen" and engine_kind in ("hf2q", "peer") and slots == 4,
            "the Qwen vendor request setup is four-slot hf2q or peer only")


def preflight(keys, setup):
    """The approval's preflight keys under a setup: the greedy-omission claim is replaced."""
    validate(setup)
    return (set(keys) - {GREEDY_PREFLIGHT}) | {SETUP_PREFLIGHT[setup]}


def sampling(setup):
    return dict(SAMPLING[validate(setup)])


def apply(request, setup):
    """Canonical (hf2q-shaped) request: add the setup's sampling fields; any prior value refuses."""
    fields = sampling(setup)
    require(not set(fields) & set(request), "request setup would overwrite a request field")
    request.update(fields)
    return request


def same(left, right):
    return type(left) is type(right) and left == right


def check(request, setup):
    """Every setup field present with its exact value and JSON type."""
    for key, value in sampling(setup).items():
        require(key in request and same(request[key], value), f"request setup drift: {key}")


def to_peer(request, setup):
    """Canonical request -> peer field names (unchanged values)."""
    check(request, setup)
    translated = dict(request)
    for canonical_name, peer_name in PEER_NAMES.items():
        require(peer_name not in translated, f"peer field {peer_name} already present")
        translated[peer_name] = translated.pop(canonical_name)
    return translated


def from_peer(request, setup):
    """Inverse of to_peer at the wire boundary."""
    canonical = dict(request)
    for canonical_name, peer_name in PEER_NAMES.items():
        require(peer_name in canonical and canonical_name not in canonical,
                f"peer request setup field {peer_name} missing")
        canonical[canonical_name] = canonical.pop(peer_name)
    check(canonical, setup)
    return canonical
