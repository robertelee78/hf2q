# Exact native DeepSeek post-unload prefix oracle

Metadata-only validation helper; no engine, tensor, device or model allocation.
Four pure native-selection/anchor/build-identity controls plus one ignored archived-wire
replay. The existing Qwen acceptance function and its negative controls are
shared through test-only visibility; Qwen replay behavior is unchanged.
Production change is only internal visibility of RECOVERY_TAIL_TOKENS=8.

Replay family MUST be deepseek4, template the exact native source constant and
source label NativeEncoding{name:DEEPSEEK_V4_FLASH_0731}, tokenizer fingerprint
matching actual runtime. It calls production native render/tokenize helpers;
no Qwen template, approximate Jinja or engine/model allocation is allowed.
For each P, A=P without actual production8-tail; for each continuation require
priorA exact currentP prefix and actualcached>=priorA.length. Usageprompt must
match full native tokenization; coldcached0; seven8192reservations and actual
DS0/max requests remain bound. Changed earlier tokens or partial reuse fail.
This is not a guessed generation-cue allowance or proof of physical KV bytes.

Config is deny-unknown-fields: family,server_slots1or4, boundfiles(tokenizer,
template,readiness,controller_receipt,server_build_receipt),template_source,
seven turns{request,receipt},output. Boundfile={absolute path,lowercaseSHA256}.
Replay one seven-turn agent at a time; N4 requires four independently bound
configs/reviews, never one agent standing in for all four. Parent controller
must prove protocol/finalverification/cleanup, samePID as readiness, correct
terminalstatus. The referenced clean build receipt is bound to68804c57/d88efd5 public0154.
The helper alone does not attest which binary ran: compose its result with an
independent raw launch/binary/source/manifest review for the same runtime PID.
No production routing or unsupported family fallback is added.

Root runs only after all models unload using exact ignored test
`deepseek_exact_native_prefix_post_oracle` and
HF2Q_DEEPSEEK_PREFIX_POST_ORACLE_CONFIG. Outputcreate_new outside Git, mode0600,
contains actual token/render/evidence/source hashes and observed counts.
PassedExactNativeSevenWireReplay remains separate from model quality,
independent patch review, cancellation/growth, N4capacity and performance.
Retained rendered-inputs-reviewed metadata can be used if its semantic hashes
match actual readiness; do not reread100GB weights to export existing metadata.

Initial N1 validation passed all three pure controls and the actual seven-wire
replay after model unload. Actual prompt counts were708,804,988,1274,1554,1699,2047;
cached counts0,772,861,1217,1328,1615,1752 cover each prior native anchor.
Independent review also confirmed native encoder/reasoning code and the
render/tokenize seams match the executed server source. This is not a speed
comparison or full PR390/Q5 acceptance. Hosted CI runs the three pure controls;
the real archived-wire replay remains opt-in and ignored without its private config.

Follow-up validation passed four pure controls and four exact seven-wire N4
replays with explicit source `2d7599907be92d17736bc205806907589e845306`
and binary SHA256 `111649ce1b0d635d0a1403f34004667e1ab61757ff203c68c513dcac912488c2`.
This composes with the independent real-launch, four-patch, tool and cleanup
review; it does not qualify cancellation, physical growth, memory highwater
or performance.
