"""Closed prospective owner-demand policy; never physical batching evidence."""
POLICY = "queued_forward_demand_v3"
CONFIG = {"population_policy": POLICY, "qwen_prefill_chunk_tokens": 128}
SLOT_FIELDS = {"forward_demand_state", "selected_forward_owner"}
ACTIVE = {"QueuedInputReady", "SelectedForwardPending", "WaitingForOwningExecutor"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_policy(header, scenario):
    policy = header["forward_demand_policy"]
    require(type(policy) is dict and set(policy) == set(CONFIG) and
            type(policy["qwen_prefill_chunk_tokens"]) is int and policy == CONFIG and
            scenario.get("forward_demand_policy") == CONFIG,
            "exact prospective queued-demand/128 config binding required")
    require(type(header.get("native_observer")) is dict and
            header["native_observer"].get("diagnostic_case") is True,
            "queued demand requires actual diagnostic native observer")
    sources = header["native_observer"].get("witness_manifest", {}).get("input_sources")
    require(type(sources) is list and sources and all(s in
            ("qwen35.greedy.actual_tokens", "qwen35.forward.actual_tokens") for s in sources),
            "queued demand requires Qwen actual-input witnesses")


def key(slot):
    return tuple(slot[k] for k in ("request_id", "slot_id", "generation", "phase"))


def input_ready(slot):
    return (slot.get("forward_eligibility") is True and
            slot["runnable_forward"] is True and slot["blocked"] is None)


class Validator:
    def __init__(self):
        self.ready = {}
        self.selected = set()
        self.installed = False
        self.admitted = False

    def quantum(self, payload, header, latest):
        require("forward_demand_policy" in header and not self.installed and
                not self.admitted and not latest and
                set(payload) == {"family", "population_policy", "qwen_prefill_chunk_tokens",
                                 "installed_before_admission"} and
                payload["family"] == "qwen35" and payload["population_policy"] == POLICY and
                type(payload["qwen_prefill_chunk_tokens"]) is int and
                payload["qwen_prefill_chunk_tokens"] == 128 and
                payload["installed_before_admission"] is True,
                "actual quantum installation must precede admission exactly once")
        self.installed = True

    def transaction(self, payload, header):
        if "forward_demand_policy" not in header:
            return
        require(payload["family"] == "qwen35", "Qwen diagnostic policy is family explicit")
        self.selected = {(s["request_id"], s["slot_id"], s["generation"])
                         for s in payload["logical_lanes"]}

    def snapshot(self, payload, header):
        if "forward_demand_policy" not in header:
            return
        slots, cause = payload["slots"], payload["cause"]
        if slots:
            require(self.installed, "admission before actual diagnostic quantum install")
            self.admitted = True
        for slot in slots:
            require(SLOT_FIELDS <= set(slot) and type(slot["selected_forward_owner"]) is bool,
                    "typed owner-demand fields required")
            selected = (slot["request_id"], slot["slot_id"], slot["generation"]) in self.selected
            expected = "Unknown"
            if cause == "owning_worker_work_state":
                expected = ("QueuedInputReady" if input_ready(slot) else
                            "NoForwardDemand" if type(slot["runnable_forward"]) is bool else "Unknown")
                selected = False
            elif cause == "owning_worker_enter_transaction":
                old = self.ready.get(key(slot))
                if old is not None and input_ready(old) and slot.get("forward_eligibility") is True:
                    expected = "SelectedForwardPending" if selected else "WaitingForOwningExecutor"
                elif old is not None and slot.get("forward_eligibility") is False:
                    expected = "NoForwardDemand"
                require(slot["runnable_forward"] is False and slot["runnable_emit"] is False,
                        "transaction does not confer executor availability")
            else:
                selected = False
            require(slot["forward_demand_state"] == expected and
                    slot["selected_forward_owner"] is selected,
                    "owner demand cannot be fabricated from phase/active/eligibility alone")
        if cause == "owning_worker_work_state":
            self.ready = {key(s): dict(s) for s in slots}
            self.selected.clear()
        elif cause != "owning_worker_enter_transaction":
            self.ready.clear()
            self.selected.clear()
        else:
            self.selected.clear()  # one transaction record authorizes one owner snapshot

    def finish(self, header):
        require("forward_demand_policy" not in header or self.installed,
                "missing actual quantum install acknowledgement")
