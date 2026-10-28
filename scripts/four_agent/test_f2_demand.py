"""Prospective policy controls through actual validation/qualification entrypoints.
Synthetic receipts are controls, never hardware or production speed evidence.
"""
import copy
import unittest
import f2_demand as demand
import f2_trace as trace
from test_f2_native_trace import fixture, record


def queued_fixture(count=1):
    records, identity, scenario = fixture()
    records[0]["payload"]["forward_demand_policy"] = dict(demand.CONFIG)
    scenario["forward_demand_policy"] = dict(demand.CONFIG)
    first = record(records, "slots")
    owner = first["payload"]["slots"][0]
    slots = []
    for i in range(count):
        slot = dict(owner, slot_id=i, request_id=5+i,
                    forward_demand_state="QueuedInputReady", selected_forward_owner=False)
        slots.append(slot)
    first["payload"].update(cause="owning_worker_work_state", slots=slots)
    quantum = dict(schema=trace.SCHEMA, seq=0, at_ns=0, kind="diagnostic_quantum",
                   payload=dict(family="qwen35", **demand.CONFIG, installed_before_admission=True))
    transaction = dict(schema=trace.SCHEMA, seq=0, at_ns=3, kind="transaction", payload={
        "family":"qwen35", "operation":"ordinary_or_temporal_mtp_decode",
        "logical_lanes":[{k:slots[0][k] for k in ("request_id","slot_id","generation")}],
        "physical_membership":"Unknown", "reason":"synthetic control, not GPU evidence"})
    busy = copy.deepcopy(first)
    busy["at_ns"] = 3
    busy["payload"]["cause"] = "owning_worker_enter_transaction"
    for i, slot in enumerate(busy["payload"]["slots"]):
        slot.update(runnable_forward=False, runnable_emit=False, inflight=None,
                    blocked="OwningWorkerTransaction", selected_forward_owner=(i==0),
                    forward_demand_state="SelectedForwardPending" if i==0 else "WaitingForOwningExecutor")
    records.insert(1, quantum)
    at = records.index(first)+1
    records[at:at] = [transaction, busy]
    for i,item in enumerate(records):
        item["seq"] = i
    return records,identity,scenario


class DemandControls(unittest.TestCase):
    def test_single_owner_service_qualifies_new_policy_without_ready_now_or_v2_reinterpretation(self):
        records,identity,scenario = queued_fixture()
        self.assertTrue(trace.validate_trace(records,identity,scenario)["qualified"])
        result=trace.qualify_window(records,(0,1),0,population_policy=demand.POLICY)
        self.assertEqual(result["status"],"Qualified")
        self.assertEqual(result["target_fraction"],1)
        self.assertEqual(result["ready_now_fraction"],0)
        self.assertIn("never Running",result["service_semantics"])
        self.assertEqual(trace.qualify_window(records,(0,1),0)["status"],"Unqualified")

    def test_four_queued_owners_do_not_fabricate_four_actual_services_or_physical_batch(self):
        records,identity,scenario=queued_fixture(4)
        validated=trace.validate_trace(records,identity,scenario)
        self.assertFalse(validated["shared_batch4_claim"])
        result=trace.qualify_window(records,(0,4),0,population_policy=demand.POLICY)
        self.assertEqual(result["target_fraction"],1)
        self.assertEqual(result["status"],"Unqualified")
        self.assertEqual(result["slots_without_required_forward_witness"],[[1,7],[2,7],[3,7]])
        self.assertEqual(result["slots_without_progress"],[[1,7],[2,7],[3,7]])

    def test_four_separate_synthetic_services_need_every_healthy_join(self):
        records,identity,scenario=queued_fixture(4)
        first=record(records,"slots")
        ready=copy.deepcopy(first)
        slots=ready["payload"]["slots"]
        templates=[copy.deepcopy(r) for r in records if r["kind"] in
                   ("forward_input","scope_definition","forward_witness","native_event")]
        transaction=copy.deepcopy(record(records,"transaction"))
        busy=copy.deepcopy(next(r for r in records if r["kind"]=="slots" and
                               r["payload"]["cause"]=="owning_worker_enter_transaction"))
        output=records[:records.index(first)+1]
        for lane in range(4):
            base=100*lane
            owner={k:slots[lane][k] for k in ("request_id","slot_id","generation")}
            row=dict(owner,token_position=123,token_count=1,phase="ordinary")
            scope={"epoch":10+lane,"id":20+lane}
            selected=copy.deepcopy(transaction);selected["at_ns"]=3+base
            selected["payload"]["logical_lanes"]=[owner]
            output.append(selected)
            state=copy.deepcopy(busy);state["at_ns"]=3+base
            for i,slot in enumerate(state["payload"]["slots"]):
                slot.update(selected_forward_owner=i==lane,forward_demand_state=
                            "SelectedForwardPending" if i==lane else "WaitingForOwningExecutor")
            output.append(state)
            for index,template in enumerate(templates):
                item=copy.deepcopy(template);item["at_ns"]=4+base+index
                payload=item["payload"]
                if item["kind"]=="forward_input":
                    payload.update(forward_id=30+lane,epoch=10+lane,
                        packed_rows=[dict(matrix_row_start=0,matrix_row_end=1,logical=row)])
                elif item["kind"]=="scope_definition":payload.update(scope=scope,rows=[row])
                elif item["kind"]=="forward_witness":payload.update(forward_id=30+lane,scope=scope)
                else:
                    payload.update(command_id=1+lane,invocation_epoch=10+lane,
                        begin_monotonic_ns=1000+item["at_ns"],end_monotonic_ns=1000+item["at_ns"])
                    if payload["scope"] is not None:payload["scope"]=scope
                output.append(item)
            restored=copy.deepcopy(ready);restored["at_ns"]=80+base;output.append(restored)
        final=copy.deepcopy(record(records,"native_final"));final["payload"]["events"]=20
        output.extend([final,copy.deepcopy(record(records,"end"))])
        for i,item in enumerate(output):item["seq"]=i
        validated=trace.validate_trace(output,identity,scenario)
        self.assertFalse(validated["shared_batch4_claim"])
        result=trace.qualify_window(output,(0,4),0,population_policy=demand.POLICY)
        self.assertEqual(result["status"],"Qualified")
        self.assertEqual(result["slots_without_required_forward_witness"],[])
        broken=copy.deepcopy(output)
        record(broken,"native_event","CheckedDrain")["payload"]["checked_status"]=5
        with self.assertRaises(ValueError):trace.validate_trace(broken,identity,scenario)

    def test_policy_install_owner_and_pending_output_cannot_be_resealed_into_valid_demand(self):
        for fault in ("missing_install","late_install","wrong_quantum","bool_quantum","owner_union",
                      "waiter_runnable","output_only","stale_demand","unbound_policy","wrong_family"):
            r,identity,scenario=queued_fixture(4)
            busy=next(x for x in r if x["kind"]=="slots" and
                      x["payload"]["cause"]=="owning_worker_enter_transaction")
            if fault=="missing_install":r.remove(record(r,"diagnostic_quantum"))
            elif fault=="late_install":
                install=record(r,"diagnostic_quantum");r.remove(install)
                install["at_ns"]=3;r.insert(r.index(busy)+1,install)
            elif fault=="wrong_quantum":record(r,"diagnostic_quantum")["payload"]["qwen_prefill_chunk_tokens"]=64
            elif fault=="bool_quantum":r[0]["payload"]["forward_demand_policy"]["qwen_prefill_chunk_tokens"]=True
            elif fault=="owner_union":busy["payload"]["slots"][1]["selected_forward_owner"]=True
            elif fault=="waiter_runnable":busy["payload"]["slots"][1]["runnable_forward"]=True
            elif fault=="output_only":
                first=record(r,"slots")["payload"]["slots"][1]
                first.update(runnable_forward=False,blocked="VerifiedOutputPendingSemanticUnknown",
                             forward_demand_state="NoForwardDemand")
            elif fault=="stale_demand":busy["payload"]["slots"][1]["generation"]+=1
            elif fault=="unbound_policy":scenario.pop("forward_demand_policy")
            elif fault=="wrong_family":
                r[0]["payload"]["native_observer"]["witness_manifest"]["input_sources"]=["gemma4.scalar_forward"]
            for i,item in enumerate(r):item["seq"]=i
            with self.assertRaises(ValueError,msg=fault):
                trace.validate_trace(r,identity,scenario)

    def test_first_window_coverage_threshold_is_not_waived_by_queued_demand(self):
        records,identity,scenario=queued_fixture(4)
        stopped=copy.deepcopy(record(records,"slots"))
        stopped["at_ns"]=1_500_000_000
        for slot in stopped["payload"]["slots"]:
            slot.update(runnable_forward=False,forward_eligibility=False,
                        blocked="ClientClosed",forward_demand_state="NoForwardDemand")
        at=next(i for i,r in enumerate(records) if r["at_ns"]>stopped["at_ns"])
        records.insert(at,stopped)
        for i,item in enumerate(records):item["seq"]=i
        trace.validate_trace(records,identity,scenario)
        result=trace.qualify_window(records,(0,4),0,population_policy=demand.POLICY)
        self.assertEqual(result["start_ns"],3)
        self.assertEqual(result["end_ns"],2_000_000_003)
        self.assertLess(result["target_fraction"],0.8)
        self.assertEqual(result["status"],"Unqualified")

    def test_old_trace_cannot_be_classified_as_new_policy(self):
        records,identity,scenario=fixture()
        trace.validate_trace(records,identity,scenario)
        with self.assertRaises(ValueError):
            trace.qualify_window(records,(0,1),0,population_policy=demand.POLICY)
