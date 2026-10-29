"""Host-only actual census parser controls; authored, not executed by author."""
import unittest
from shared_cancel_witness import census, MARKER

ROW = "Row { request: %s, handle: SlotHandle { slot_id: SlotId(%s), generation: 7 }, position: 128, token: 42 }"

def line(a=(0, 0), b=(1, 1)):
    return ("INFO " + MARKER + " width=2 raw_target_ns=100 owners=[" +
            ROW % a + ", " + ROW % b +
            "] singleton_attention_maps=128 shared_operator_maps=66 width_counts=[0,0,1,0,0]")

class CensusControls(unittest.TestCase):
    def test_actual_zero_request_and_distinct_owners(self):
        owners = census(line())
        self.assertEqual([r["request"] for r in owners], [0, 1])
        self.assertEqual([r["slot"] for r in owners], [0, 1])
    def test_role_or_http_overlap_never_witnesses_shared(self):
        self.assertIsNone(census("HTTP two open streams"))
    def test_duplicate_request_or_same_slot_refused(self):
        for bad in (line((0, 0), (0, 1)), line((0, 0), (1, 0))):
            with self.assertRaises(ValueError): census(bad)
    def test_wrong_width_missing_map_or_incomplete_owner_refused(self):
        for bad in (line().replace("width=2", "width=1"),
                    line().replace("shared_operator_maps=66", "shared_operator_maps=65"),
                    line().replace("singleton_attention_maps=128", "singleton_attention_maps=1280"),
                    line().replace(ROW % (1, 1), "Unknown")):
            with self.assertRaises(ValueError): census(bad)
    def test_out_of_profile_geometry_refused(self):
        with self.assertRaises(ValueError): census(line((0, 0), (1, 4)))

if __name__ == "__main__": unittest.main()
