"""Distinct four-agent coding task instances for the ADR-063 Q1 quality campaign.

`qwen_n4_live_smoke.py --instance N` selects row N of fixtures/coding-instances.json.
Every row is a different parser of the same difficulty class (see the table's
`difficulty_class`): another function, unsigned type, inclusive bounds and test
values, with the canonical paths, tools, six-step protocol and test counts.
Without --instance nothing here is imported and the canonical parse_port
workload is byte-identical to earlier receipts.
"""
import hashlib
from pathlib import Path
import re
from string import Template

from qwen38_n1_verdict import WORKFLOW_STEPS, loads, require
from qwen38_n1_tools import LOCK, MANIFEST

TABLE = Path(__file__).with_name("fixtures") / "coding-instances.json"
SCHEMA = "hf2q.four-agent.coding-instances.v1"
# Approval sources added to the client's source freeze when --instance is used.
MODULES = ("coding_instances.py", "test_coding_instances.py")
MAXIMUM = {"u8": 2**8 - 1, "u16": 2**16 - 1, "u32": 2**32 - 1}
ROW_KEYS = {"id", "function", "type", "min", "max", "nouns", "slug", "word"}

LIBRARY = Template('''pub mod ports;
#[cfg(test)]
mod tests {
    use super::ports::$function;
    #[test]
    fn out_of_range_${slug}_rejected() {
        assert!($function("$below").is_err());
        assert!($function("$above").is_err());
    }
    #[test]
    fn boundary_${slug}_accepted() {
        assert_eq!($function("$min").unwrap(), $min);
        assert_eq!($function("$max").unwrap(), $max);
    }
}
''')
INITIAL = Template("pub fn $function(s: &str) -> $type { s.parse().unwrap() }\n")
TESTS = Template('''use qwen38_n1_fixture::ports::$function;
#[test]
fn rejects_invalid_$slug() {
    for s in [$rejects] {
        assert!($function(s).is_err(), "must reject {s}");
    }
}
#[test]
fn accepts_${type}_$slug() {
    assert_eq!($function("$min").unwrap(), $min);
    assert_eq!($function("$max").unwrap(), $max);
}
''')
TASK = Template("Fix $function to reject values below $min and above $max, accept valid $type $nouns")


class Workload:
    """One rendered instance: prompt text and fixture files."""

    def __init__(self, row, table_sha256):
        self.instance = row["id"]
        self.table_sha256 = table_sha256
        self.function = row["function"]
        values = {**row, "below": row["min"] - 1, "above": row["max"] + 1}
        task = TASK.substitute(values)
        self.user = task + ", and report the tests you ran."
        self.workflow = (WORKFLOW_STEPS + task +
                         f", and return a Result<{row['type']}, String> for invalid input rather than panic.")
        self.rejects = (str(row["min"] - 1), str(row["max"] + 1), row["word"], "-1",
                        str(MAXIMUM[row["type"]] + 1))
        self.immutable = {"Cargo.toml": MANIFEST, "Cargo.lock": LOCK,
                          "src/lib.rs": LIBRARY.substitute(values),
                          "tests/ports.rs": TESTS.substitute(
                              values, rejects=", ".join(f'"{s}"' for s in self.rejects))}
        self.initial = INITIAL.substitute(values)

    def binding(self):
        """Approval keys; q5 case approvals set them per instance."""
        return {"instance": self.instance, "instance_table_sha256": self.table_sha256}

    def record(self):
        """Client summary record."""
        return {"id": self.instance, "table_sha256": self.table_sha256}


def validate_row(row, index):
    require(isinstance(row, dict) and set(row) == ROW_KEYS and type(row["id"]) is int and row["id"] == index,
            "instance rows must be contiguous from 0")
    require(all(type(row[k]) is str for k in ("function", "type", "nouns", "slug", "word")), "instance text fields")
    require(re.fullmatch(r"parse_[a-z0-9_]+", row["function"]) is not None and row["function"] != "parse_port",
            "instance function name")
    require(row["type"] in MAXIMUM, "instance type must be u8, u16 or u32")
    require(all(type(row[k]) is int for k in ("min", "max")) and 0 < row["min"] < row["max"] < MAXIMUM[row["type"]],
            "instance bounds must be explicit on both sides: 0 < min < max < T::MAX")
    require(re.fullmatch(r"[a-z0-9_]+", row["slug"]) is not None, "instance test slug")
    require(re.fullmatch(r"[a-z]+", row["word"]) is not None, "instance non-numeric reject word")
    require(re.fullmatch(r"[A-Za-z0-9 -]+", row["nouns"]) is not None, "instance nouns")


def load(path=TABLE):
    raw = Path(path).read_bytes()
    table = loads(raw)
    require(isinstance(table, dict) and table.get("schema") == SCHEMA and
            isinstance(table.get("instances"), list) and table["instances"], "instance table schema")
    rows = table["instances"]
    for index, row in enumerate(rows):
        validate_row(row, index)
    for key in ("function", "slug"):
        require(len({row[key] for row in rows}) == len(rows), f"duplicate instance {key}")
    require(len({(row["type"], row["min"], row["max"]) for row in rows}) == len(rows),
            "duplicate instance type/range")
    return rows, hashlib.sha256(raw).hexdigest()


def select(instance, path=TABLE):
    rows, digest = load(path)
    require(type(instance) is int and 0 <= instance < len(rows),
            f"task instance must be 0..{len(rows) - 1}")
    return Workload(rows[instance], digest)
