//! Issue #727: the shared spill store is deterministic and bounded.

use super::{SlotSpillStore, SpillInsert};

fn stored<T>(outcome: SpillInsert<T>) -> (u64, Vec<u64>) {
    match outcome {
        SpillInsert::Stored { stamp, evicted } => (
            stamp,
            evicted.into_iter().map(|entry| entry.stamp).collect(),
        ),
        SpillInsert::Rejected(_) => panic!("entry rejected"),
    }
}

#[test]
fn store_evicts_oldest_stamps_to_hold_budget_and_entry_cap() {
    let mut store = SlotSpillStore::new(100, 3);
    assert!(store.is_empty());
    assert_eq!(stored(store.insert(40, 'a')), (0, vec![]));
    assert_eq!(stored(store.insert(40, 'b')), (1, vec![]));
    // 40 + 40 + 30 > 100: the oldest entry makes room.
    assert_eq!(stored(store.insert(30, 'c')), (2, vec![0]));
    assert_eq!(store.resident_bytes(), 70);
    assert_eq!(stored(store.insert(10, 'd')), (3, vec![]));
    // The entry cap (3) evicts the oldest even though bytes would fit.
    assert_eq!(stored(store.insert(5, 'e')), (4, vec![1]));
    assert_eq!(store.len(), 3);
    assert_eq!(store.resident_bytes(), 45);
    let newest: Vec<char> = store
        .newest_first()
        .map(|(_, entry)| entry.payload)
        .collect();
    assert_eq!(newest, vec!['e', 'd', 'c']);

    // An entry above the whole budget is refused and evicts nothing.
    match store.insert(101, 'x') {
        SpillInsert::Rejected(payload) => assert_eq!(payload, 'x'),
        SpillInsert::Stored { .. } => panic!("oversized entry stored"),
    }
    assert_eq!(store.len(), 3);

    // Taking an entry frees its bytes; stamps never repeat.
    let (index, _) = store
        .newest_first()
        .find(|(_, entry)| entry.payload == 'd')
        .expect("entry d");
    let taken = store.take(index);
    assert_eq!((taken.stamp, taken.bytes, taken.payload), (3, 10, 'd'));
    assert_eq!(store.resident_bytes(), 35);
    assert_eq!(stored(store.insert(65, 'f')), (5, vec![]));
    assert_eq!(stored(store.insert(1, 'g')), (6, vec![2]));
}

#[test]
fn store_outcome_depends_only_on_the_operation_sequence() {
    let run = || {
        let mut store = SlotSpillStore::new(64, 4);
        let mut log = Vec::new();
        for (step, bytes) in [17u64, 30, 9, 41, 12, 64, 3, 22].into_iter().enumerate() {
            if step == 5 {
                let index = store.newest_first().last().expect("entry").0;
                log.push(("take", store.take(index).stamp));
            }
            let (stamp, evicted) = stored(store.insert(bytes, step));
            log.push(("insert", stamp));
            log.extend(evicted.into_iter().map(|stamp| ("evict", stamp)));
        }
        (log, store.resident_bytes())
    };
    assert_eq!(run(), run());
}

#[test]
fn zero_budget_or_zero_entries_disables_the_store() {
    let mut store = SlotSpillStore::new(0, 4);
    assert!(!store.admits(1));
    assert!(matches!(store.insert(1, ()), SpillInsert::Rejected(())));
    let mut store = SlotSpillStore::new(1 << 30, 0);
    assert!(!store.admits(1));
    assert!(matches!(store.insert(1, ()), SpillInsert::Rejected(())));
}
