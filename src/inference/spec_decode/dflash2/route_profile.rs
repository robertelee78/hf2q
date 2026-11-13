//! Route profile of each pinned Qwen DFlash2 descriptor (ADR-067 D6).
//!
//! [`SWITCHES`] is the one table of fast-route switches. Each row names a
//! switch, the switch it cannot run without, the prepared planes it keeps
//! resident, and the value the measured combined configuration of each
//! pinned Qwen descriptor used. A switch is resolved once per model load:
//!
//! 1. a non-blank process environment value wins, in either direction, so
//!    every switch stays an individual escape hatch and A/B arm;
//! 2. otherwise the profile's value, when the DFlash2 policy selected the
//!    profile for the loaded model, unless the policy withheld the row for
//!    this artifact (its kernels do not admit the storage, or its planes do
//!    not fit in memory) or the switch it requires did not resolve on;
//! 3. otherwise the switch is unset and its reader applies the default it
//!    had before profiles existed.
//!
//! The resolved set belongs to one model. Its load thread and its worker
//! thread bind it ([`bind`], [`bind_thread`]); a thread that binds nothing
//! reads the process environment alone, so every path outside `hf2q serve`
//! keeps the behaviour it had before profiles existed. Two models resident
//! in one process therefore never read each other's profile.
//!
//! Adding a switch to the bundle is one row here; its reader takes the value
//! from [`with_value`] or [`is_one`] instead of the process environment.

use std::cell::{Cell, RefCell};
use std::sync::{Arc, LazyLock};

use super::schema::{DFlash2CheckpointDescriptor, QWEN36_DFLASH2, QWEN38_DFLASH2};

/// Prepared planes a switch keeps resident beside the GGUF weights. Each set
/// is a second copy, in the prepared layout, of the named weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PlaneSet {
    /// Dense FFN gate, up and down.
    DenseFfn,
    /// DeltaNet and full-attention projections.
    Projections,
    /// Output head.
    Head,
    /// Routed MoE expert stacks.
    MoeExperts,
    /// Shared MoE expert.
    MoeShared,
}

/// One fast-route switch and its value in each descriptor's profile.
pub(crate) struct SwitchRow {
    pub(crate) name: &'static str,
    /// A switch earlier in the table that this one cannot run without. Its
    /// profile value applies only when that switch resolved on.
    pub(crate) requires: Option<&'static str>,
    /// Plane sets the switch prepares when on.
    pub(crate) planes: &'static [PlaneSet],
    /// Value in the Qwen3.8 dense profile; `None` leaves the switch unset.
    pub(crate) qwen38: Option<&'static str>,
    /// Value in the Qwen3.6 MoE profile; `None` leaves the switch unset.
    pub(crate) qwen36: Option<&'static str>,
}

const fn row(
    name: &'static str,
    requires: Option<&'static str>,
    planes: &'static [PlaneSet],
    qwen38: Option<&'static str>,
    qwen36: Option<&'static str>,
) -> SwitchRow {
    SwitchRow {
        name,
        requires,
        planes,
        qwen38,
        qwen36,
    }
}

const ON: Option<&str> = Some("1");
const AF4: Option<&str> = Some("HF2Q_DFLASH2_DRAFT_AF4_G64");
const M32: Option<&str> = Some("HF2Q_REPACKED_GGUF_M32");
const PREFILL: Option<&str> = Some("HF2Q_QWEN_PREFILL_PREPARED");
const BATCHED: Option<&str> = Some("HF2Q_QWEN_MOE_BATCHED");
const STAGED_M8: Option<&str> = Some("HF2Q_QWEN_MOE_STAGED_M8");
use PlaneSet::{DenseFfn, Head, MoeExperts, MoeShared, Projections};

/// The route profile table: one row per switch, one column per descriptor.
#[rustfmt::skip]
pub(crate) const SWITCHES: &[SwitchRow] = &[
    //  switch                              requires   planes                     Qwen3.8  Qwen3.6
    row("HF2Q_DFLASH2_DRAFT_AF4_G64",       None,      &[],                       ON,      ON),
    row("HF2Q_DFLASH2_DRAFT_STAGED",        AF4,       &[Head],                   ON,      ON),
    row("HF2Q_DFLASH2_GPU_TARGET_TOKENS",   None,      &[],                       None,    ON),
    row("HF2Q_DFLASH2_CONTEXT_FIRST",       None,      &[],                       ON,      ON),
    // Per-model readers; no profile value until the combined configuration
    // with them is qualified.
    row("HF2Q_DFLASH2_DRAFT_PREFETCH",      None,      &[],                       None,    None),
    row("HF2Q_DFLASH2_GPU_DECISIONS",       None,      &[],                       None,    None),
    row("HF2Q_REPACKED_GGUF_M32",           None,      &[DenseFfn],               ON,      ON),
    row("HF2Q_REPACKED_GGUF_M32_PROJ",      M32,       &[Projections],            ON,      ON),
    row("HF2Q_REPACKED_GGUF_M32_HEAD",      M32,       &[Head],                   ON,      ON),
    row("HF2Q_REPACKED_GGUF_ANY_WIDTH",     M32,       &[],                       ON,      ON),
    row("HF2Q_REPACKED_GGUF_SHARED_HEAD",   None,      &[],                       ON,      ON),
    row("HF2Q_QWEN_VERIFY_ATTN_TILE",       None,      &[],                       ON,      ON),
    row("HF2Q_QWEN_VERIFY_MIXER_FUSED",     None,      &[],                       ON,      ON),
    row("HF2Q_QWEN_VERIFY_REPLAY_REUSE",    None,      &[],                       ON,      ON),
    row("HF2Q_QWEN_PREFILL_PREPARED",       None,      &[DenseFfn, Projections],  ON,      ON),
    row("HF2Q_QWEN_PREFILL_FOLD_CUE",       PREFILL,   &[],                       ON,      ON),
    row("HF2Q_QWEN_MOE_BATCHED",            None,      &[],                       None,    ON),
    row("HF2Q_QWEN_MOE_STAGED_M8",          BATCHED,   &[MoeExperts],             None,    ON),
    row("HF2Q_QWEN_MOE_LIVE_TILES",         STAGED_M8, &[MoeShared],              None,    ON),
];

const SWITCH_COUNT: usize = SWITCHES.len();
const _: () = assert!(SWITCH_COUNT <= u32::BITS as usize);

/// Index of a switch in [`SWITCHES`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Switch(usize);

const fn same(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// The table row of `name`. Evaluated in a `const`, a name that is not in
/// the table fails the build.
pub(crate) const fn switch(name: &str) -> Switch {
    let mut index = 0;
    while index < SWITCH_COUNT {
        if same(SWITCHES[index].name, name) {
            return Switch(index);
        }
        index += 1;
    }
    panic!("switch is not in the DFlash2 route profile table")
}

/// Column of the table: the profile of one pinned Qwen descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RouteProfile {
    Qwen38,
    Qwen36,
}

impl RouteProfile {
    /// The profile of `descriptor`, if it has one.
    pub(crate) fn of(descriptor: &DFlash2CheckpointDescriptor) -> Option<Self> {
        if descriptor.id == QWEN38_DFLASH2.id {
            Some(Self::Qwen38)
        } else if descriptor.id == QWEN36_DFLASH2.id {
            Some(Self::Qwen36)
        } else {
            None
        }
    }

    /// The descriptor this profile belongs to.
    #[cfg(test)]
    pub(crate) fn descriptor(self) -> &'static DFlash2CheckpointDescriptor {
        match self {
            Self::Qwen38 => &QWEN38_DFLASH2,
            Self::Qwen36 => &QWEN36_DFLASH2,
        }
    }

    fn value(self, row: &SwitchRow) -> Option<&'static str> {
        match self {
            Self::Qwen38 => row.qwen38,
            Self::Qwen36 => row.qwen36,
        }
    }
}

/// Where a switch's resolved value came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SwitchSource {
    /// The process environment.
    Env,
    /// The active route profile.
    Profile,
    /// Neither: the reader's own default.
    Default,
    /// The profile has a value, and it was not applied to this artifact;
    /// the reader's own default stands.
    Withheld,
}

impl SwitchSource {
    fn label(self) -> &'static str {
        match self {
            Self::Env => "env",
            Self::Profile => "profile",
            Self::Default => "default",
            Self::Withheld => "withheld",
        }
    }
}

/// Whether a raw switch value turns its route on, for the rules that are
/// decided before a reader parses the value.
pub(crate) fn turns_on(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "on" | "true"
        )
    })
}

/// Every table switch resolved for one model load.
#[derive(Debug)]
pub(crate) struct ResolvedRoutes {
    profile: Option<RouteProfile>,
    values: [Option<String>; SWITCH_COUNT],
    sources: [SwitchSource; SWITCH_COUNT],
    /// Why a profile value was withheld, per switch.
    withheld: [Option<String>; SWITCH_COUNT],
    /// Bit `i` is set when switch `i` resolved to exactly `1`.
    ones: u32,
}

impl ResolvedRoutes {
    /// Resolve every switch. `profile` is the active profile, if the policy
    /// selected one; `withhold` gives the reason a row's profile value must
    /// not apply to this artifact, if any; `env` reads one process
    /// environment variable.
    pub(crate) fn resolve(
        profile: Option<RouteProfile>,
        withhold: &dyn Fn(&SwitchRow) -> Option<String>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Self {
        let mut values: [Option<String>; SWITCH_COUNT] = std::array::from_fn(|_| None);
        let mut sources = [SwitchSource::Default; SWITCH_COUNT];
        let mut withheld: [Option<String>; SWITCH_COUNT] = std::array::from_fn(|_| None);
        let mut ones = 0;
        for (index, row) in SWITCHES.iter().enumerate() {
            let explicit = env(row.name).filter(|value| !value.trim().is_empty());
            let (value, source) = match (explicit, profile.and_then(|profile| profile.value(row))) {
                (Some(value), _) => (Some(value), SwitchSource::Env),
                (None, None) => (None, SwitchSource::Default),
                (None, Some(value)) => {
                    let unmet = row
                        .requires
                        .filter(|required| !turns_on(values[switch(required).0].as_deref()));
                    let reason = match unmet {
                        Some(required) => Some(format!("{required} is off")),
                        None => withhold(row),
                    };
                    match reason {
                        Some(reason) => {
                            withheld[index] = Some(reason);
                            (None, SwitchSource::Withheld)
                        }
                        None => (Some(value.to_owned()), SwitchSource::Profile),
                    }
                }
            };
            if value.as_deref() == Some("1") {
                ones |= 1 << index;
            }
            values[index] = value;
            sources[index] = source;
        }
        Self {
            profile,
            values,
            sources,
            withheld,
            ones,
        }
    }

    /// The process environment alone: no profile.
    pub(crate) fn from_process_env() -> Self {
        Self::resolve(None, &|_| None, &|name| std::env::var(name).ok())
    }

    /// The active profile, if any.
    pub(crate) fn profile(&self) -> Option<RouteProfile> {
        self.profile
    }

    /// Whether any profile value was withheld from this artifact.
    pub(crate) fn any_withheld(&self) -> bool {
        self.withheld.iter().any(Option::is_some)
    }

    /// `NAME: reason` for every withheld profile value, in table order.
    pub(crate) fn withheld_audit(&self) -> String {
        let reasons: Vec<String> = SWITCHES
            .iter()
            .zip(&self.withheld)
            .filter_map(|(row, reason)| Some(format!("{}: {}", row.name, reason.as_deref()?)))
            .collect();
        if reasons.is_empty() {
            "none".to_owned()
        } else {
            reasons.join("; ")
        }
    }

    /// The resolved value of `switch`; `None` is unset.
    pub(crate) fn value(&self, switch: Switch) -> Option<&str> {
        self.values[switch.0].as_deref()
    }

    /// Where the value of `switch` came from.
    #[cfg(test)]
    pub(crate) fn source(&self, switch: Switch) -> SwitchSource {
        self.sources[switch.0]
    }

    /// Whether `switch` resolved to exactly `1`.
    #[cfg(test)]
    pub(crate) fn is_one(&self, switch: Switch) -> bool {
        self.ones & (1 << switch.0) != 0
    }

    /// Whether the value of `switch` turns its route on (`1`, `on`, `true`).
    pub(crate) fn is_on(&self, switch: Switch) -> bool {
        turns_on(self.values[switch.0].as_deref())
    }

    /// Every switch as `(name, value, source, why withheld)`, in table order.
    pub(crate) fn entries(
        &self,
    ) -> impl Iterator<Item = (&'static str, Option<&str>, &'static str, Option<&str>)> {
        SWITCHES.iter().enumerate().map(|(index, row)| {
            (
                row.name,
                self.values[index].as_deref(),
                self.sources[index].label(),
                self.withheld[index].as_deref(),
            )
        })
    }

    /// Plane sets of every row whose value turns its route on. A row is on
    /// when its value is `1`, `on` or `true`; the plane estimate is taken
    /// before a reader could reject another value.
    pub(crate) fn plane_sets(&self) -> Vec<PlaneSet> {
        let mut sets = Vec::new();
        for (index, row) in SWITCHES.iter().enumerate() {
            if turns_on(self.values[index].as_deref()) {
                for set in row.planes {
                    if !sets.contains(set) {
                        sets.push(*set);
                    }
                }
            }
        }
        sets
    }

    /// `NAME=value(source)` for every table switch, in table order.
    pub(crate) fn audit(&self) -> String {
        SWITCHES
            .iter()
            .enumerate()
            .map(|(index, row)| {
                format!(
                    "{}={}({})",
                    row.name,
                    self.values[index].as_deref().unwrap_or("unset"),
                    self.sources[index].label()
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

thread_local! {
    static BOUND: RefCell<Option<Arc<ResolvedRoutes>>> = const { RefCell::new(None) };
    /// The bound routes' exact-`1` bits, for readers on the dispatch path.
    static BOUND_ONES: Cell<Option<u32>> = const { Cell::new(None) };
}

static PROCESS: LazyLock<Arc<ResolvedRoutes>> =
    LazyLock::new(|| Arc::new(ResolvedRoutes::from_process_env()));

/// The routes bound to this thread, or the process environment alone.
pub(crate) fn current() -> Arc<ResolvedRoutes> {
    BOUND
        .with(|bound| bound.borrow().clone())
        .unwrap_or_else(|| Arc::clone(&PROCESS))
}

/// Apply `read` to the resolved value of `switch` on this thread; `None` is
/// unset.
pub(crate) fn with_value<R>(switch: Switch, read: impl FnOnce(Option<&str>) -> R) -> R {
    BOUND.with(|bound| match bound.borrow().as_deref() {
        Some(routes) => read(routes.value(switch)),
        None => read(PROCESS.value(switch)),
    })
}

/// Whether `switch` resolved to exactly `1` on this thread.
pub(crate) fn is_one(switch: Switch) -> bool {
    let ones = BOUND_ONES.with(Cell::get).unwrap_or_else(|| PROCESS.ones);
    ones & (1 << switch.0) != 0
}

fn replace(routes: Option<Arc<ResolvedRoutes>>) -> Option<Arc<ResolvedRoutes>> {
    BOUND_ONES.with(|ones| ones.set(routes.as_ref().map(|routes| routes.ones)));
    BOUND.with(|bound| bound.replace(routes))
}

/// Restores the thread's previous binding when dropped.
pub(crate) struct Bound {
    previous: Option<Arc<ResolvedRoutes>>,
}

/// Bind `routes` to this thread until the guard drops.
#[must_use]
pub(crate) fn bind(routes: Arc<ResolvedRoutes>) -> Bound {
    Bound {
        previous: replace(Some(routes)),
    }
}

/// Bind `routes` to this thread for the rest of its life (a model's worker).
pub(crate) fn bind_thread(routes: Arc<ResolvedRoutes>) {
    replace(Some(routes));
}

impl Drop for Bound {
    fn drop(&mut self) {
        replace(self.previous.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        move |name| map.get(name).cloned()
    }

    const M32: Switch = switch("HF2Q_REPACKED_GGUF_M32");
    const PREFILL: Switch = switch("HF2Q_QWEN_PREFILL_PREPARED");
    const MOE_BATCHED: Switch = switch("HF2Q_QWEN_MOE_BATCHED");
    const MOE_STAGED: Switch = switch("HF2Q_QWEN_MOE_STAGED_M8");
    const GPU_TOKENS: Switch = switch("HF2Q_DFLASH2_GPU_TARGET_TOKENS");
    const ATTN_TILE: Switch = switch("HF2Q_QWEN_VERIFY_ATTN_TILE");

    #[test]
    fn table_names_are_unique_hf2q_switches() {
        for (index, row) in SWITCHES.iter().enumerate() {
            assert!(row.name.starts_with("HF2Q_"), "{}", row.name);
            assert_eq!(switch(row.name), Switch(index), "duplicate {}", row.name);
        }
    }

    #[test]
    fn without_a_profile_every_switch_is_the_environment_or_unset() {
        let none = ResolvedRoutes::resolve(None, &|_| None, &env(&[]));
        assert_eq!(none.ones, 0);
        for index in 0..SWITCH_COUNT {
            assert_eq!(none.value(Switch(index)), None);
            assert_eq!(none.source(Switch(index)), SwitchSource::Default);
        }
        assert!(none.plane_sets().is_empty());
        let explicit = ResolvedRoutes::resolve(
            None,
            &|_| None,
            &env(&[
                ("HF2Q_REPACKED_GGUF_M32", "1"),
                ("HF2Q_QWEN_MOE_BATCHED", "on"),
            ]),
        );
        assert!(explicit.is_one(M32));
        assert_eq!(explicit.source(M32), SwitchSource::Env);
        assert_eq!(explicit.value(MOE_BATCHED), Some("on"));
        assert!(!explicit.is_one(MOE_BATCHED), "`on` is not exactly 1");
        assert_eq!(explicit.value(PREFILL), None);
        assert_eq!(explicit.plane_sets(), vec![PlaneSet::DenseFfn]);
    }

    #[test]
    fn profiles_carry_the_measured_combined_configuration() {
        let dense = ResolvedRoutes::resolve(Some(RouteProfile::Qwen38), &|_| None, &env(&[]));
        for name in [
            "HF2Q_DFLASH2_DRAFT_AF4_G64",
            "HF2Q_DFLASH2_DRAFT_STAGED",
            "HF2Q_DFLASH2_CONTEXT_FIRST",
            "HF2Q_QWEN_VERIFY_MIXER_FUSED",
            "HF2Q_QWEN_VERIFY_REPLAY_REUSE",
            "HF2Q_QWEN_PREFILL_FOLD_CUE",
            "HF2Q_REPACKED_GGUF_M32",
            "HF2Q_REPACKED_GGUF_M32_PROJ",
            "HF2Q_REPACKED_GGUF_M32_HEAD",
            "HF2Q_REPACKED_GGUF_ANY_WIDTH",
            "HF2Q_REPACKED_GGUF_SHARED_HEAD",
            "HF2Q_QWEN_VERIFY_ATTN_TILE",
            "HF2Q_QWEN_PREFILL_PREPARED",
        ] {
            assert!(dense.is_one(switch(name)), "{name}");
            assert_eq!(dense.source(switch(name)), SwitchSource::Profile, "{name}");
        }
        // The dense descriptor keeps its own GPU-token default and has no
        // MoE layers.
        for name in [
            "HF2Q_DFLASH2_GPU_TARGET_TOKENS",
            "HF2Q_QWEN_MOE_BATCHED",
            "HF2Q_QWEN_MOE_STAGED_M8",
            "HF2Q_QWEN_MOE_LIVE_TILES",
        ] {
            assert_eq!(dense.value(switch(name)), None, "{name}");
            assert_eq!(dense.source(switch(name)), SwitchSource::Default, "{name}");
        }
        let moe = ResolvedRoutes::resolve(Some(RouteProfile::Qwen36), &|_| None, &env(&[]));
        for row in SWITCHES.iter().filter(|row| row.qwen36.is_some()) {
            assert!(moe.is_one(switch(row.name)), "{}", row.name);
        }
        // Rows whose readers are per model and whose combined configuration
        // is not qualified yet carry no profile value.
        for name in ["HF2Q_DFLASH2_DRAFT_PREFETCH", "HF2Q_DFLASH2_GPU_DECISIONS"] {
            for routes in [&dense, &moe] {
                assert_eq!(routes.value(switch(name)), None, "{name}");
                assert_eq!(routes.source(switch(name)), SwitchSource::Default, "{name}");
            }
        }
        assert_eq!(
            moe.plane_sets(),
            vec![
                PlaneSet::Head,
                PlaneSet::DenseFfn,
                PlaneSet::Projections,
                PlaneSet::MoeExperts,
                PlaneSet::MoeShared
            ]
        );
    }

    #[test]
    fn an_explicit_value_overrides_the_profile_in_both_directions() {
        let routes = ResolvedRoutes::resolve(
            Some(RouteProfile::Qwen38),
            &|_| None,
            &env(&[
                ("HF2Q_QWEN_PREFILL_PREPARED", "0"),
                ("HF2Q_QWEN_VERIFY_ATTN_TILE", "0"),
                ("HF2Q_DFLASH2_GPU_TARGET_TOKENS", "0"),
                ("HF2Q_QWEN_MOE_BATCHED", "1"),
                ("HF2Q_REPACKED_GGUF_M32", "  "),
            ]),
        );
        assert_eq!(routes.value(PREFILL), Some("0"));
        assert_eq!(routes.source(PREFILL), SwitchSource::Env);
        assert!(!routes.is_one(ATTN_TILE));
        assert_eq!(routes.value(GPU_TOKENS), Some("0"));
        // On although the dense profile leaves it unset.
        assert!(routes.is_one(MOE_BATCHED));
        assert_eq!(routes.source(MOE_BATCHED), SwitchSource::Env);
        // A blank value is unset: the profile applies.
        assert!(routes.is_one(M32));
        assert_eq!(routes.source(M32), SwitchSource::Profile);
    }

    #[test]
    fn a_withheld_row_keeps_its_default_and_takes_its_dependents_with_it() {
        // The policy withholds the batched MoE route; staged M8 and the live
        // tiles cannot run without it. An explicit value is never withheld.
        let routes = ResolvedRoutes::resolve(
            Some(RouteProfile::Qwen36),
            &|row| {
                (row.name == "HF2Q_QWEN_MOE_BATCHED").then(|| "shared expert is Q5_K".to_owned())
            },
            &env(&[("HF2Q_REPACKED_GGUF_M32", "0")]),
        );
        for name in [
            "HF2Q_QWEN_MOE_BATCHED",
            "HF2Q_QWEN_MOE_STAGED_M8",
            "HF2Q_QWEN_MOE_LIVE_TILES",
            "HF2Q_REPACKED_GGUF_M32_PROJ",
            "HF2Q_REPACKED_GGUF_M32_HEAD",
            "HF2Q_REPACKED_GGUF_ANY_WIDTH",
        ] {
            assert_eq!(routes.value(switch(name)), None, "{name}");
            assert_eq!(
                routes.source(switch(name)),
                SwitchSource::Withheld,
                "{name}"
            );
        }
        assert_eq!(routes.source(M32), SwitchSource::Env);
        assert!(
            routes.is_one(PREFILL),
            "an unrelated row keeps its profile value"
        );
        assert!(routes.any_withheld());
        let audit = routes.withheld_audit();
        assert!(
            audit.contains("HF2Q_QWEN_MOE_BATCHED: shared expert is Q5_K"),
            "{audit}"
        );
        assert!(
            audit.contains("HF2Q_QWEN_MOE_STAGED_M8: HF2Q_QWEN_MOE_BATCHED is off"),
            "{audit}"
        );
        assert!(
            audit.contains("HF2Q_REPACKED_GGUF_M32_PROJ: HF2Q_REPACKED_GGUF_M32 is off"),
            "{audit}"
        );
        assert_eq!(
            routes.plane_sets(),
            vec![PlaneSet::Head, PlaneSet::DenseFfn, PlaneSet::Projections]
        );
        assert!(routes
            .audit()
            .contains("HF2Q_QWEN_MOE_BATCHED=unset(withheld)"));

        // An explicit dependent survives a withheld requirement: the reader
        // decides, as it does without a profile.
        let explicit = ResolvedRoutes::resolve(
            Some(RouteProfile::Qwen36),
            &|row| (row.name == "HF2Q_QWEN_MOE_BATCHED").then(|| "test".to_owned()),
            &env(&[("HF2Q_QWEN_MOE_STAGED_M8", "1")]),
        );
        assert!(explicit.is_one(MOE_STAGED));
        assert_eq!(explicit.source(MOE_STAGED), SwitchSource::Env);

        let full = ResolvedRoutes::resolve(Some(RouteProfile::Qwen36), &|_| None, &env(&[]));
        assert!(!full.any_withheld());
        assert_eq!(full.withheld_audit(), "none");
    }

    #[test]
    fn every_requirement_names_an_earlier_row() {
        for (index, row) in SWITCHES.iter().enumerate() {
            if let Some(required) = row.requires {
                assert!(
                    switch(required).0 < index,
                    "{} requires {required}",
                    row.name
                );
            }
        }
    }

    #[test]
    fn audit_names_every_switch_with_its_value_and_source() {
        let routes = ResolvedRoutes::resolve(
            Some(RouteProfile::Qwen38),
            &|_| None,
            &env(&[("HF2Q_QWEN_VERIFY_ATTN_TILE", "0")]),
        );
        let audit = routes.audit();
        assert_eq!(audit.split(' ').count(), SWITCH_COUNT);
        assert!(audit.contains("HF2Q_REPACKED_GGUF_M32=1(profile)"));
        assert!(audit.contains("HF2Q_QWEN_VERIFY_ATTN_TILE=0(env)"));
        assert!(audit.contains("HF2Q_QWEN_MOE_BATCHED=unset(default)"));
    }

    #[test]
    fn a_binding_is_per_thread_and_restored_on_drop() {
        let bound = Arc::new(ResolvedRoutes::resolve(
            Some(RouteProfile::Qwen36),
            &|_| None,
            &env(&[]),
        ));
        let inner = Arc::new(ResolvedRoutes::resolve(None, &|_| None, &env(&[])));
        let unbound = || BOUND.with(|bound| bound.borrow().is_none());
        {
            let _outer = bind(Arc::clone(&bound));
            assert!(is_one(MOE_BATCHED));
            assert!(with_value(MOE_BATCHED, |value| value == Some("1")));
            assert_eq!(current().profile(), Some(RouteProfile::Qwen36));
            {
                let _inner = bind(inner);
                assert!(!is_one(MOE_BATCHED));
                assert!(with_value(MOE_BATCHED, |value| value.is_none()));
            }
            assert!(is_one(MOE_BATCHED), "restored");
            // Another thread never sees this thread's binding.
            assert!(std::thread::spawn(unbound).join().expect("thread"));
        }
        assert!(unbound());
        assert_eq!(BOUND_ONES.with(Cell::get), None);
    }

    #[test]
    fn every_profile_belongs_to_a_pinned_qwen_descriptor() {
        for profile in [RouteProfile::Qwen38, RouteProfile::Qwen36] {
            let descriptor = profile.descriptor();
            assert_eq!(RouteProfile::of(descriptor), Some(profile));
            assert!(descriptor.target_source_sha256.is_some());
        }
        assert_eq!(
            RouteProfile::of(&super::super::schema::GEMMA4_E2B_DFLASH2),
            None
        );
    }
}
