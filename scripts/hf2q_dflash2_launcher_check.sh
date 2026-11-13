#!/usr/bin/env bash
# Launcher-side validation of the DFlash2 variables (ADR-067 D6), sourced by
# the Qwen launchers. It rejects only what no policy state accepts, before a
# large model is loaded:
#
# - a value of HF2Q_DFLASH2 that hf2q refuses;
# - under HF2Q_DFLASH2=1 (DFlash2 required), an explicit drafter path that is
#   not a file, or one path variable without the other.
#
# Under the default policy (unset) a stale or incomplete path pair must not
# prevent startup: hf2q reports it and resolves the pinned drafter. Under
# HF2Q_DFLASH2=0 the path variables are not read at all.

hf2q_dflash2_launcher_check() {
    local policy path_var
    policy="$(printf '%s' "${HF2Q_DFLASH2:-}" | tr '[:upper:]' '[:lower:]')"
    case "$policy" in
        ''|0|off|false) return 0 ;;
        1|on|true) ;;
        *)
            echo "HF2Q_DFLASH2 must be 0/off/false or 1/on/true when set (got: $HF2Q_DFLASH2)" >&2
            return 3
            ;;
    esac
    if [[ -n "${HF2Q_DFLASH2_CONFIG:-}" && -z "${HF2Q_DFLASH2_WEIGHTS:-}" ]] ||
        [[ -z "${HF2Q_DFLASH2_CONFIG:-}" && -n "${HF2Q_DFLASH2_WEIGHTS:-}" ]]; then
        echo "HF2Q_DFLASH2=1: HF2Q_DFLASH2_CONFIG and HF2Q_DFLASH2_WEIGHTS are used only together" >&2
        return 3
    fi
    for path_var in HF2Q_DFLASH2_CONFIG HF2Q_DFLASH2_WEIGHTS; do
        if [[ -n "${!path_var:-}" && ! -f "${!path_var}" ]]; then
            echo "HF2Q_DFLASH2=1 and $path_var is not a file: ${!path_var}" >&2
            return 3
        fi
    done
    return 0
}
