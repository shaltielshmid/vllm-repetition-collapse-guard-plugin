from __future__ import annotations

import logging
import os
from typing import List, Optional, Any

log = logging.getLogger(__name__)

import os

def validate_buffer_size(value):
    ivalue = int(value)
    if not (ivalue > 0 and (ivalue & (ivalue - 1) == 0)):
        raise ValueError("BUFFER_SIZE must be a power of 2 and greater than 0.")
    return ivalue

class RepetitionGuard:
    """
    Detects when generation gets stuck in a short repeating loop.
    This is intentionally hyper‑simple and allocation‑free in the hot path.
    """

    BUFFER_SIZE = validate_buffer_size(os.getenv("BUFFER_SIZE", "1024")) # must be a power of 2
    MASK = BUFFER_SIZE - 1
    MAX_TOKEN_REP = int(os.getenv("MAX_TOKEN_REP", "32"))
    MIN_GRAM_REP = int(os.getenv("MIN_GRAM_REP", "5"))
    MAX_NGRAM_LEN = int(os.getenv("MAX_NGRAM_LEN", "12"))
    MIN_NGRAM_LEN = int(os.getenv("MIN_NGRAM_LEN", "3"))

    __slots__ = (
        "_history",
        "_index",
        "_max_history_index",
        "_consecutive_token_run_count",
    )

    def __init__(self) -> None:
        self._history = [0] * self.BUFFER_SIZE
        self._index = 0
        self._max_history_index = 0
        self._consecutive_token_run_count = 1

    def new_token(self, token_id: Optional[int]) -> bool:
        """
        Feed a new token into the guard.

        Returns True if a repetition pattern (single‑token or N‑gram)
        crosses the configured thresholds and we should stop.
        """
        if token_id is None:
            return False

        # retrieve the values for faster retrieval(?)
        hist = self._history
        mask = self.MASK

        idx = self._index
        hist[idx] = token_id
        idx = (idx + 1) & mask # cheap trick, since mask is 2^N-1, it's like running modulo but faster
        self._index = idx

        if self._max_history_index < self.BUFFER_SIZE:
            self._max_history_index += 1

        # Single‑token run: previous token equals current token
        if self._max_history_index > 1 and hist[(idx - 2) & mask] == token_id:
            self._consecutive_token_run_count += 1
            if self._consecutive_token_run_count >= self.MAX_TOKEN_REP:
                return True
        else:
            self._consecutive_token_run_count = 1

        # Not enough history yet for N‑gram check
        if self._max_history_index < self.MAX_TOKEN_REP:
            return False

        max_rep = self.MAX_TOKEN_REP
        min_rep = self.MIN_GRAM_REP

        # N‑gram repetition detection
        for p in range(self.MIN_NGRAM_LEN, self.MAX_NGRAM_LEN + 1):
            ok = True
            reps = max(max_rep // p, min_rep)
            for k in range(1, p + 1):
                x = hist[(idx - k) & mask]
                for j in range(1, reps):
                    if x != hist[(idx - k - j * p) & mask]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True

        return False


# -----------------------------------------------------------------------------
# vLLM wiring – hook V1 OutputProcessor.process_outputs
# -----------------------------------------------------------------------------

_FINISH_REASON_STOP: Any | None = None
_REPGUARD_ENABLED = os.environ.get("VLLM_REPGUARD_ENABLE", "1") not in ("0", "false", "False")


def _resolve_finish_reason_stop() -> Any | None:
    """Best‑effort lookup for FinishReason.ABORT from vLLM V1 or V0."""
    global _FINISH_REASON_STOP
    if _FINISH_REASON_STOP is not None:
        return _FINISH_REASON_STOP

    # Try V1 first (vllm.v1.engine)
    try:
        import vllm.v1.engine as v1_engine  # type: ignore[attr-defined]
        FR = getattr(v1_engine, "FinishReason", None)
        if FR is not None and hasattr(FR, "ABORT"):
            _FINISH_REASON_STOP = FR.ABORT
            return _FINISH_REASON_STOP
    except Exception:
        pass

    # Fallback: older V0 layout (best effort)
    try:
        import vllm.engine as engine_mod  # type: ignore[attr-defined]
        FR = getattr(engine_mod, "FinishReason", None)
        if FR is not None and hasattr(FR, "STOP"):
            _FINISH_REASON_STOP = FR.ABORT
            return _FINISH_REASON_STOP
    except Exception:
        pass

    log.warning(
        "vllm_repguard_plugin: could not resolve FinishReason.ABORT; "
        "will still abort requests, but finish_reason may be generic."
    )
    _FINISH_REASON_STOP = None
    return None


def _install_v1_output_processor_hook() -> None:
    """
    Install a wrapper around vLLM V1 OutputProcessor.process_outputs.

    This runs the RepetitionGuard on newly generated token IDs for each
    request and, when triggered, marks the request as STOP and asks
    EngineCore to abort it (via reqs_to_abort), exactly like stop strings.
    """
    try:
        from vllm.v1.engine.output_processor import OutputProcessor  # type: ignore
    except Exception:
        log.info(
            "vllm_repguard_plugin: V1 OutputProcessor not found; "
            "skipping V1 hook."
        )
        return

    if getattr(OutputProcessor, "_repguard_wrapped", False):
        # Already installed (e.g. in a forked process).
        return

    stop_enum = _resolve_finish_reason_stop()
    original = OutputProcessor.process_outputs

    def process_outputs_with_repguard(
        self,
        engine_core_outputs,
        engine_core_timestamp=None,
        iteration_stats=None,
    ):
        # First pass: update repetition guards and mark any cores that need to stop.
        repguard_abort_ids: List[str] = []
        request_states = getattr(self, "request_states", {})

        for core_out in engine_core_outputs:
            req_id = getattr(core_out, "request_id", None)
            if req_id is None:
                continue

            req_state = request_states.get(req_id)
            if req_state is None:
                # Could be an already‑aborted request.
                continue

            # Only apply to text generation; skip pooling‑only outputs.
            if getattr(core_out, "pooling_output", None) is not None:
                continue

            guard: RepetitionGuard = getattr(req_state, "_rep_guard", None)
            if guard is None:
                guard = RepetitionGuard()
                setattr(req_state, "_rep_guard", guard)

            triggered = False
            for tok in getattr(core_out, "new_token_ids", []):
                if guard.new_token(tok):
                    triggered = True
                    break

            if triggered:
                if stop_enum is not None:
                    core_out.finish_reason = stop_enum
                # Surface a human‑readable stop_reason
                core_out.stop_reason = "repetition_guard"
                repguard_abort_ids.append(req_id)

        # Let the original OutputProcessor do its job:
        # - detokenization
        # - stop‑string checking
        # - RequestOutput construction
        # - stats updates
        out = original(self, engine_core_outputs, engine_core_timestamp, iteration_stats)

        # Add our aborts to the scheduler's abort list.
        if repguard_abort_ids:
            abort_list: List[str] = getattr(out, "reqs_to_abort", [])
            existing = set(abort_list)
            for rid in repguard_abort_ids:
                if rid not in existing:
                    abort_list.append(rid)
                    existing.add(rid)

        return out

    # Install wrapper
    OutputProcessor._original_process_outputs = original  # type: ignore[attr-defined]
    OutputProcessor.process_outputs = process_outputs_with_repguard  # type: ignore[assignment]
    OutputProcessor._repguard_wrapped = True  # type: ignore[attr-defined]

    log.info("vllm_repguard_plugin: hooked V1 OutputProcessor.process_outputs")


def register_repguard() -> None:
    """
    Entry point for vLLM's `vllm.general_plugins` group.

    Register this in your package metadata as:

        [options.entry_points]
        vllm.general_plugins =
            repguard = vllm_repguard_plugin:register_repguard

    or in setup.py (see below).
    """
    if not _REPGUARD_ENABLED:
        log.info(
            "vllm_repguard_plugin: disabled (VLLM_REPGUARD_ENABLE=0/false)."
        )
        return

    log.info("=" * 60)
    log.info("vllm_repguard_plugin: initializing repetition guard plugin")
    log.info("=" * 60)

    _install_v1_output_processor_hook()

    log.info("vllm_repguard_plugin: initialization complete")
    log.info("=" * 60)


