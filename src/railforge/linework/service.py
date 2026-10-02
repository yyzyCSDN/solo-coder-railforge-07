"""Application service for joint construction-blockade approvals.

The service owns an append-only per-order ledger. State is the fold of that
ledger (:func:`railforge.linework.joint_approval.replay`), so a cross-midnight
blockade and every revocation can be reconstructed deterministically.

Publication rule: the ``linework.blockade.approved`` event is only published
through the atomic control plane when all five parties approve. Any denial
fails closed - the ballot is recorded as ``rejected`` in the ledger and no
approved event ever reaches the event stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from railforge.core.time import utc
from railforge.core.versioning import Cell
from railforge.linework.checks import (
    AdjacentTask,
    MaintenanceWorker,
    crew_party,
    mutex_party,
    occupancy_party,
    protection_party,
    version_party,
)
from railforge.linework.joint_approval import (
    PARTIES,
    ApprovalEvent,
    BlockadeOrder,
    JointApprovalOrder,
    JointApprovalRejected,
    OrderRevoked,
    PartyVote,
    StaleApprovalVersion,
    replay,
)
from railforge.workflows.control import RailForgeWorkflowControl


@dataclass(frozen=True)
class BallotResult:
    order_id: str
    status: str
    votes: tuple[PartyVote, ...]
    reasons: tuple[str, ...]
    cross_midnight: bool
    version: int
    receipt: object | None = None


def _order_digest(order: BlockadeOrder) -> dict:
    s, e = order.protected_window
    return {
        "order_id": order.order_id,
        "segment": order.segment,
        "block": order.block,
        "group": order.group,
        "start": order.start.isoformat(),
        "end": order.end.isoformat(),
        "protected_start": s.isoformat(),
        "protected_end": e.isoformat(),
        "cross_midnight": order.crosses_midnight,
        "docket_version": order.docket_version,
    }


class BlockadeJointApprovalService:
    def __init__(self, control: RailForgeWorkflowControl | None = None):
        self.control = control or RailForgeWorkflowControl()
        self._orders: dict[str, BlockadeOrder] = {}
        self._ledger: dict[str, list[ApprovalEvent]] = {}
        # Current approval-docket version per order (optimistic registry).
        self._docket_versions: Cell[dict[str, int]] = Cell({})

    # ------------------------------------------------------------------ #
    # Ledger access / replay
    # ------------------------------------------------------------------ #

    def ledger(self, order_id: str) -> tuple[ApprovalEvent, ...]:
        return tuple(self._ledger.get(order_id, ()))

    def state(self, order_id: str) -> JointApprovalOrder:
        if order_id not in self._ledger:
            raise KeyError(order_id)
        return replay(self._ledger[order_id])

    def replay(self, order_id: str) -> JointApprovalOrder:
        """Rebuild aggregate state from the ledger (cross-midnight safe)."""
        return self.state(order_id)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def open(self, order: BlockadeOrder, at: datetime) -> ApprovalEvent:
        """Register a blockade request and append the ``requested`` event."""
        if order.order_id in self._ledger:
            raise ValueError("order already open: " + order.order_id)
        at = utc(at)
        event = ApprovalEvent(
            sequence=1,
            order_id=order.order_id,
            kind="requested",
            at=at,
            cross_midnight=order.crosses_midnight,
        )
        JointApprovalOrder(order.order_id).apply(event)
        self._orders[order.order_id] = order
        self._ledger[order.order_id] = [event]
        snap = self._docket_versions.snapshot()
        versions = dict(snap.value)
        versions.setdefault(order.order_id, order.docket_version)
        self._docket_versions.compare_and_set(snap.version, versions)
        return event

    def submit(
        self,
        order_id: str,
        occupancies: list,
        possessions: list,
        workers: list[MaintenanceWorker],
        qualification_rules: list,
        adjacent: list[AdjacentTask],
        clearance: timedelta,
        check_at: datetime,
        expected_version: int,
        at: datetime,
        operation_id: str | None = None,
    ) -> BallotResult:
        """Run all five parties and publish ``approved`` only on a clean sweep."""
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(order_id)
        state = self.state(order_id)
        if state.status == "revoked":
            raise OrderRevoked(order_id)
        if state.status != "pending":
            raise ValueError(f"order {order_id} is {state.status}, not pending")

        # Optimistic guard: nobody may have bumped the docket version.
        snap = self._docket_versions.snapshot()
        if snap.value.get(order_id) != expected_version:
            raise StaleApprovalVersion(
                f"expected={expected_version} current={snap.value.get(order_id)}"
            )

        # Every party is evaluated independently - never short circuit, so a
        # single denial cannot hide further defects.
        party_results: dict[str, tuple[bool, str]] = {
            "protection": protection_party(order, possessions),
            "occupancy": occupancy_party(order, occupancies, clearance),
            "crew": crew_party(order, workers, qualification_rules, utc(check_at)),
            "mutex": mutex_party(order, adjacent),
            "version": version_party(order, expected_version),
        }
        votes = tuple(PartyVote(p, party_results[p][0], party_results[p][1])
                      for p in PARTIES)
        denied = tuple(v.reason for v in votes if not v.approved)
        approved = not denied
        kind = "approved" if approved else "rejected"
        at = utc(at)

        event = ApprovalEvent(
            sequence=state.version + 1,
            order_id=order_id,
            kind=kind,
            at=at,
            votes=votes,
            reasons=denied,
            cross_midnight=order.crosses_midnight,
        )

        receipt = None
        if approved:
            def action():
                return {"approved": True, "order": _order_digest(order),
                        "parties": list(PARTIES)}
            evidence = {"cross_midnight": order.crosses_midnight,
                        "docket_version": order.docket_version}
            receipt = self.control.run(
                operation_id or f"blockade:{order_id}:approve",
                "linework.blockade.approved", order_id, at, action, evidence,
            )

        # Append the ballot to the ledger regardless of outcome: a rejection
        # is recorded, but deliberately never published to the event stream.
        self._append(state, event)
        if not approved:
            raise JointApprovalRejected(denied)
        return BallotResult(order_id, "approved", votes, denied,
                            order.crosses_midnight, event.sequence, receipt)

    def revoke(self, order_id: str, at: datetime, reason: str = "",
               operation_id: str | None = None) -> BallotResult:
        """Revoke a pending/approved blockade; revocation itself is published."""
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(order_id)
        state = self.state(order_id)
        if state.status == "revoked":
            raise OrderRevoked(order_id)
        if state.status not in ("pending", "approved"):
            raise ValueError(f"order {order_id} is {state.status}; cannot revoke")
        at = utc(at)
        event = ApprovalEvent(
            sequence=state.version + 1,
            order_id=order_id,
            kind="revoked",
            at=at,
            reasons=(reason,) if reason else (),
            cross_midnight=order.crosses_midnight,
        )

        def action():
            return {"revoked": True, "reason": reason, "order": _order_digest(order)}

        receipt = self.control.run(
            operation_id or f"blockade:{order_id}:revoke",
            "linework.blockade.revoked", order_id, at, action,
            {"replaces_status": state.status, "cross_midnight": order.crosses_midnight},
        )
        self._append(state, event)
        return BallotResult(order_id, "revoked", tuple(state.votes.values()),
                            event.reasons, order.crosses_midnight, event.sequence, receipt)

    def bump_docket_version(self, order_id: str, new_version: int) -> int:
        """Administrative docket revision used to invalidate stale approvals."""
        snap = self._docket_versions.snapshot()
        versions = dict(snap.value)
        if order_id not in versions:
            raise KeyError(order_id)
        versions[order_id] = new_version
        self._docket_versions.compare_and_set(snap.version, versions)
        return new_version

    # ------------------------------------------------------------------ #

    def _append(self, state: JointApprovalOrder, event: ApprovalEvent):
        state.apply(event)
        self._ledger[event.order_id].append(event)
