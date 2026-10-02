"""The five independent clearance checks for a construction-blockade ballot.

Each checker is a pure function returning ``(ok, reason)`` so the joint
approval service can collect every party's decision without short-circuiting:
a later checker must still be evaluated even when an earlier one denies.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from railforge.crew.qualification import QualificationRule, active as active_rule
from railforge.linework.joint_approval import BlockadeOrder
from railforge.possession.windows import Possession, conflicts as possession_conflicts
from railforge.signaling.block_occupancy import Occupancy, conflicts as occupancy_conflicts


@dataclass(frozen=True)
class AdjacentTask:
    """Another maintenance/construction task that may be mutually exclusive."""

    task_id: str
    group: str
    start: datetime
    end: datetime
    adjacent_segment: str


@dataclass(frozen=True)
class MaintenanceWorker:
    worker_id: str
    qualification: str
    valid_from: datetime
    valid_to: datetime | None


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def protection_party(order: BlockadeOrder, existing: list[Possession]) -> tuple[bool, str]:
    """封锁保护窗：the protected window (setup + release margins) must be clear."""
    proxy = Possession(
        id=order.order_id,
        segment=order.segment,
        start=order.start,
        end=order.end,
        setup=order.setup,
        release=order.release,
    )
    clashes = possession_conflicts(proxy, existing)
    if clashes:
        return False, "protection_window_conflict:" + ",".join(clashes)
    return True, ""


def occupancy_party(order: BlockadeOrder, occupancies: list[Occupancy],
                    clearance: timedelta) -> tuple[bool, str]:
    """闭塞占用：no train may occupy the block during the blockade window."""
    candidate = Occupancy(
        train="blockade:" + order.order_id,
        block=order.block,
        enter=order.start,
        exit=order.end,
        direction=0,
    )
    clashes = occupancy_conflicts(candidate, occupancies, clearance)
    if clashes:
        return False, "block_occupancy_conflict:" + ",".join(clashes)
    return True, ""


def crew_party(order: BlockadeOrder, workers: list[MaintenanceWorker],
               rules: list[QualificationRule], at: datetime) -> tuple[bool, str]:
    """维护人员资格：every worker needs an active, valid qualification at ``at``."""
    unqualified: list[str] = []
    for w in workers:
        rule = active_rule(rules, w.qualification, at)
        if rule is None:
            unqualified.append(w.worker_id)
            continue
        vf = w.valid_from.astimezone(order.start.tzinfo)
        vt = w.valid_to.astimezone(order.start.tzinfo) if w.valid_to else None
        if not (vf <= at and (vt is None or at < vt)):
            unqualified.append(w.worker_id)
    if unqualified:
        return False, "crew_qualification:" + ",".join(sorted(unqualified))
    return True, ""


def mutex_party(order: BlockadeOrder, adjacent: list[AdjacentTask]) -> tuple[bool, str]:
    """相邻任务互斥组：same-group tasks on the adjacent segment cannot overlap."""
    hits = []
    for t in adjacent:
        if t.task_id == order.order_id or t.group != order.group:
            continue
        if t.adjacent_segment != order.segment:
            continue
        if _overlaps(order.start, order.end,
                     t.start.astimezone(order.start.tzinfo),
                     t.end.astimezone(order.start.tzinfo)):
            hits.append(t.task_id)
    if hits:
        return False, "adjacent_mutex_conflict:" + ",".join(sorted(hits))
    return True, ""


def version_party(order: BlockadeOrder, current_version: int) -> tuple[bool, str]:
    """审批版本：the docket presented must match the version approved for ballot."""
    if order.docket_version != current_version:
        return False, f"stale_version:expected={current_version},actual={order.docket_version}"
    return True, ""
