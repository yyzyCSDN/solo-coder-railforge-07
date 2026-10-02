"""Joint approval for line construction blockades.

Five parties -- protection window, block occupancy, crew qualification,
mutex group and approval version -- are evaluated together and the
``approved`` event only exists when every party approves.  State is
derived by replaying the event log, so cross-midnight windows (stored as
absolute UTC instants) and revocations fold back deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from railforge.core.intervals import Interval, merge
from railforge.core.time import utc
from railforge.possession.windows import Possession
from railforge.possession.windows import conflicts as window_conflicts
from railforge.possession.windows import protected_window
from railforge.signaling.block_occupancy import Occupancy
from railforge.signaling.block_occupancy import conflicts as occupancy_conflicts

PARTIES = ('protection_window', 'block_occupancy', 'crew_qualification', 'mutex_group', 'approval_version')

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class ApprovalRequest:
    request_id: str
    segment: str
    start: datetime
    end: datetime
    setup: timedelta
    release: timedelta
    blocks: tuple[str, ...] = ()
    required_qualifications: tuple[str, ...] = ()
    mutex_group: str | None = None
    expected_version: int = 0


@dataclass(frozen=True)
class GroupBooking:
    request_id: str
    group: str
    start: datetime
    end: datetime
    status: str = 'approved'


@dataclass(frozen=True)
class Rejection:
    party: str
    reason: str


@dataclass(frozen=True)
class JointDecision:
    request_id: str
    approved: bool
    window: tuple[datetime, datetime]
    rejections: tuple[Rejection, ...]


@dataclass(frozen=True)
class ApprovalEvent:
    request: str
    sequence: int
    kind: str
    at: datetime
    detail: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ApprovalState:
    request: str
    status: str = 'draft'
    version: int = 0
    window: tuple[datetime, datetime] | None = None
    reasons: tuple[tuple[str, str], ...] = ()


class JointApprovalDenied(RuntimeError):
    """Raised when any party rejects; nothing is committed or published."""

    def __init__(self, rejections):
        self.rejections = tuple(rejections)
        super().__init__('joint approval denied: ' + '; '.join(f'{r.party}: {r.reason}' for r in self.rejections))


def _possession(request: ApprovalRequest) -> Possession:
    return Possession(request.request_id, request.segment, request.start, request.end, request.setup, request.release)


def request_window(request: ApprovalRequest):
    """Protected window (setup .. release) as absolute UTC instants."""
    return protected_window(_possession(request))


def apply(state: ApprovalState, event: ApprovalEvent) -> ApprovalState:
    if event.request != state.request or event.sequence != state.version + 1:
        raise ValueError('event sequence')
    if event.kind == 'submitted':
        if state.status != 'draft':
            raise ValueError('illegal transition')
        detail = dict(event.detail)
        ws = utc(datetime.fromisoformat(detail['start']))
        we = utc(datetime.fromisoformat(detail['end']))
        if we <= ws:
            raise ValueError('window')
        return ApprovalState(state.request, 'submitted', event.sequence, (ws, we), ())
    if event.kind == 'approved':
        if state.status != 'submitted':
            raise ValueError('illegal transition')
        return ApprovalState(state.request, 'approved', event.sequence, state.window, ())
    if event.kind == 'rejected':
        if state.status != 'submitted':
            raise ValueError('illegal transition')
        return ApprovalState(state.request, 'rejected', event.sequence, state.window, tuple(event.detail))
    if event.kind == 'revoked':
        if state.status != 'approved':
            raise ValueError('illegal transition')
        return ApprovalState(state.request, 'revoked', event.sequence, state.window, tuple(event.detail))
    raise ValueError('event kind')


def replay(request_id: str, events) -> ApprovalState:
    state = ApprovalState(request_id)
    for e in sorted(events, key=lambda e: e.sequence):
        state = apply(state, e)
    return state


def _window_rejections(request, possessions):
    hits = window_conflicts(_possession(request), possessions)
    return [Rejection('protection_window', 'conflicts: ' + ','.join(hits))] if hits else []


def _occupancy_rejections(request, ws, we, occupancies, clearance):
    hits = set()
    for block in request.blocks:
        probe = Occupancy(request.request_id, block, ws, we, 0)
        hits.update(occupancy_conflicts(probe, occupancies, clearance))
    return [Rejection('block_occupancy', 'trains: ' + ','.join(sorted(hits)))] if hits else []


def _qualification_rejections(request, ws, we, rules):
    out = []
    for scope in request.required_qualifications:
        ivs = []
        for r in rules:
            if r.scope not in (scope, '*'):
                continue
            s = utc(r.effective_from)
            e = utc(r.effective_to) if r.effective_to else _FAR_FUTURE
            if e <= s:
                raise ValueError('rule interval')
            ivs.append(Interval(s, e))
        if not any(iv.start <= ws and iv.end >= we for iv in merge(ivs)):
            out.append(Rejection('crew_qualification', f'scope {scope} not covered'))
    return out


def _mutex_rejections(request, ws, we, bookings):
    if not request.mutex_group:
        return []
    hits = []
    for b in bookings:
        if b.group != request.mutex_group or b.request_id == request.request_id or b.status != 'approved':
            continue
        bs, be = utc(b.start), utc(b.end)
        if be <= bs:
            raise ValueError('booking window')
        if max(ws, bs) < min(we, be):
            hits.append(b.request_id)
    return [Rejection('mutex_group', 'overlap: ' + ','.join(sorted(hits)))] if hits else []


def _version_rejections(request, base_version):
    if request.expected_version != base_version:
        return [Rejection('approval_version', f'expected={request.expected_version} actual={base_version}')]
    return []


def evaluate(request, base_version, possessions, occupancies, clearance, qualification_rules, bookings):
    """Dry-run the joint check; every party is evaluated, none short-circuits."""
    ws, we = request_window(request)
    rejections = (
        _window_rejections(request, possessions)
        + _occupancy_rejections(request, ws, we, occupancies, clearance)
        + _qualification_rejections(request, ws, we, qualification_rules)
        + _mutex_rejections(request, ws, we, bookings)
        + _version_rejections(request, base_version)
    )
    return JointDecision(request.request_id, not rejections, (ws, we), tuple(rejections))


def approve(request, events, possessions, occupancies, clearance, qualification_rules, bookings, at):
    """Replay the log, submit if needed, then append the joint decision.

    Returns (decision, log, state).  The decision event is ``approved`` only
    when every party approves; otherwise a single ``rejected`` event carries
    all parties' reasons.
    """
    state = replay(request.request_id, events)
    base_version = state.version
    ws, we = request_window(request)
    if state.window is not None and state.window != (ws, we):
        raise ValueError('request window differs from submitted window')
    log = list(events)
    if state.status == 'draft':
        event = ApprovalEvent(request.request_id, state.version + 1, 'submitted', at,
                              (('start', ws.isoformat()), ('end', we.isoformat())))
        state = apply(state, event)
        log.append(event)
    if state.status != 'submitted':
        raise ValueError('request not decidable')
    decision = evaluate(request, base_version, possessions, occupancies, clearance, qualification_rules, bookings)
    detail = tuple((r.party, r.reason) for r in decision.rejections)
    event = ApprovalEvent(request.request_id, state.version + 1,
                          'approved' if decision.approved else 'rejected', at, detail)
    log.append(event)
    return decision, tuple(log), apply(state, event)


def revoke(request_id, events, at, reason=''):
    """Append a ``revoked`` event; replaying the resulting log folds to revoked."""
    state = replay(request_id, events)
    if state.status != 'approved':
        raise ValueError('only approved requests can be revoked')
    detail = (('reason', reason),) if reason else ()
    event = ApprovalEvent(request_id, state.version + 1, 'revoked', at, detail)
    return tuple(list(events) + [event]), apply(state, event)
