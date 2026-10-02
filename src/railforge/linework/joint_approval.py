"""Joint approval workflow for new-line construction blockades.

Five parties must clear a blockade request before an ``approved`` event may be
published:

* ``protection`` - the blockade protection window must be conflict free;
* ``occupancy``  - no conflicting signalling block occupancy;
* ``crew``       - every maintenance worker must hold a valid qualification;
* ``mutex``      - no mutually exclusive adjacent task overlaps the window;
* ``version``    - the request must carry the current approval-docket version.

The decision ledger is a pure, append-only event-sourced aggregate. Both
cross-midnight windows and revocation are first-class events, so the whole
lifecycle is deterministic and can be replayed from the log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# The five approval parties, in deterministic ballot order.
PARTIES = ("protection", "occupancy", "crew", "mutex", "version")


class JointApprovalRejected(RuntimeError):
    """Raised when one or more parties deny the joint ballot."""

    def __init__(self, reasons: tuple[str, ...]):
        self.reasons = tuple(sorted(set(reasons)))
        super().__init__("joint approval rejected: " + ",".join(self.reasons))


class StaleApprovalVersion(RuntimeError):
    """Raised when the submitted docket version is not current."""


class OrderRevoked(RuntimeError):
    """Raised when a revoked order is used after revocation."""


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class BlockadeOrder:
    """A requested line-construction blockade.

    ``start``/``end`` are stored in UTC. A window where ``end`` falls on a
    later UTC date is a cross-midnight blockade and is flagged explicitly so
    replay preserves the fact.
    """

    order_id: str
    segment: str
    start: datetime
    end: datetime
    block: str
    group: str
    docket_version: int
    setup: timedelta = timedelta(0)
    release: timedelta = timedelta(0)

    def __post_init__(self):
        s, e = _utc(self.start), _utc(self.end)
        if e <= s:
            raise ValueError("blockade window must be positive")
        if self.setup.total_seconds() < 0 or self.release.total_seconds() < 0:
            raise ValueError("protection margins must be non-negative")
        # dataclass is frozen; normalise through object.__setattr__.
        object.__setattr__(self, "start", s)
        object.__setattr__(self, "end", e)

    @property
    def protected_window(self) -> tuple[datetime, datetime]:
        return (self.start - self.setup, self.end + self.release)

    @property
    def crosses_midnight(self) -> bool:
        return self.start.date() != self.end.date()


@dataclass(frozen=True)
class PartyVote:
    party: str
    approved: bool
    reason: str = ""


# --------------------------------------------------------------------------- #
# Ledger events
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ApprovalEvent:
    sequence: int
    order_id: str
    kind: str                       # requested | approved | rejected | revoked
    at: datetime
    votes: tuple[PartyVote, ...] = ()
    reasons: tuple[str, ...] = ()
    cross_midnight: bool = False

    def __post_init__(self):
        object.__setattr__(self, "at", _utc(self.at))


# --------------------------------------------------------------------------- #
# Aggregate state (the fold of the ledger)
# --------------------------------------------------------------------------- #

@dataclass
class JointApprovalOrder:
    order_id: str
    version: int = 0
    decided_version: int = 0
    status: str = "new"             # new | approved | rejected | revoked
    parties: tuple[str, ...] = PARTIES
    votes: dict[str, PartyVote] = field(default_factory=dict)
    cross_midnight: bool = False
    last_event: int = 0

    def _ensure_open(self, event: ApprovalEvent):
        if self.status == "revoked":
            raise OrderRevoked(self.order_id)
        if event.order_id != self.order_id:
            raise ValueError("event belongs to another order")
        if event.sequence != self.version + 1:
            raise ValueError("event sequence gap")

    def apply(self, event: ApprovalEvent) -> "JointApprovalOrder":
        self._ensure_open(event)
        self.version = event.sequence
        self.last_event = event.sequence
        if event.kind == "requested":
            if self.status != "new":
                raise ValueError("order already requested")
            self.status = "pending"
            self.cross_midnight = event.cross_midnight
        elif event.kind in ("approved", "rejected"):
            if self.status != "pending":
                raise ValueError("order not awaiting a ballot")
            self._record_ballot(event)
            self.status = event.kind
            self.decided_version = event.sequence
        elif event.kind == "revoked":
            # Revocation is only legal while pending or already approved.
            if self.status not in ("pending", "approved"):
                raise ValueError("order not open to revocation")
            self.status = "revoked"
        else:
            raise ValueError("unknown approval event kind: " + event.kind)
        return self

    def _record_ballot(self, event: ApprovalEvent):
        seen: set[str] = set()
        for vote in event.votes:
            if vote.party not in self.parties:
                raise ValueError("unknown party: " + vote.party)
            if vote.party in seen:
                raise ValueError("duplicate party vote: " + vote.party)
            seen.add(vote.party)
            self.votes[vote.party] = vote
        if seen != set(self.parties):
            raise ValueError("every party must vote")


def replay(events: list[ApprovalEvent]) -> JointApprovalOrder:
    """Fold the ledger into current state; rejects out-of-order history."""
    ordered = sorted(events, key=lambda e: e.sequence)
    if not ordered:
        raise ValueError("empty ledger")
    state = JointApprovalOrder(ordered[0].order_id)
    for event in ordered:
        state.apply(event)
    return state


# --------------------------------------------------------------------------- #
# The joint decision
# --------------------------------------------------------------------------- #

def request_joint_approval(
    order: BlockadeOrder,
    checks: dict[str, bool],
    reasons: dict[str, str] | None = None,
    at: datetime | None = None,
    sequence: int = 2,
) -> ApprovalEvent:
    """Run the five-party joint ballot for ``order``.

    ``checks`` maps each party to its clearance decision. Missing parties are
    treated as denials so an ``approved`` event can never slip through on an
    incomplete ballot. Either all five parties approve (event kind
    ``approved``) or the ballot fails closed with ``rejected`` and the denying
    parties' reasons.
    """
    at = _utc(at or datetime.now(timezone.utc))
    reasons = reasons or {}
    votes: list[PartyVote] = []
    denied: list[str] = []
    for party in PARTIES:
        if party not in checks:
            approved, reason = False, "party_did_not_vote"
        else:
            approved = bool(checks[party])
            reason = "" if approved else reasons.get(party, "denied")
        votes.append(PartyVote(party, approved, reason))
        if not approved:
            denied.append(reason or party)
    kind = "approved" if not denied else "rejected"
    return ApprovalEvent(
        sequence=sequence,
        order_id=order.order_id,
        kind=kind,
        at=at,
        votes=tuple(votes),
        reasons=tuple(denied),
        cross_midnight=order.crosses_midnight,
    )
