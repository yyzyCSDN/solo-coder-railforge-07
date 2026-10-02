import unittest
from datetime import datetime, timedelta, timezone

from railforge.api import RailForgeService
from railforge.crew.qualification import QualificationRule
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
    BlockadeOrder,
    JointApprovalRejected,
    JointApprovalOrder,
    OrderRevoked,
    StaleApprovalVersion,
    replay,
    request_joint_approval,
)
from railforge.linework.service import BlockadeJointApprovalService
from railforge.possession.windows import Possession
from railforge.signaling.block_occupancy import Occupancy

U = timezone.utc


def order(order_id="BLK-1", start=datetime(2026, 3, 1, 22, tzinfo=U),
          hours=4, version=3, group="G1", segment="S-UP", block="B9"):
    return BlockadeOrder(
        order_id=order_id, segment=segment, start=start,
        end=start + timedelta(hours=hours), block=block, group=group,
        docket_version=version, setup=timedelta(minutes=15),
        release=timedelta(minutes=15),
    )


def qualified_worker(wid="W1", scope="line-welder", at=datetime(2026, 3, 1, tzinfo=U)):
    return MaintenanceWorker(wid, scope, at - timedelta(days=1), at + timedelta(days=30))


def qualification_rule(scope="line-welder", at=datetime(2026, 3, 1, tzinfo=U)):
    return QualificationRule("R1", scope, at - timedelta(days=365), None, 10, "certified")


def submit_kwargs(service, oid, **over):
    at = datetime(2026, 3, 1, 21, tzinfo=U)
    base = dict(
        order_id=oid, occupancies=[], possessions=[],
        workers=[qualified_worker()], qualification_rules=[qualification_rule()],
        adjacent=[], clearance=timedelta(minutes=10), check_at=at,
        expected_version=3, at=at,
    )
    base.update(over)
    return base


class PartyCheckerTests(unittest.TestCase):
    def test_protection_window_margins_conflict(self):
        o = order()
        # Existing possession ends at 22:10, inside the 21:45 setup margin.
        clash = Possession("P-OLD", "S-UP", datetime(2026, 3, 1, 18, tzinfo=U),
                           datetime(2026, 3, 1, 22, 10, tzinfo=U),
                           timedelta(0), timedelta(0))
        ok, reason = protection_party(o, [clash])
        self.assertFalse(ok)
        self.assertIn("protection_window_conflict", reason)
        self.assertTrue(protection_party(o, [])[0])

    def test_block_occupancy_conflict(self):
        o = order()
        occ = Occupancy("T88", "B9", datetime(2026, 3, 1, 23, tzinfo=U),
                        datetime(2026, 3, 2, 1, tzinfo=U), 1)
        ok, reason = occupancy_party(o, [occ], timedelta(minutes=10))
        self.assertFalse(ok)
        self.assertIn("block_occupancy_conflict", reason)
        self.assertTrue(occupancy_party(o, [], timedelta(minutes=10))[0])

    def test_crew_qualification_required(self):
        o = order()
        at = datetime(2026, 3, 1, 21, tzinfo=U)
        expired = MaintenanceWorker("W2", "line-welder",
                                    at - timedelta(days=90), at - timedelta(days=1))
        ok, reason = crew_party(o, [qualified_worker(), expired],
                                [qualification_rule()], at)
        self.assertFalse(ok)
        self.assertIn("W2", reason)
        self.assertTrue(crew_party(o, [qualified_worker()],
                                   [qualification_rule()], at)[0])

    def test_adjacent_mutex_group(self):
        o = order()
        same_group_overlap = AdjacentTask(
            "T-A", "G1", datetime(2026, 3, 1, 23, tzinfo=U),
            datetime(2026, 3, 2, 2, tzinfo=U), "S-UP")
        ok, reason = mutex_party(o, [same_group_overlap])
        self.assertFalse(ok)
        self.assertIn("adjacent_mutex_conflict", reason)
        # Different group or non-overlapping is fine.
        other_group = AdjacentTask("T-B", "G2", same_group_overlap.start,
                                   same_group_overlap.end, "S-UP")
        self.assertTrue(mutex_party(o, [other_group])[0])

    def test_docket_version_must_match(self):
        self.assertTrue(version_party(order(version=3), 3)[0])
        ok, reason = version_party(order(version=2), 3)
        self.assertFalse(ok)
        self.assertIn("stale_version", reason)


class JointApprovalAggregateTests(unittest.TestCase):
    def test_cross_midnight_flag_is_normalized_and_preserved(self):
        o = order(start=datetime(2026, 3, 1, 22, tzinfo=U), hours=4)
        self.assertTrue(o.crosses_midnight)
        svc = BlockadeJointApprovalService()
        svc.open(o, datetime(2026, 3, 1, 21, tzinfo=U))
        result = svc.submit(**submit_kwargs(svc, "BLK-1"))
        self.assertTrue(result.cross_midnight)
        replayed = svc.replay("BLK-1")
        self.assertTrue(replayed.cross_midnight)

    def test_any_party_rejection_blocks_approved_event(self):
        svc = BlockadeJointApprovalService()
        o = order()
        svc.open(o, datetime(2026, 3, 1, 21, tzinfo=U))
        bad_occ = [Occupancy("T88", "B9", datetime(2026, 3, 1, 23, tzinfo=U),
                             datetime(2026, 3, 2, 1, tzinfo=U), 1)]
        with self.assertRaises(JointApprovalRejected) as ctx:
            svc.submit(**submit_kwargs(svc, "BLK-1", occupancies=bad_occ))
        self.assertTrue(any("block_occupancy_conflict" in r for r in ctx.exception.reasons))
        # The ledger recorded the rejection...
        state = svc.replay("BLK-1")
        self.assertEqual(state.status, "rejected")
        self.assertFalse(state.votes["occupancy"].approved)
        # ...but no approved event reached the stream.
        topics = [e.topic for e in svc.control.stream.snapshot()]
        self.assertNotIn("linework.blockade.approved", topics)
        self.assertEqual(svc.control.stream.head, 0)

    def test_all_five_parties_approve_publishes_atomically(self):
        svc = BlockadeJointApprovalService()
        o = order()
        svc.open(o, datetime(2026, 3, 1, 21, tzinfo=U))
        result = svc.submit(**submit_kwargs(svc, "BLK-1"))
        self.assertEqual(result.status, "approved")
        self.assertEqual(len(result.votes), 5)
        self.assertEqual({v.party for v in result.votes}, set(PARTIES))
        events = svc.control.stream.snapshot()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].topic, "linework.blockade.approved")
        self.assertEqual(events[0].payload["status"], "committed")
        self.assertEqual(events[0].payload["control_path"], "atomic")
        self.assertTrue(events[0].payload["cross_midnight"])
        self.assertTrue(svc.control.stream.verify_chain())
        self.assertTrue(svc.control.trail.verify())

    def test_missing_party_vote_fails_closed(self):
        o = order()
        event = request_joint_approval(o, {p: True for p in PARTIES if p != "mutex"})
        self.assertEqual(event.kind, "rejected")
        self.assertFalse(next(v for v in event.votes if v.party == "mutex").approved)

    def test_stale_docket_version_rejected_before_publication(self):
        svc = BlockadeJointApprovalService()
        o = order(version=3)
        svc.open(o, datetime(2026, 3, 1, 21, tzinfo=U))
        svc.bump_docket_version("BLK-1", 4)
        with self.assertRaises(StaleApprovalVersion):
            svc.submit(**submit_kwargs(svc, "BLK-1", expected_version=3))
        # Still pending; nothing published.
        self.assertEqual(svc.replay("BLK-1").status, "pending")
        self.assertEqual(svc.control.stream.head, 0)
        # Resubmitting a docket carrying the fresh version approves.
        fresh = order(version=4)
        svc._orders["BLK-1"] = fresh
        result = svc.submit(**submit_kwargs(svc, "BLK-1", expected_version=4))
        self.assertEqual(result.status, "approved")

    def test_revocation_is_published_and_blocks_further_submit(self):
        svc = BlockadeJointApprovalService()
        o = order()
        svc.open(o, datetime(2026, 3, 1, 21, tzinfo=U))
        svc.submit(**submit_kwargs(svc, "BLK-1"))
        svc.revoke("BLK-1", datetime(2026, 3, 2, 3, tzinfo=U), reason="weather")
        state = svc.replay("BLK-1")
        self.assertEqual(state.status, "revoked")
        topics = [e.topic for e in svc.control.stream.snapshot()]
        self.assertEqual(topics, ["linework.blockade.approved",
                                  "linework.blockade.revoked"])
        with self.assertRaises(OrderRevoked):
            svc.submit(**submit_kwargs(svc, "BLK-1"))

    def test_ledger_replay_is_deterministic_across_midnight_and_revoke(self):
        svc = BlockadeJointApprovalService()
        o = order()
        svc.open(o, datetime(2026, 3, 1, 21, tzinfo=U))
        svc.submit(**submit_kwargs(svc, "BLK-1"))
        svc.revoke("BLK-1", datetime(2026, 3, 2, 4, tzinfo=U))
        events = svc.ledger("BLK-1")
        self.assertEqual([e.kind for e in events],
                         ["requested", "approved", "revoked"])
        # Re-fold the same events from scratch (e.g. after a restart).
        rebuilt = replay(list(events))
        self.assertEqual(rebuilt.status, "revoked")
        self.assertEqual(rebuilt.version, 3)
        self.assertTrue(rebuilt.cross_midnight)
        self.assertTrue(all(v.approved for v in rebuilt.votes.values()))

    def test_rejected_order_cannot_be_revoked(self):
        svc = BlockadeJointApprovalService()
        o = order()
        svc.open(o, datetime(2026, 3, 1, 21, tzinfo=U))
        with self.assertRaises(JointApprovalRejected):
            svc.submit(**submit_kwargs(
                svc, "BLK-1",
                occupancies=[Occupancy("T88", "B9", datetime(2026, 3, 1, 23, tzinfo=U),
                                       datetime(2026, 3, 2, 1, tzinfo=U), 1)]))
        with self.assertRaises(ValueError):
            svc.revoke("BLK-1", datetime(2026, 3, 2, tzinfo=U))
        self.assertEqual(svc.replay("BLK-1").status, "rejected")

    def test_replay_rejects_gapped_or_foreign_history(self):
        o = order()
        svc = BlockadeJointApprovalService()
        svc.open(o, datetime(2026, 3, 1, 21, tzinfo=U))
        events = list(svc.ledger("BLK-1"))
        from railforge.linework.joint_approval import ApprovalEvent
        bad = ApprovalEvent(sequence=3, order_id="BLK-1", kind="revoked",
                            at=datetime(2026, 3, 2, tzinfo=U))
        with self.assertRaises(ValueError):
            replay(events + [bad])
        foreign = ApprovalEvent(sequence=2, order_id="OTHER", kind="revoked",
                                at=datetime(2026, 3, 2, tzinfo=U))
        with self.assertRaises(ValueError):
            replay(events + [foreign])


class ServiceWiringTests(unittest.TestCase):
    def test_service_exposes_joint_approval_on_shared_control_plane(self):
        service = RailForgeService()
        o = order()
        service.blockade_approvals.open(o, datetime(2026, 3, 1, 21, tzinfo=U))
        result = service.blockade_approvals.submit(
            **submit_kwargs(service.blockade_approvals, "BLK-1"))
        self.assertEqual(result.status, "approved")
        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())
        health = service.health()
        self.assertIn("linework.joint_approval", health["capabilities"])


if __name__ == "__main__":
    unittest.main()
