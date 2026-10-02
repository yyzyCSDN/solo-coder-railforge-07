import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from railforge.api import RailForgeService
from railforge.crew.qualification import QualificationRule
from railforge.possession.joint_approval import (
    ApprovalEvent,
    ApprovalRequest,
    GroupBooking,
    JointApprovalDenied,
    replay,
)
from railforge.possession.windows import Possession
from railforge.signaling.block_occupancy import Occupancy

U = timezone.utc
AT = datetime(2026, 2, 28, 12, tzinfo=U)
CLEARANCE = timedelta(minutes=5)
# Protected window of the standard request: 2026-03-01 21:30 -> 2026-03-02 02:30 UTC (crosses midnight).
WS = datetime(2026, 3, 1, 21, 30, tzinfo=U)
WE = datetime(2026, 3, 2, 2, 30, tzinfo=U)
OPEN_RULE = QualificationRule("Q1", "welding", datetime(2026, 3, 1, tzinfo=U), None, 1, "certified")


def request(version=0, group="G1"):
    return ApprovalRequest(
        request_id="REQ-1",
        segment="S1",
        start=datetime(2026, 3, 1, 22, tzinfo=U),
        end=datetime(2026, 3, 2, 2, tzinfo=U),
        setup=timedelta(minutes=30),
        release=timedelta(minutes=30),
        blocks=("B1",),
        required_qualifications=("welding",),
        mutex_group=group,
        expected_version=version,
    )


def base():
    return {
        "possessions": [Possession("P2", "S1", datetime(2026, 3, 2, 5, tzinfo=U),
                                   datetime(2026, 3, 2, 6, tzinfo=U), timedelta(0), timedelta(0))],
        "occupancies": [Occupancy("T1", "B1", datetime(2026, 3, 1, 18, tzinfo=U),
                                  datetime(2026, 3, 1, 19, tzinfo=U), 1)],
        "rules": [OPEN_RULE],
        "bookings": [GroupBooking("REQ-9", "G1", datetime(2026, 3, 2, 4, tzinfo=U),
                                  datetime(2026, 3, 2, 6, tzinfo=U))],
    }


def approve(service, operation_id, req, inputs, events=()):
    return service.features.approve_line_blockade(
        operation_id, req, list(events), inputs["possessions"], inputs["occupancies"],
        CLEARANCE, inputs["rules"], inputs["bookings"], AT)


class JointApprovalTests(unittest.TestCase):
    def denied(self, req, inputs, events=()):
        service = RailForgeService()
        with self.assertRaises(JointApprovalDenied) as ctx:
            approve(service, "op-1", req, inputs, events)
        # Any party's veto must block the approved event: nothing committed, nothing published.
        self.assertEqual(service.control.stream.head, 0)
        self.assertIsNone(service.control.store.read("REQ-1"))
        return {r.party for r in ctx.exception.rejections}

    def test_joint_approval_commits_atomically(self):
        service = RailForgeService()
        receipt = approve(service, "op-1", request(), base())
        self.assertEqual(receipt.event.topic, "feature.line_blockade.approved")
        self.assertEqual(receipt.event.payload["status"], "committed")
        decision = receipt.result["decision"]
        self.assertTrue(decision.approved)
        self.assertEqual(decision.rejections, ())
        state = receipt.result["state"]
        self.assertEqual(state.status, "approved")
        self.assertEqual(state.version, 2)
        self.assertEqual(state.window, (WS, WE))
        self.assertEqual(replay("REQ-1", receipt.result["events"]), state)
        self.assertEqual(replay("REQ-1", list(reversed(receipt.result["events"]))), state)
        self.assertTrue(service.control.stream.verify_chain())
        self.assertTrue(service.control.trail.verify())

    def test_protection_window_vetoes(self):
        inputs = base()
        inputs["possessions"].append(Possession("P1", "S1", datetime(2026, 3, 1, 23, tzinfo=U),
                                                datetime(2026, 3, 2, 1, tzinfo=U), timedelta(0), timedelta(0)))
        self.assertEqual(self.denied(request(), inputs), {"protection_window"})

    def test_block_occupancy_vetoes(self):
        inputs = base()
        inputs["occupancies"].append(Occupancy("T9", "B1", datetime(2026, 3, 1, 21, tzinfo=U),
                                               datetime(2026, 3, 1, 22, tzinfo=U), 1))
        self.assertEqual(self.denied(request(), inputs), {"block_occupancy"})

    def test_qualification_vetoes_when_rule_expires_at_midnight(self):
        inputs = base()
        inputs["rules"] = [QualificationRule("Q1", "welding", datetime(2026, 3, 1, tzinfo=U),
                                             datetime(2026, 3, 2, tzinfo=U), 1, "certified")]
        self.assertEqual(self.denied(request(), inputs), {"crew_qualification"})

    def test_mutex_group_vetoes(self):
        inputs = base()
        inputs["bookings"].append(GroupBooking("REQ-0", "G1", datetime(2026, 3, 2, 0, tzinfo=U),
                                               datetime(2026, 3, 2, 3, tzinfo=U)))
        self.assertEqual(self.denied(request(), inputs), {"mutex_group"})

    def test_approval_version_vetoes_stale_snapshot(self):
        self.assertEqual(self.denied(request(version=1), base()), {"approval_version"})

    def test_all_parties_report_together(self):
        inputs = base()
        inputs["possessions"].append(Possession("P1", "S1", datetime(2026, 3, 1, 23, tzinfo=U),
                                                datetime(2026, 3, 2, 1, tzinfo=U), timedelta(0), timedelta(0)))
        inputs["occupancies"].append(Occupancy("T9", "B1", datetime(2026, 3, 1, 21, tzinfo=U),
                                               datetime(2026, 3, 1, 22, tzinfo=U), 1))
        parties = self.denied(request(version=3), inputs)
        self.assertEqual(parties, {"protection_window", "block_occupancy", "approval_version"})

    def test_midnight_handoff_qualification_approves(self):
        inputs = base()
        inputs["rules"] = [
            QualificationRule("Q1", "welding", datetime(2026, 3, 1, tzinfo=U),
                              datetime(2026, 3, 2, tzinfo=U), 1, "certified"),
            QualificationRule("Q2", "welding", datetime(2026, 3, 2, tzinfo=U), None, 1, "certified"),
        ]
        receipt = approve(RailForgeService(), "op-1", request(), inputs)
        self.assertTrue(receipt.result["decision"].approved)

    def test_revocation_replays(self):
        service = RailForgeService()
        receipt = approve(service, "op-1", request(), base())
        log = receipt.result["events"]
        revoked = service.features.revoke_line_blockade("op-2", "REQ-1", log, "storm",
                                                        datetime(2026, 3, 1, 20, tzinfo=U))
        state = revoked.result["state"]
        self.assertEqual(state.status, "revoked")
        self.assertEqual(state.version, 3)
        self.assertEqual(state.window, (WS, WE))
        self.assertEqual(replay("REQ-1", revoked.result["events"]), state)
        self.assertEqual(replay("REQ-1", list(reversed(revoked.result["events"]))), state)
        # Append-only stream: the approved event stays, the revocation folds on top.
        topics = [e.topic for e in service.control.stream.snapshot()]
        self.assertIn("feature.line_blockade.approved", topics)
        self.assertIn("feature.line_blockade.revoked", topics)
        with self.assertRaises(ValueError):
            service.features.revoke_line_blockade("op-3", "REQ-1", revoked.result["events"], "again", AT)
        with self.assertRaises(ValueError):
            service.features.revoke_line_blockade("op-4", "REQ-1", [], "never approved", AT)

    def test_submitted_event_replays_across_timezones(self):
        event = ApprovalEvent("REQ-1", 1, "submitted", AT,
                              (("start", "2026-03-02T05:30:00+08:00"),
                               ("end", "2026-03-02T10:30:00+08:00")))
        state = replay("REQ-1", [event])
        self.assertEqual(state.window, (WS, WE))

    def test_stale_log_and_window_guard(self):
        submitted = ApprovalEvent("REQ-1", 1, "submitted", AT,
                                  (("start", WS.isoformat()), ("end", WE.isoformat())))
        self.assertEqual(self.denied(request(version=0), base(), events=[submitted]), {"approval_version"})
        receipt = approve(RailForgeService(), "op-1", request(version=1), base(), events=[submitted])
        self.assertEqual(receipt.result["state"].version, 2)
        changed = replace(request(version=1), start=datetime(2026, 3, 1, 21, tzinfo=U))
        with self.assertRaises(ValueError):
            approve(RailForgeService(), "op-1", changed, base(), events=[submitted])

    def test_retry_after_denial_is_clean(self):
        service = RailForgeService()
        inputs = base()
        inputs["occupancies"].append(Occupancy("T9", "B1", datetime(2026, 3, 1, 21, tzinfo=U),
                                               datetime(2026, 3, 1, 22, tzinfo=U), 1))
        with self.assertRaises(JointApprovalDenied):
            approve(service, "op-1", request(), inputs)
        inputs["occupancies"].pop()
        receipt = approve(service, "op-1", request(), inputs)
        self.assertTrue(receipt.result["decision"].approved)
        self.assertIs(approve(service, "op-1", request(), inputs), receipt)


if __name__ == "__main__":
    unittest.main()
