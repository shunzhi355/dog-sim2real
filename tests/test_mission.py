"""Synthetic perception/ACK replay: no camera, hardware, sleeps or wall clock."""

from dataclasses import asdict
import json

import pytest

from dog3588.mission import MissionConfig, MissionFSM, MissionInput, MissionState


LINE = {"found": True, "error": 0., "heading": 0., "confidence": 1.}


class Replay:
    def __init__(self, **overrides):
        cfg = dict(parcel_colour="yellow", route_map={"yellow": {"red+green": "left"}},
                   return_route_map={"yellow": {"green": "right"}})
        cfg.update(overrides)
        self.fsm = MissionFSM(cfg, mission_id="test-round")
        self.now = 0.
        self.requests = []

    def tick(self, dt=.1, **values):
        self.now = round(self.now + dt, 8)
        sample = dict(now=self.now, vision_stamp=self.now, tof_stamp=self.now,
                      tof_valid=True, line=LINE)
        sample.update(values)
        output = self.fsm.step(MissionInput(**sample))
        if output.delivery_request:
            self.requests.append(output.delivery_request)
        return output

    def frames(self, count=2, **values):
        for _ in range(count):
            output = self.tick(**values)
        return output

    def start(self):
        assert self.tick(start=True).state == MissionState.FOLLOW

    def dock(self):
        self.start()
        assert self.frames(colours=[self.fsm.parcel_colour]).state == MissionState.DOCK
        assert self.frames(events={"docked"}).state == MissionState.DELIVERY
        return self.requests[0].request_id

    def deliver(self):
        request_id = self.dock()
        self.tick(delivery_ack_id=request_id, delivery_ack_status="completed",
                  delivery_ack_stamp=self.now + .1, events={"delivery_confirmed"})
        assert self.tick(events={"delivery_confirmed"}).state == MissionState.RETURN_TURN
        return request_id

    def align_return(self):
        self.frames(events={"return_aligned"})
        assert self.frames(count=2).state == MissionState.RETURN


@pytest.mark.parametrize("colour", ["yellow", "brown"])
def test_complete_round_with_outbound_and_return_obstacles_junctions(colour):
    r = Replay(parcel_colour=colour, route_map={colour: {"red+green": "left"}},
               return_route_map={colour: {"green": "right"}})
    r.start()
    for leg in (MissionState.FOLLOW, MissionState.RETURN):
        assert r.frames(colours=["blue"]).state == MissionState.APPROACH
        assert r.tick().velocity[0] == pytest.approx(.06)
        assert r.frames(events={"obstacle_near"}).state == MissionState.CROSS
        assert r.frames(events={"obstacle_crossed"}).state == MissionState.STABILIZE
        for _ in range(30):
            out = r.tick(stable=True)
            assert out.state == MissionState.STABILIZE and out.velocity == (0., 0., 0.)
        assert r.tick(stable=True).state == leg
        marker = ["red", "green"] if leg == MissionState.FOLLOW else ["green"]
        assert r.frames(colours=marker).state == MissionState.TURN
        assert r.tick(line={"found": False}).velocity[2] * (1 if leg == MissionState.FOLLOW else -1) > 0
        assert r.frames(events={"turn_complete"}).state == MissionState.TURN
        assert r.frames(count=2).state == leg
        if leg == MissionState.RETURN:
            break
        wrong = "brown" if colour == "yellow" else "yellow"
        assert r.frames(colours=[wrong]).state == MissionState.FOLLOW
        assert r.frames(colours=[colour]).state == MissionState.DOCK
        assert r.frames(events={"docked"}).state == MissionState.DELIVERY
        request = r.requests[0]
        assert request.colour == colour
        assert r.tick(delivery_ack_id=request.request_id, delivery_ack_status="accepted",
                      delivery_ack_stamp=r.now + .1).state == MissionState.DELIVERY
        assert r.frames(events={"delivery_confirmed"}).state == MissionState.DELIVERY
        assert r.tick(delivery_ack_id=request.request_id, delivery_ack_status="completed",
                      delivery_ack_stamp=r.now + .1).state == MissionState.RETURN_TURN
        r.align_return()
    assert r.frames(events={"home_visible"}).state == MissionState.HOME
    assert r.frames(events={"home_confirmed"}).state == MissionState.DONE
    assert r.fsm.obstacles_completed == 2 and r.fsm.delivery_completed
    assert len(r.requests) == 1
    assert r.tick(dt=1000, start=True).velocity == (0., 0., 0.)


def test_start_requires_request_and_debounced_parcel_recognition():
    r = Replay(parcel_colour=None)
    assert r.tick(parcel_colour="yellow").state == MissionState.START
    assert r.tick(start=True, parcel_colour="yellow").state == MissionState.START
    # Re-reading a cached frame at policy frequency cannot confirm recognition.
    stamp = r.now
    for _ in range(5):
        assert r.tick(dt=.01, vision_stamp=stamp, parcel_colour="yellow").state == MissionState.START
    assert r.tick(parcel_colour="yellow").state == MissionState.FOLLOW
    assert r.fsm.parcel_colour == "yellow"


def test_household_colour_does_not_identify_parcel_and_start_times_out():
    r = Replay(parcel_colour=None, startup_timeout_s=.5)
    r.tick(start=True, colours=["yellow"])
    assert r.frames(colours=["yellow"]).state == MissionState.START
    assert r.tick(dt=.4).reason == "start_timeout"


def test_line_controller_sign_and_configured_gait():
    gait = (1.5, .575, .02, 0., .5, .5, 0.)
    r = Replay(gait=gait)
    r.start()
    out = r.tick(line=dict(LINE, error=.5))
    assert out.velocity[0] == pytest.approx(.06)
    assert out.velocity[2] < 0 and out.gait == gait


@pytest.mark.parametrize("marker", [["red"], ["green"], ["red", "green"]])
def test_no_default_route_direction(marker):
    r = Replay(route_map={})
    r.start()
    out = r.frames(colours=marker)
    assert out.state == MissionState.FAULT and out.reason.startswith("unmapped_route:")
    assert out.velocity == (0., 0., 0.)


def test_route_signature_must_be_debounced_and_junction_cleared_to_rearm():
    r = Replay(route_map={"yellow": {"green": "right", "red": "left"}}, reacquire_frames=1)
    r.start()
    assert r.tick(colours=["red"]).state == MissionState.FOLLOW
    assert r.tick(colours=["green"]).state == MissionState.FOLLOW
    assert r.tick(colours=["green"]).state == MissionState.TURN
    assert r.tick(colours=["green"]).velocity[2] < 0
    r.frames(colours=["green"], events={"turn_complete"})
    assert r.frames(colours=["red"]).state == MissionState.FOLLOW
    # A single missing frame is insufficient to rearm the junction.
    r.tick()
    assert r.frames(colours=["red"]).state == MissionState.FOLLOW
    r.frames()
    assert r.frames(colours=["red"]).state == MissionState.TURN


def test_blue_does_not_claim_geometric_crossing_on_timeout():
    r = Replay(approach_timeout_s=.5)
    r.start()
    r.frames(colours=["blue"])
    assert r.tick(dt=.5).reason == "approach_timeout"
    assert r.fsm.obstacles_completed == 0
    r = Replay(cross_timeout_s=.5)
    r.start()
    r.frames(colours=["blue"])
    r.frames(events={"obstacle_near"})
    assert r.tick(dt=.5).reason == "cross_timeout"
    assert r.fsm.obstacles_completed == 0


def test_stability_resets_on_unstable_frame_and_sampling_gap():
    r = Replay()
    r.start()
    r.frames(colours=["blue"])
    r.frames(events={"obstacle_near"})
    r.frames(events={"obstacle_crossed"})
    r.frames(count=20, stable=True)
    r.tick(stable=False)
    r.frames(count=20, stable=True)
    assert r.fsm.obstacles_completed == 0
    # A jump with a fresh frame proves nothing about the intervening interval.
    assert r.tick(dt=3.1, stable=True).state == MissionState.STABILIZE
    assert r.frames(count=29, stable=True).state == MissionState.STABILIZE
    assert r.tick(stable=True).state == MissionState.FOLLOW


def test_old_obstacle_event_cannot_complete_next_obstacle():
    r = Replay()
    r.start()
    r.frames(colours=["blue"])
    r.frames(events={"obstacle_near"})
    r.frames(events={"obstacle_crossed"})
    r.frames(count=31, stable=True, events={"obstacle_crossed"})
    r.frames(colours=["blue"], events={"obstacle_crossed"})
    assert r.frames(events={"obstacle_near", "obstacle_crossed"}).state == MissionState.CROSS
    assert r.frames(events={"obstacle_crossed"}).state == MissionState.CROSS


@pytest.mark.parametrize("values,reason", [
    ({"vision_stamp": None}, "stale_vision"),
    ({"tof_valid": False}, "invalid_or_stale_tof"),
    ({"tof_stamp": None}, "invalid_or_stale_tof"),
    ({"line": {"found": False}}, "lost_line"),
    ({"line": dict(LINE, confidence=.1)}, "lost_line"),
    ({"line": dict(LINE, error=float("nan"))}, "lost_line"),
])
def test_sensor_failure_stops_immediately_then_latches_timeout(values, reason):
    r = Replay()
    r.start()
    out = r.tick(**values)
    assert out.mode == "stop" and out.state == MissionState.FOLLOW
    assert out.velocity == (0., 0., 0.) and out.reason == reason
    out = r.tick(dt=2., **values)
    assert out.state == MissionState.FAULT and out.reason == reason + "_timeout"
    assert r.tick(start=True).state == MissionState.FAULT


def test_recovery_requires_distinct_good_frames_and_keeps_deadline():
    r = Replay()
    r.start()
    r.tick(line={"found": False})
    assert r.tick().mode == "stop"
    stamp = r.now
    assert r.tick(dt=.01, vision_stamp=stamp).mode == "stop"
    assert r.tick().mode == "stop"
    assert r.tick().velocity[0] > 0
    r = Replay(path_timeout_s=.5)
    r.start()
    r.tick(line={"found": False})
    assert r.tick(dt=.5).reason == "path_timeout"


def test_turn_needs_event_and_visual_reacquisition_with_bounded_timeout():
    r = Replay(turn_timeout_s=1.)
    r.start()
    r.frames(colours=["red", "green"])
    assert r.tick().state == MissionState.TURN
    assert r.tick(dt=1.).reason == "turn_timeout"
    r = Replay()
    r.start()
    r.frames(colours=["red", "green"])
    r.frames(events={"turn_complete"}, line={"found": False})
    assert r.tick(line={"found": False}).velocity == (0., 0., 0.)
    assert r.frames().state == MissionState.TURN
    assert r.tick().state == MissionState.FOLLOW


@pytest.mark.parametrize("status", ["unsupported", "failed"])
def test_bridge_cannot_deliver_faults_without_claiming_success(status):
    r = Replay()
    request_id = r.dock()
    out = r.tick(delivery_ack_id=request_id, delivery_ack_status=status,
                 delivery_ack_stamp=r.now + .1)
    assert out.reason == "delivery_" + status and out.state == MissionState.FAULT
    assert not r.fsm.delivery_completed and len(r.requests) == 1


def test_delivery_request_is_once_and_ack_must_match_be_fresh_and_completed():
    r = Replay(delivery_timeout_s=2.)
    request_id = r.dock()
    r.frames(events={"delivery_confirmed", "docked"})
    for identity, status, stamp in [("other", "completed", r.now + .1),
                                     (request_id, "completed", 0.),
                                     (request_id, "accepted", r.now + .3)]:
        assert r.tick(delivery_ack_id=identity, delivery_ack_status=status,
                      delivery_ack_stamp=stamp).state == MissionState.DELIVERY
    assert len(r.requests) == 1 and not r.fsm.delivery_completed
    assert r.tick(dt=2.).reason == "delivery_timeout"


def test_completed_ack_alone_is_not_physical_delivery():
    r = Replay(delivery_timeout_s=.5)
    request_id = r.dock()
    assert r.tick(delivery_ack_id=request_id, delivery_ack_status="completed",
                  delivery_ack_stamp=r.now + .1).state == MissionState.DELIVERY
    assert r.tick(dt=.5).reason == "delivery_timeout"
    assert not r.fsm.delivery_completed


def test_delivery_ack_recorded_during_sensor_stop_without_resuming_motion():
    r = Replay()
    request_id = r.dock()
    out = r.tick(vision_stamp=None, delivery_ack_id=request_id,
                 delivery_ack_status="completed", delivery_ack_stamp=r.now + .1)
    assert out.mode == "stop"
    r.frames(count=3)
    assert r.frames(events={"delivery_confirmed"}).state == MissionState.RETURN_TURN


def test_return_never_claims_home_from_elapsed_time_or_early_confirmation():
    r = Replay(path_timeout_s=3.)
    r.deliver()
    r.align_return()
    assert r.frames(events={"home_confirmed"}).state == MissionState.RETURN
    assert r.frames(events={"home_visible", "home_confirmed"}).state == MissionState.HOME
    assert r.frames(events={"home_confirmed"}).state == MissionState.HOME
    assert r.tick(dt=3.).state == MissionState.FAULT


def test_distinct_rounds_have_distinct_delivery_request_ids_and_deterministic_replay():
    records = []
    for index, changes in enumerate((dict(start=True), dict(colours=["yellow"]),
                                    dict(colours=["yellow"]), dict(events=["docked"]),
                                    dict(events=["docked"]))):
        stamp = (index + 1) * .1
        records.append(json.loads(json.dumps(dict(now=stamp, vision_stamp=stamp,
                         tof_stamp=stamp, tof_valid=True, line=LINE, **changes))))
    outputs = []
    for _ in range(2):
        fsm = MissionFSM({"parcel_colour": "yellow"}, mission_id="same-replay")
        outputs.append([asdict(fsm.step(MissionInput(**record))) for record in records])
    assert outputs[0] == outputs[1]
    other = MissionFSM({"parcel_colour": "yellow"}, mission_id="next-round")
    for record in records:
        out = other.step(MissionInput(**record))
    assert out.delivery_request.request_id != outputs[0][-1]["delivery_request"]["request_id"]


@pytest.mark.parametrize("changes", [
    {"stable_duration_s": 2.9}, {"path_timeout_s": 0}, {"turn_rad_s": float("nan")},
    {"debounce_frames": True}, {"reacquire_frames": 0}, {"gait": [1, 2]},
    {"route_map": {"yellow": {"red": "guess"}}}, {"parcel_colour": "red"},
    {"min_line_confidence": float("nan")},
])
def test_config_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        MissionConfig(**changes)


@pytest.mark.parametrize("values,reason", [
    ({"fault": "fallen"}, "external_fault:fallen"),
    ({"events": {"guess_home"}}, "unknown_perception_event_or_colour"),
    ({"now": float("nan")}, "invalid_monotonic_clock"),
    ({"now": 0.}, "invalid_monotonic_clock"),
])
def test_faults_are_latched_and_zero_velocity(values, reason):
    r = Replay()
    r.start()
    out = r.tick(**values)
    assert out.state == MissionState.FAULT and out.reason == reason
    assert r.tick().velocity == (0., 0., 0.)


def test_future_and_reordered_sensor_stamps_stop_and_replayed_events_do_not_progress():
    r = Replay()
    r.start()
    assert r.tick(vision_stamp=100).mode == "stop"
    r.frames(count=3)
    assert r.tick(tof_stamp=r.now - .05).reason == "reordered_tof"
    r.frames(count=3)
    assert r.tick(vision_stamp=r.now - .1).reason == "reordered_vision"


def test_straight_route_and_unmapped_return_route():
    r = Replay(route_map={"yellow": {"red": "straight"}}, reacquire_frames=1)
    r.start()
    r.frames(colours=["red"])
    assert r.tick().velocity == (.06, 0., 0.)
    assert r.frames(events={"turn_complete"}).state == MissionState.FOLLOW
    r = Replay(return_route_map={})
    r.deliver()
    r.align_return()
    assert r.frames(colours=["green"]).reason == "unmapped_route:green"


def test_continuously_visible_blue_is_not_a_second_obstacle():
    r = Replay()
    r.start()
    r.frames(colours=["blue"])
    r.frames(colours=["blue"], events={"obstacle_near"})
    r.frames(colours=["blue"], events={"obstacle_crossed"})
    assert r.frames(count=31, colours=["blue"], stable=True).state == MissionState.FOLLOW
    assert r.frames(colours=["blue"]).state == MissionState.FOLLOW
    assert r.fsm.obstacles_completed == 1


def test_sensor_fault_cannot_evade_timeout_by_alternating_bad_sources():
    r = Replay(sensor_fault_timeout_s=.5)
    r.start()
    r.tick(vision_stamp=None)
    r.tick(tof_valid=False)
    r.tick(vision_stamp=None)
    out = r.tick(dt=.3, tof_valid=False)
    assert out.state == MissionState.FAULT and out.reason == "stale_vision_timeout"


@pytest.mark.parametrize("values", [{"stable": "false"}, {"tof_valid": 1}, {"start": "true"}])
def test_json_boolean_strings_cannot_confirm_evidence(values):
    r = Replay()
    r.start()
    assert r.tick(**values).reason == "invalid_input_boolean"


def test_bad_line_payload_stops_instead_of_raising():
    r = Replay()
    r.start()
    out = r.tick(line=None)
    assert out.mode == "stop" and out.reason == "lost_line"


def test_ambiguous_household_cannot_issue_delivery():
    r = Replay()
    r.start()
    assert r.tick(colours=["yellow", "brown"]).reason == "ambiguous_household"
    assert not r.requests


def test_route_config_is_snapshotted_against_external_mutation():
    routes = {"yellow": {"red": "left"}}
    r = Replay(route_map=routes)
    routes["yellow"]["red"] = "right"
    r.start()
    r.frames(colours=["red"])
    assert r.tick().velocity[2] > 0
