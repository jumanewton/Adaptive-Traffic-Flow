"""Tests for Phase 6 advanced control modules."""

from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timezone

from traffic_control.mpc import MPCController, TrafficFlowModel
from traffic_control.coordination import (
    MultiIntersectionCoordinator,
    CoordinationStrategy,
    IntersectionLink,
    CorridorConfig,
)
from traffic_control.emergency import (
    EmergencyVehicleDetector,
    PreemptionManager,
    EmergencyVehicle,
    EmergencyVehicleType,
    PreemptionStatus,
    EmergencyPriorityController,
)
from traffic_control.rl import (
    RLController,
    QLearningAgent,
    TrafficEnv,
    RLTransition,
)
from traffic_domain.schemas import (
    IntersectionConfig,
    ApproachConfig,
    LaneConfig,
    SignalPhase,
    TrafficState,
    ApproachState,
    LaneState,
    ControllerAction,
)
from traffic_config.settings import ControlSettings, get_settings


def create_test_intersection(intersection_id: str = "test-intersection") -> IntersectionConfig:
    """Create a test intersection configuration."""
    return IntersectionConfig(
        intersection_id=intersection_id,
        name="Test Intersection",
        approaches=[
            ApproachConfig(
                approach_id="northbound",
                name="Northbound",
                lanes=[
                    LaneConfig(lane_id="nb_left", approach_id="northbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="nb_thru", approach_id="northbound"),
                    LaneConfig(lane_id="nb_right", approach_id="northbound", is_turning=True, turn_direction="right"),
                ],
                signal_group="NS",
                has_crosswalk=True,
            ),
            ApproachConfig(
                approach_id="southbound",
                name="Southbound",
                lanes=[
                    LaneConfig(lane_id="sb_left", approach_id="southbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="sb_thru", approach_id="southbound"),
                    LaneConfig(lane_id="sb_right", approach_id="southbound", is_turning=True, turn_direction="right"),
                ],
                signal_group="NS",
                has_crosswalk=True,
            ),
            ApproachConfig(
                approach_id="eastbound",
                name="Eastbound",
                lanes=[
                    LaneConfig(lane_id="eb_left", approach_id="eastbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="eb_thru", approach_id="eastbound"),
                    LaneConfig(lane_id="eb_right", approach_id="eastbound", is_turning=True, turn_direction="right"),
                ],
                signal_group="EW",
                has_crosswalk=True,
            ),
            ApproachConfig(
                approach_id="westbound",
                name="Westbound",
                lanes=[
                    LaneConfig(lane_id="wb_left", approach_id="westbound", is_turning=True, turn_direction="left"),
                    LaneConfig(lane_id="wb_thru", approach_id="westbound"),
                    LaneConfig(lane_id="wb_right", approach_id="westbound", is_turning=True, turn_direction="right"),
                ],
                signal_group="EW",
                has_crosswalk=True,
            ),
        ],
        phases=[
            SignalPhase.NORTH_SOUTH_GREEN,
            SignalPhase.NORTH_SOUTH_YELLOW,
            SignalPhase.EAST_WEST_GREEN,
            SignalPhase.EAST_WEST_YELLOW,
        ],
    )


def create_test_state(intersection: IntersectionConfig) -> TrafficState:
    """Create a test traffic state."""
    now = datetime.now(timezone.utc)
    approaches = []
    for approach_config in intersection.approaches:
        lanes = {}
        for lane_config in approach_config.lanes:
            lanes[lane_config.lane_id] = LaneState(
                lane_id=lane_config.lane_id,
                vehicle_count=5,
                queue_length_m=25.0,
                occupancy=0.3,
                average_speed_kmh=30.0,
                measured_at=now,
            )
        approaches.append(ApproachState(
            approach_id=approach_config.approach_id,
            vehicle_count=15,
            queue_length_m=25.0,
            average_speed_kmh=30.0,
            occupancy=0.3,
            lanes=lanes,
            measured_at=now,
        ))

    return TrafficState(
        intersection_id=intersection.intersection_id,
        observed_at=now,
        approaches=approaches,
    )


class TestMPCController:
    """Test MPC Controller."""

    def test_mpc_creation(self):
        """Test MPC controller creation."""
        intersection = create_test_intersection()
        settings = get_settings().control
        controller = MPCController(intersection, settings, horizon=20)

        assert controller is not None
        assert controller._horizon == 20

    def test_mpc_compute_action(self):
        """Test MPC action computation."""
        intersection = create_test_intersection()
        settings = get_settings().control
        controller = MPCController(intersection, settings, horizon=10)

        state = create_test_state(intersection)
        recommendation = controller.compute_action(state)

        assert recommendation is not None
        assert recommendation.controller_type == "mpc"
        assert recommendation.action in [ControllerAction.HOLD_PHASE, ControllerAction.SWITCH_PHASE, ControllerAction.EXTEND_GREEN]

    def test_traffic_flow_model(self):
        """Test traffic flow model prediction."""
        intersection = create_test_intersection()
        model = TrafficFlowModel(intersection, dt=1.0)

        state = create_test_state(intersection)
        phase_seq = [SignalPhase.NORTH_SOUTH_GREEN, SignalPhase.EAST_WEST_GREEN]
        durations = [30.0, 30.0]

        queue_pred, wait_pred = model.predict(state, phase_seq, durations, horizon=10)

        assert queue_pred.shape == (10, 4)
        assert wait_pred.shape == (10, 4)
        assert np.all(queue_pred >= 0)
        assert np.all(wait_pred >= 0)


class TestCoordination:
    """Test Multi-Intersection Coordination."""

    def test_coordinator_creation(self):
        """Test coordinator creation."""
        intersection1 = create_test_intersection("int-1")
        intersection2 = create_test_intersection("int-2")

        link = IntersectionLink(
            upstream_id="int-1",
            downstream_id="int-2",
            distance_m=500.0,
            travel_time_s=36.0,
            approach_upstream="southbound",
            approach_downstream="northbound",
        )

        settings = get_settings().control
        coordinator = MultiIntersectionCoordinator(
            intersections={"int-1": intersection1, "int-2": intersection2},
            links=[link],
            settings=settings,
            strategy=CoordinationStrategy.GREEN_WAVE,
        )

        assert coordinator is not None
        assert coordinator._strategy == CoordinationStrategy.GREEN_WAVE

    def test_coordination_plan(self):
        """Test coordination plan computation."""
        intersection1 = create_test_intersection("int-1")
        intersection2 = create_test_intersection("int-2")

        link = IntersectionLink(
            upstream_id="int-1",
            downstream_id="int-2",
            distance_m=500.0,
            travel_time_s=36.0,
            approach_upstream="southbound",
            approach_downstream="northbound",
        )

        settings = get_settings().control
        coordinator = MultiIntersectionCoordinator(
            intersections={"int-1": intersection1, "int-2": intersection2},
            links=[link],
            settings=settings,
        )

        plan = coordinator.compute_coordination_plan()

        assert plan is not None
        assert plan.cycle_length > 0
        assert "int-1" in plan.offsets
        assert "int-2" in plan.offsets
        assert 0 <= plan.bandwidth <= 1


class TestEmergencyPriority:
    """Test Emergency Vehicle Priority."""

    def test_emergency_detector(self):
        """Test emergency vehicle detector."""
        settings = get_settings().control
        detector = EmergencyVehicleDetector(settings)

        vehicle = detector.process_detection(
            vehicle_id="amb-001",
            vehicle_type=EmergencyVehicleType.AMBULANCE,
            intersection_id="test-intersection",
            approach_id="northbound",
            distance_m=200.0,
            speed_kmh=50.0,
        )

        assert vehicle.vehicle_id == "amb-001"
        assert vehicle.vehicle_type == EmergencyVehicleType.AMBULANCE
        assert vehicle.eta_seconds > 0

        # Check retrieval
        vehicles = detector.get_vehicles_for_intersection("test-intersection")
        assert len(vehicles) == 1

        highest = detector.get_highest_priority_vehicle("test-intersection")
        assert highest.vehicle_id == "amb-001"

    def test_preemption_manager(self):
        """Test preemption manager."""
        intersection = create_test_intersection()
        settings = get_settings().control
        detector = EmergencyVehicleDetector(settings)
        preemption = PreemptionManager(intersection, settings, detector)

        # No emergency - no preemption
        state = create_test_state(intersection)
        preempted, rec = preemption.check_preemption(state)
        assert not preempted

        # Add emergency vehicle
        detector.process_detection(
            vehicle_id="amb-001",
            vehicle_type=EmergencyVehicleType.AMBULANCE,
            intersection_id="test-intersection",
            approach_id="northbound",
            distance_m=100.0,  # Close enough for preemption
            speed_kmh=50.0,
        )

        preempted, rec = preemption.check_preemption(state)
        assert preempted
        assert rec is not None
        assert rec.controller_type == "emergency_preemption"

        # Check status
        status = preemption.get_preemption_status()
        assert status["status"] == "active"

    def test_emergency_priority_controller(self):
        """Test emergency priority controller wrapper."""
        from traffic_control.fixed_time import FixedTimeController

        intersection = create_test_intersection()
        settings = get_settings().control
        base_controller = FixedTimeController(intersection, settings)

        emergency_controller = EmergencyPriorityController(
            base_controller, intersection, settings
        )

        state = create_test_state(intersection)
        rec = emergency_controller.compute_action(state)

        # Should be normal fixed time action
        assert rec.controller_type == "fixed_time"

        # Add emergency
        emergency_controller.process_emergency_event(
            type('Event', (), {
                'event_type': 'emergency_vehicle',
                'intersection_id': 'test-intersection',
                'payload': {
                    'vehicle_id': 'amb-001',
                    'vehicle_type': 'ambulance',
                    'approach_id': 'northbound',
                    'distance_m': 100.0,
                    'speed_kmh': 50.0,
                    'siren_active': True,
                }
            })()
        )

        rec = emergency_controller.compute_action(state)
        assert rec.controller_type == "emergency_preemption"


class TestRLController:
    """Test RL Controller."""

    def test_rl_environment(self):
        """Test traffic environment."""
        intersection = create_test_intersection()
        settings = get_settings().control
        env = TrafficEnv(intersection, settings)

        obs = env.reset()
        assert obs.shape[0] == env._obs_dim

        # Step
        obs, reward, done, info = env.step(0)  # HOLD
        assert obs.shape[0] == env._obs_dim
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_qlearning_agent(self):
        """Test Q-learning agent."""
        intersection = create_test_intersection()
        settings = get_settings().control
        env = TrafficEnv(intersection, settings)

        agent = QLearningAgent(
            state_dim=env._obs_dim,
            action_dim=env._action_space,
        )

        obs = env.reset()
        action = agent.act(obs, training=True)
        assert 0 <= action < env._action_space

        # Learn
        next_obs, reward, done, _ = env.step(action)
        from traffic_control.rl import RLTransition
        transition = RLTransition(
            state=obs,
            action=action,
            reward=reward,
            next_state=next_obs,
            done=done,
        )
        agent.learn(transition)

    def test_rl_controller(self):
        """Test RL controller."""
        intersection = create_test_intersection()
        settings = get_settings().control

        controller = RLController(intersection, settings, training=False)

        state = create_test_state(intersection)
        rec = controller.compute_action(state)

        assert rec is not None
        assert rec.controller_type == "rl"
        assert rec.action in [ControllerAction.HOLD_PHASE, ControllerAction.SWITCH_PHASE, ControllerAction.EXTEND_GREEN]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])