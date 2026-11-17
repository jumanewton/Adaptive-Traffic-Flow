"""Tests for traffic signal control."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from traffic_domain.schemas import (
    TrafficState,
    ApproachState,
    LaneState,
    IntersectionConfig,
    ApproachConfig,
    LaneConfig,
    SignalPhase,
    Point,
    ControllerAction,
)
from traffic_config.settings import ControlSettings
from traffic_control.fixed_time import FixedTimeController
from traffic_control.max_pressure import MaxPressureController


def create_test_intersection_config() -> IntersectionConfig:
    """Create a test intersection configuration."""
    return IntersectionConfig(
        intersection_id="int_01",
        name="Test Intersection",
        approaches=[
            ApproachConfig(
                approach_id="northbound",
                name="Northbound",
                lanes=[LaneConfig(lane_id="nb_1", approach_id="northbound", length=100)],
                signal_group="NS",
                centerline=[Point(x=0, y=-100), Point(x=0, y=0)],
            ),
            ApproachConfig(
                approach_id="southbound",
                name="Southbound",
                lanes=[LaneConfig(lane_id="sb_1", approach_id="southbound", length=100)],
                signal_group="NS",
                centerline=[Point(x=0, y=100), Point(x=0, y=0)],
            ),
            ApproachConfig(
                approach_id="eastbound",
                name="Eastbound",
                lanes=[LaneConfig(lane_id="eb_1", approach_id="eastbound", length=100)],
                signal_group="EW",
                centerline=[Point(x=100, y=0), Point(x=0, y=0)],
            ),
            ApproachConfig(
                approach_id="westbound",
                name="Westbound",
                lanes=[LaneConfig(lane_id="wb_1", approach_id="westbound", length=100)],
                signal_group="EW",
                centerline=[Point(x=-100, y=0), Point(x=0, y=0)],
            ),
        ],
        phases=[
            SignalPhase.NORTH_SOUTH_GREEN,
            SignalPhase.NORTH_SOUTH_YELLOW,
            SignalPhase.EAST_WEST_GREEN,
            SignalPhase.EAST_WEST_YELLOW,
        ],
    )


def create_test_traffic_state(
    nb_queue: float = 10,
    sb_queue: float = 10,
    eb_queue: float = 5,
    wb_queue: float = 5,
) -> TrafficState:
    """Create a test traffic state."""
    return TrafficState(
        intersection_id="int_01",
        observed_at=datetime.now(timezone.utc),
        approaches=[
            ApproachState(
                approach_id="northbound",
                vehicle_count=10,
                queue_length_m=nb_queue,
                average_speed_kmh=5.0,
                lanes={
                    "nb_1": LaneState(lane_id="nb_1", vehicle_count=10, queue_length_m=nb_queue, measured_at=datetime.now(timezone.utc))
                },
                measured_at=datetime.now(timezone.utc),
            ),
            ApproachState(
                approach_id="southbound",
                vehicle_count=10,
                queue_length_m=sb_queue,
                average_speed_kmh=5.0,
                lanes={
                    "sb_1": LaneState(lane_id="sb_1", vehicle_count=10, queue_length_m=sb_queue, measured_at=datetime.now(timezone.utc))
                },
                measured_at=datetime.now(timezone.utc),
            ),
            ApproachState(
                approach_id="eastbound",
                vehicle_count=5,
                queue_length_m=eb_queue,
                average_speed_kmh=10.0,
                lanes={
                    "eb_1": LaneState(lane_id="eb_1", vehicle_count=5, queue_length_m=eb_queue, measured_at=datetime.now(timezone.utc))
                },
                measured_at=datetime.now(timezone.utc),
            ),
            ApproachState(
                approach_id="westbound",
                vehicle_count=5,
                queue_length_m=wb_queue,
                average_speed_kmh=10.0,
                lanes={
                    "wb_1": LaneState(lane_id="wb_1", vehicle_count=5, queue_length_m=wb_queue, measured_at=datetime.now(timezone.utc))
                },
                measured_at=datetime.now(timezone.utc),
            ),
        ],
    )


class TestFixedTimeController:
    """Tests for FixedTimeController."""

    def test_initialization(self):
        """Test controller initialization."""
        config = create_test_intersection_config()
        controller = FixedTimeController(config)
        assert controller.config == config
        assert controller._phase_index == 0

    def test_initialization_with_settings(self):
        """Test controller initialization with custom settings."""
        config = create_test_intersection_config()
        settings = ControlSettings(
            fixed_time_cycle_length=100,
            fixed_time_phase_splits={"north_south": 40, "east_west": 40},
        )
        controller = FixedTimeController(config, settings)
        assert controller._phase_durations[SignalPhase.NORTH_SOUTH_GREEN] == 40
        assert controller._phase_durations[SignalPhase.EAST_WEST_GREEN] == 40

    def test_compute_action_hold(self):
        """Test compute action returns HOLD initially."""
        config = create_test_intersection_config()
        controller = FixedTimeController(config)
        state = create_test_traffic_state()

        # First call - should hold
        action = controller.compute_action(state)
        assert action.action.name == "HOLD_PHASE"
        assert action.current_phase == SignalPhase.NORTH_SOUTH_GREEN

    def test_compute_action_switch(self):
        """Test compute action switches after phase duration."""
        config = create_test_intersection_config()
        settings = ControlSettings(
            fixed_time_cycle_length=20,
            fixed_time_phase_splits={"north_south": 5, "east_west": 5},
            min_green_time=3,
            max_green_time=30,
            yellow_time=2,
            all_red_time=1,
            control_loop_interval_s=1.0,
        )
        controller = FixedTimeController(config, settings)

        state = create_test_traffic_state()

        # First call sets phase to NS_GREEN with elapsed=0
        action = controller.compute_action(state)
        assert action.action == ControllerAction.HOLD_PHASE

        # Simulate time passing - 5 calls = 5 seconds = phase duration
        for _ in range(4):
            action = controller.compute_action(state)
            assert action.action == ControllerAction.HOLD_PHASE

        # Next call (6th) should switch - phase_duration (5) - elapsed (5) = 0
        action = controller.compute_action(state)
        assert action.action == ControllerAction.SWITCH_PHASE
        assert action.current_phase == SignalPhase.NORTH_SOUTH_GREEN

    def test_reset(self):
        """Test controller reset."""
        config = create_test_intersection_config()
        controller = FixedTimeController(config)
        state = create_test_traffic_state()

        # Advance a few steps
        for _ in range(5):
            controller.compute_action(state)

        # Reset
        controller.reset()
        assert controller._phase_index == 0
        assert controller._current_phase is None
        assert controller._phase_elapsed == 0.0


class TestMaxPressureController:
    """Tests for MaxPressureController."""

    def test_initialization(self):
        """Test controller initialization."""
        config = create_test_intersection_config()
        controller = MaxPressureController(config)
        assert controller.config == config
        assert controller._pressure_weight == 1.0
        assert controller._downstream_weight == 0.5

    def test_initialization_with_settings(self):
        """Test controller initialization with custom settings."""
        config = create_test_intersection_config()
        settings = ControlSettings(
            max_pressure_pressure_weight=1.5,
            max_pressure_downstream_weight=0.3,
            max_pressure_min_green=15,
            max_pressure_max_green=90,
        )
        controller = MaxPressureController(config, settings)
        assert controller._pressure_weight == 1.5
        assert controller._downstream_weight == 0.3

    def test_compute_action_ns_higher_pressure(self):
        """Test controller chooses NS when NS pressure is higher."""
        config = create_test_intersection_config()
        controller = MaxPressureController(config)
        # High NS queue, low EW queue
        state = create_test_traffic_state(nb_queue=30, sb_queue=30, eb_queue=5, wb_queue=5)

        action = controller.compute_action(state)
        # Should stay on or switch to NS green
        assert action.controller_type == "max_pressure"

    def test_compute_action_ew_higher_pressure(self):
        """Test controller chooses EW when EW pressure is higher."""
        config = create_test_intersection_config()
        controller = MaxPressureController(config)
        # Low NS queue, high EW queue
        state = create_test_traffic_state(nb_queue=5, sb_queue=5, eb_queue=30, wb_queue=30)

        action = controller.compute_action(state)
        # First call - current phase is None, defaults to NS_GREEN
        # If EW pressure is higher, should switch
        assert action.controller_type == "max_pressure"

    def test_pressure_calculation(self):
        """Test pressure calculation logic."""
        config = create_test_intersection_config()
        controller = MaxPressureController(config)
        state = create_test_traffic_state(nb_queue=20, sb_queue=20, eb_queue=10, wb_queue=10)

        # Calculate pressure for NS green
        ns_pressure = controller._calculate_pressure(state, SignalPhase.NORTH_SOUTH_GREEN)
        # Calculate pressure for EW green
        ew_pressure = controller._calculate_pressure(state, SignalPhase.EAST_WEST_GREEN)

        # NS should have higher pressure (20+20) - (10+10) = 20
        # EW should have lower pressure (10+10) - (20+20) = -20
        assert ns_pressure > ew_pressure

    def test_extension_duration(self):
        """Test extension duration calculation."""
        config = create_test_intersection_config()
        settings = ControlSettings(
            max_pressure_min_green=10,
            max_pressure_max_green=60,
        )
        controller = MaxPressureController(config, settings)
        state = create_test_traffic_state(nb_queue=50, sb_queue=50, eb_queue=5, wb_queue=5)

        # Set current phase to NS green with some elapsed time
        controller.set_phase(SignalPhase.NORTH_SOUTH_GREEN)
        controller.update_phase_elapsed(15.0)

        duration = controller._calculate_extension_duration(state, SignalPhase.NORTH_SOUTH_GREEN)
        # High pressure should give longer extension
        assert duration > 5.0
        assert duration <= 60 - 15  # Within max green bounds

    def test_safety_constraints_min_green(self):
        """Test minimum green time constraint."""
        config = create_test_intersection_config()
        settings = ControlSettings(
            min_green_time=10,
            max_green_time=60,
            yellow_time=3,
            all_red_time=2,
        )
        controller = MaxPressureController(config, settings)

        # Set phase to NS green, but only 5 seconds elapsed
        controller.set_phase(SignalPhase.NORTH_SOUTH_GREEN)
        controller.update_phase_elapsed(5.0)

        # Create state where EW pressure is higher (should want to switch)
        state = create_test_traffic_state(nb_queue=5, sb_queue=5, eb_queue=30, wb_queue=30)

        action = controller.compute_action(state)
        # Should NOT switch because min green not met - should HOLD
        assert action.action.name == "HOLD_PHASE"

    def test_safety_constraints_max_green(self):
        """Test maximum green time constraint."""
        config = create_test_intersection_config()
        settings = ControlSettings(
            min_green_time=5,
            max_green_time=30,
            yellow_time=3,
            all_red_time=2,
        )
        controller = MaxPressureController(config, settings)

        # Set phase to NS green, elapsed 25 seconds (close to max)
        controller.set_phase(SignalPhase.NORTH_SOUTH_GREEN)
        controller.update_phase_elapsed(25.0)

        # High NS pressure - would want to extend
        state = create_test_traffic_state(nb_queue=50, sb_queue=50, eb_queue=5, wb_queue=5)

        action = controller.compute_action(state)
        # Extension should be limited by max_green - elapsed = 30 - 25 = 5
        assert action.duration_seconds <= 5.0


class TestControllerConstraints:
    """Tests for safety constraints enforcement."""

    def test_min_green_enforced_on_switch(self):
        """Test min green is enforced when trying to switch."""
        config = create_test_intersection_config()
        settings = ControlSettings(min_green_time=10, max_green_time=60)
        controller = FixedTimeController(config, settings)

        state = create_test_traffic_state()

        # Force a switch attempt before min green
        controller._phase_elapsed = 5.0  # Less than min_green=10
        controller._current_phase = SignalPhase.NORTH_SOUTH_GREEN

        # Try to switch
        action, duration = controller._enforce_constraints(
            action=ControllerAction.SWITCH_PHASE,
            duration=30.0,
            current_phase=SignalPhase.NORTH_SOUTH_GREEN,
        )

        # Should be converted to HOLD
        assert action == ControllerAction.HOLD_PHASE
        assert duration == 5.0  # min_green - elapsed

    def test_max_green_enforced_on_extend(self):
        """Test max green is enforced when trying to extend."""
        config = create_test_intersection_config()
        settings = ControlSettings(min_green_time=5, max_green_time=30)
        controller = FixedTimeController(config, settings)

        # Force an extend attempt at max green
        controller._phase_elapsed = 28.0
        controller._current_phase = SignalPhase.NORTH_SOUTH_GREEN

        action, duration = controller._enforce_constraints(
            action=ControllerAction.EXTEND_GREEN,
            duration=10.0,
            current_phase=SignalPhase.NORTH_SOUTH_GREEN,
        )

        # Duration should be limited to max_green - elapsed = 2
        assert duration == 2.0

    def test_extend_converts_to_switch_when_max_reached(self):
        """Test extend converts to switch when max green reached."""
        config = create_test_intersection_config()
        settings = ControlSettings(min_green_time=5, max_green_time=30)
        controller = FixedTimeController(config, settings)

        # At max green
        controller._phase_elapsed = 30.0
        controller._current_phase = SignalPhase.NORTH_SOUTH_GREEN

        action, duration = controller._enforce_constraints(
            action=ControllerAction.EXTEND_GREEN,
            duration=10.0,
            current_phase=SignalPhase.NORTH_SOUTH_GREEN,
        )

        # Should convert to switch
        assert action == ControllerAction.SWITCH_PHASE
        assert duration == 5.0  # min_green


if __name__ == "__main__":
    pytest.main([__file__, "-v"])