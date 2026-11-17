"""SUMO simulation runner using TraCI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from traffic_config.settings import SimulationSettings, get_settings
from traffic_domain.schemas import (
    TrafficState,
    ControllerRecommendation,
    SignalPhase,
    IntersectionConfig,
    ApproachState,
    LaneState,
)
from traffic_simulation.interface import SimulatorInterface, SimulatorState
from traffic_observability.logging import get_logger


logger = get_logger(__name__)


class SUMORunner(SimulatorInterface):
    """SUMO simulation runner using TraCI."""

    def __init__(self, settings: SimulationSettings | None = None):
        """Initialize SUMO runner."""
        self._settings = settings or get_settings().simulation
        self._traci = None
        self._sumo_process = None
        self._running = False
        self._step = 0
        self._intersection_config: IntersectionConfig | None = None
        self._phase_mapping: dict[str, int] = {}  # phase name -> SUMO phase index

    def _ensure_sumo_home(self) -> None:
        """Ensure SUMO_HOME is set."""
        sumo_home = self._settings.sumo_home or os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise RuntimeError(
                "SUMO_HOME not set. Set SUMO_HOME environment variable or configure in settings."
            )

        tools_path = Path(sumo_home) / "tools"
        if str(tools_path) not in sys.path:
            sys.path.append(str(tools_path))

    def start(self, config: IntersectionConfig) -> None:
        """Start SUMO simulation."""
        self._ensure_sumo_home()

        import traci

        self._traci = traci

        # Build SUMO command
        sumo_binary = self._settings.sumo_binary
        if self._settings.sumo_gui_binary and os.environ.get("SUMO_GUI"):
            sumo_binary = self._settings.sumo_gui_binary

        sumo_cmd = [
            sumo_binary,
            "-c", str(self._settings.network_file),
            "--step-length", str(self._settings.step_length),
            "--seed", str(self._settings.seed),
            "--remote-port", str(self._settings.traci_port),
            "--start",
        ]

        if self._settings.route_file:
            sumo_cmd.extend(["-r", str(self._settings.route_file)])

        for add_file in self._settings.additional_files:
            sumo_cmd.extend(["-a", str(add_file)])

        # Output options
        output_dir = Path(self._settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._settings.emit_queue:
            sumo_cmd.extend(["--queue-output", str(output_dir / "queue.xml")])

        if self._settings.emit_emissions:
            sumo_cmd.extend(["--emission-output", str(output_dir / "emissions.xml")])

        if self._settings.emit_fcd:
            sumo_cmd.extend(["--fcd-output", str(output_dir / "fcd.xml")])

        logger.info("Starting SUMO", command=" ".join(sumo_cmd))

        # Start TraCI connection
        try:
            self._traci.start(sumo_cmd, port=self._settings.traci_port)
        except Exception as e:
            logger.error("Failed to start SUMO", error=str(e))
            raise

        self._running = True
        self._intersection_config = config
        self._step = 0

        # Initialize phase mapping
        self._init_phase_mapping(config)

        logger.info("SUMO started successfully", port=self._settings.traci_port)

    def _init_phase_mapping(self, config: IntersectionConfig) -> None:
        """Initialize mapping from our phase names to SUMO phase indices."""
        # Get traffic light IDs
        tl_ids = self._traci.trafficlight.getIDList()

        for tl_id in tl_ids:
            program = self._traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
            phases = program.phases

            # Map our phases to SUMO phases
            for i, phase in enumerate(phases):
                phase_state = phase.state
                # Simple mapping based on phase state pattern
                if "G" in phase_state and "g" not in phase_state:
                    if "N" in phase_state or "S" in phase_state:
                        self._phase_mapping["north_south_green"] = i
                    elif "E" in phase_state or "W" in phase_state:
                        self._phase_mapping["east_west_green"] = i
                elif "y" in phase_state.lower():
                    if "N" in phase_state or "S" in phase_state:
                        self._phase_mapping["north_south_yellow"] = i
                    elif "E" in phase_state or "W" in phase_state:
                        self._phase_mapping["east_west_yellow"] = i

        logger.info("Phase mapping initialized", mapping=self._phase_mapping)

    def step(self, actions: dict[str, ControllerRecommendation]) -> SimulatorState:
        """Advance simulation by one step."""
        if not self._running:
            raise RuntimeError("Simulation not running")

        # Apply control actions
        for intersection_id, action in actions.items():
            self._apply_action(intersection_id, action)

        # Step simulation
        self._traci.simulationStep()
        self._step += 1

        # Collect state
        state = self.get_state()
        return state

    def _apply_action(self, intersection_id: str, action: ControllerRecommendation) -> None:
        """Apply control action to SUMO traffic light."""
        # Map our phase to SUMO phase index
        sumo_phase = self._phase_mapping.get(action.current_phase.value.lower())
        if sumo_phase is None:
            logger.warning("No SUMO phase mapping for", phase=action.current_phase.value)
            return

        # For now, just set the phase
        # In reality, we'd need to handle phase transitions properly
        tl_ids = self._traci.trafficlight.getIDList()
        if tl_ids:
            self._traci.trafficlight.setPhase(tl_ids[0], sumo_phase)

    def get_state(self) -> SimulatorState:
        """Get current simulation state."""
        if not self._running:
            raise RuntimeError("Simulation not running")

        # Get vehicle data
        vehicle_ids = self._traci.vehicle.getIDList()
        vehicles = {}
        for vid in vehicle_ids:
            try:
                pos = self._traci.vehicle.getPosition(vid)
                speed = self._traci.vehicle.getSpeed(vid)
                lane = self._traci.vehicle.getLaneID(vid)
                vehicles[vid] = {
                    "position": pos,
                    "speed": speed,
                    "lane": lane,
                }
            except Exception:
                pass

        # Get traffic light states
        intersections = {}
        if self._intersection_config:
            tl_ids = self._traci.trafficlight.getIDList()
            for tl_id in tl_ids:
                try:
                    phase = self._traci.trafficlight.getPhase(tl_id)
                    # Convert SUMO phase to our state
                    approach_states = []
                    # Simplified: create dummy approach states
                    for approach in self._intersection_config.approaches:
                        approach_states.append(ApproachState(
                            approach_id=approach.approach_id,
                            vehicle_count=0,
                            queue_length_m=0.0,
                            average_speed_kmh=0.0,
                            measured_at=datetime.now(timezone.utc),
                        ))

                    intersections[tl_id] = TrafficState(
                        intersection_id=tl_id,
                        observed_at=datetime.now(timezone.utc),
                        approaches=approach_states,
                        source="simulation",
                    )
                except Exception as e:
                    logger.warning("Failed to get TL state", tl_id=tl_id, error=str(e))

        return SimulatorState(
            step=self._step,
            time=self._step * self._settings.step_length,
            intersections=intersections,
            vehicles=vehicles,
            metrics={},
        )

    def set_signal(self, intersection_id: str, phase: SignalPhase) -> None:
        """Set signal phase for an intersection."""
        if not self._running:
            raise RuntimeError("Simulation not running")

        sumo_phase = self._phase_mapping.get(phase.value.lower())
        if sumo_phase is not None:
            tl_ids = self._traci.trafficlight.getIDList()
            if tl_ids:
                self._traci.trafficlight.setPhase(tl_ids[0], sumo_phase)

    def close(self) -> None:
        """Close SUMO simulation."""
        if self._running and self._traci:
            try:
                self._traci.close()
            except Exception as e:
                logger.warning("Error closing TraCI", error=str(e))
            self._running = False
            logger.info("SUMO closed")

    @property
    def is_running(self) -> bool:
        return self._running

    def run_episode(
        self,
        controller,
        max_steps: int | None = None,
    ) -> list[SimulatorState]:
        """Run a full simulation episode with a controller.

        Args:
            controller: Controller instance with compute_action method
            max_steps: Maximum steps (defaults to simulation_duration)

        Returns:
            List of simulator states per step
        """
        max_steps = max_steps or int(self._settings.simulation_duration / self._settings.step_length)
        states = []

        for step in range(max_steps):
            # Get current state
            state = self.get_state()

            # Compute actions for each intersection
            actions = {}
            for intersection_id, traffic_state in state.intersections.items():
                action = controller.compute_action(traffic_state)
                actions[intersection_id] = action

            # Step simulation
            next_state = self.step(actions)
            states.append(next_state)

            # Check if simulation ended
            if self._traci.simulation.getMinExpectedNumber() == 0:
                logger.info("All vehicles arrived, ending simulation")
                break

        return states