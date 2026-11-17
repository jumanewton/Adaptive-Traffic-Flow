"""Control Service main entry point.

Runs simulation experiments comparing different traffic signal controllers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from traffic_config.settings import get_settings
from traffic_domain.schemas import (
    ControllerRecommendation,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    IntersectionConfig,
    SignalPhase,
    TrafficState,
)
from traffic_observability.logging import configure_logging, get_logger
from traffic_observability.metrics import get_metrics_registry, create_traffic_metrics
from traffic_simulation.sumo_runner import SUMORunner
from traffic_control.base import BaseController
from traffic_control.fixed_time import FixedTimeController
from traffic_control.max_pressure import MaxPressureController

logger = get_logger(__name__)


class ExperimentRunner:
    """Runs controlled simulation experiments comparing signal controllers."""

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        settings: Any | None = None,
    ):
        """Initialize experiment runner.

        Args:
            intersection_config: Intersection configuration
            settings: Optional simulation settings override
        """
        self.config = intersection_config
        self._settings = settings or get_settings()
        self._runner = SUMORunner(self._settings.simulation)
        self._metrics = get_metrics_registry()
        self._traffic_metrics = create_traffic_metrics(self._metrics, "experiment_runner")

    def create_controller(
        self,
        controller_type: str,
        intersection_config: IntersectionConfig,
    ) -> BaseController:
        """Create controller instance by type."""
        control_settings = self._settings.control

        if controller_type == "fixed_time":
            return FixedTimeController(intersection_config, control_settings)
        elif controller_type == "max_pressure":
            return MaxPressureController(intersection_config, control_settings)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

    def run_single_experiment(
        self,
        experiment_config: ExperimentConfig,
        controller_type: str,
    ) -> ExperimentResult:
        """Run a single experiment with given controller.

        Args:
            experiment_config: Experiment configuration
            controller_type: Type of controller to test

        Returns:
            Experiment results
        """
        experiment_id = uuid4()
        logger.info(
            "Starting experiment",
            experiment_id=str(experiment_id),
            controller=controller_type,
            steps=experiment_config.simulation_steps,
        )

        # Create controller
        controller = self.create_controller(controller_type, self.config)

        # Start simulation
        self._runner.start(self.config)

        # Metrics collection
        waiting_times = []
        queue_lengths = []
        throughput = 0
        stops = 0
        phase_changes = 0
        prev_phase = None

        try:
            max_steps = experiment_config.simulation_steps
            step_length = self._settings.simulation.step_length

            for step in range(max_steps):
                # Get current state
                state = self._runner.get_state()

                # Collect metrics from all intersections
                for intersection_id, traffic_state in state.intersections.items():
                    # Waiting time (sum of queue lengths * step_length)
                    for approach in traffic_state.approaches:
                        waiting_times.append(approach.queue_length_m * step_length)
                        queue_lengths.append(approach.queue_length_m)

                    # Count phase changes
                    if traffic_state.current_phase and traffic_state.current_phase != prev_phase:
                        phase_changes += 1
                        prev_phase = traffic_state.current_phase

                # Compute actions for each intersection
                actions = {}
                for intersection_id, traffic_state in state.intersections.items():
                    action = controller.compute_action(traffic_state)
                    actions[intersection_id] = action

                # Step simulation
                next_state = self._runner.step(actions)

                # Count throughput (vehicles that arrived)
                if hasattr(next_state, 'vehicles'):
                    throughput += len(next_state.vehicles)

                # Progress logging
                if step % 300 == 0 and step > 0:
                    elapsed = step * step_length
                    logger.info(
                        "Experiment progress",
                        step=step,
                        elapsed_s=elapsed,
                        intersections=len(state.intersections),
                    )

                # Check if simulation ended
                if hasattr(self._runner, '_traci') and self._runner._traci:
                    try:
                        if self._runner._traci.simulation.getMinExpectedNumber() == 0:
                            logger.info("All vehicles arrived, ending early")
                            break
                    except Exception:
                        pass

        except Exception as e:
            logger.exception("Experiment failed", error=str(e))
            return ExperimentResult(
                experiment_id=experiment_id,
                experiment_name=experiment_config.name,
                controller_type=controller_type,
                status=ExperimentStatus.FAILED,
                started_at=datetime.now(timezone.utc),
                error_message=str(e),
            )
        finally:
            self._runner.close()

        # Calculate final metrics
        avg_waiting_time = float(np.mean(waiting_times)) if waiting_times else 0.0
        avg_queue_length = float(np.mean(queue_lengths)) if queue_lengths else 0.0
        total_throughput = throughput
        stops_per_veh = 0.0  # Would need detailed vehicle tracking

        result = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=experiment_config.name,
            controller_type=controller_type,
            status=ExperimentStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            simulation_steps=step + 1,
            avg_waiting_time=avg_waiting_time,
            avg_queue_length=avg_queue_length,
            throughput_veh_per_hour=total_throughput * (3600 / max(1, step * step_length)),
            stops_per_vehicle=stops_per_veh,
            phase_changes=phase_changes,
            metrics={
                "waiting_times": waiting_times[-100:] if len(waiting_times) > 100 else waiting_times,
                "queue_lengths": queue_lengths[-100:] if len(queue_lengths) > 100 else queue_lengths,
            },
        )

        logger.info(
            "Experiment completed",
            experiment_id=str(experiment_id),
            avg_waiting_time=avg_waiting_time,
            avg_queue_length=avg_queue_length,
            throughput=total_throughput,
        )

        return result

    def run_comparison(
        self,
        experiment_config: ExperimentConfig,
        controllers: list[str] | None = None,
    ) -> dict[str, ExperimentResult]:
        """Run comparison experiment across multiple controllers.

        Args:
            experiment_config: Experiment configuration
            controllers: List of controller types to compare

        Returns:
            Dict mapping controller type to results
        """
        controllers = controllers or experiment_config.controllers
        results = {}

        for controller_type in controllers:
            logger.info("Running controller", controller=controller_type)
            result = self.run_single_experiment(experiment_config, controller_type)
            results[controller_type] = result

            # Brief pause between experiments
            time.sleep(1)

        # Log comparison summary
        self._log_comparison(results)

        return results

    def _log_comparison(self, results: dict[str, ExperimentResult]) -> None:
        """Log comparison summary."""
        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPARISON RESULTS")
        logger.info("=" * 60)

        for controller, result in results.items():
            logger.info(
                f"Controller: {controller}",
                waiting_time=f"{result.avg_waiting_time:.2f}s",
                queue_length=f"{result.avg_queue_length:.2f}m",
                throughput=f"{result.throughput_veh_per_hour:.0f} veh/h",
                phase_changes=result.phase_changes,
                status=result.status.value,
            )

        # Calculate improvement
        if "fixed_time" in results and "max_pressure" in results:
            ft = results["fixed_time"]
            mp = results["max_pressure"]
            if ft.avg_waiting_time > 0:
                improvement = (ft.avg_waiting_time - mp.avg_waiting_time) / ft.avg_waiting_time * 100
                logger.info(
                    "IMPROVEMENT",
                    waiting_time_reduction_pct=f"{improvement:.1f}%",
                    target_met=improvement >= 15.0,
                )


def create_default_experiment_config(
    name: str = "baseline_comparison",
    intersection_id: str = "intersection-01",
) -> ExperimentConfig:
    """Create default experiment configuration."""
    return ExperimentConfig(
        name=name,
        intersection_id=intersection_id,
        scenario_config={
            "demand": "peak",
            "north_south_volume": 800,
            "east_west_volume": 600,
            "turning_ratios": {"left": 0.1, "through": 0.8, "right": 0.1},
        },
        controllers=["fixed_time", "max_pressure"],
        simulation_steps=3600,  # 1 hour at 1s steps
        random_seed=42,
    )


def create_default_intersection_config() -> IntersectionConfig:
    """Create default intersection configuration matching SUMO network."""
    return IntersectionConfig(
        intersection_id="intersection-01",
        name="Main St & 1st Ave",
        approaches=[
            ApproachConfig(
                approach_id="northbound",
                name="Northbound Main St",
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
                name="Southbound Main St",
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
                name="Eastbound 1st Ave",
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
                name="Westbound 1st Ave",
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


# Import needed schemas
from traffic_domain.schemas import ApproachConfig, LaneConfig


def run_experiment(
    experiment_name: str = "baseline_comparison",
    controllers: list[str] | None = None,
    steps: int = 3600,
    output_file: str | None = None,
    sumo_gui: bool = False,
) -> dict[str, ExperimentResult]:
    """Run a traffic signal control experiment.

    Args:
        experiment_name: Name of the experiment
        controllers: List of controller types to test
        steps: Number of simulation steps
        output_file: Optional JSON output file for results
        sumo_gui: Whether to run SUMO with GUI

    Returns:
        Dict mapping controller type to results
    """
    settings = get_settings()

    if sumo_gui:
        import os
        os.environ["SUMO_GUI"] = "1"

    # Configure logging
    obs_cfg = settings.observability
    configure_logging(
        level=obs_cfg.log_level,
        json_output=obs_cfg.log_json,
        service_name="control_service",
    )

    # Create configurations
    intersection_config = create_default_intersection_config()
    experiment_config = create_default_experiment_config(experiment_name)
    experiment_config.simulation_steps = steps
    if controllers:
        experiment_config.controllers = controllers

    # Run experiment
    runner = ExperimentRunner(intersection_config, settings)
    results = runner.run_comparison(experiment_config)

    # Save results
    if output_file:
        output_data = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intersection_id": intersection_config.intersection_id,
            "controllers": {k: v.model_dump(mode="json") for k, v in results.items()},
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info("Results saved", file=output_file)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Traffic Control Experiment Runner")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="baseline_comparison",
        help="Experiment name",
    )
    parser.add_argument(
        "-c", "--controllers",
        nargs="+",
        choices=["fixed_time", "max_pressure"],
        default=["fixed_time", "max_pressure"],
        help="Controllers to test",
    )
    parser.add_argument(
        "-s", "--steps",
        type=int,
        default=3600,
        help="Simulation steps (seconds)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run SUMO with GUI",
    )
    parser.add_argument(
        "--sumo-home",
        help="Path to SUMO installation",
    )

    args = parser.parse_args()

    if args.sumo_home:
        import os
        os.environ["SUMO_HOME"] = args.sumo_home

    try:
        results = run_experiment(
            experiment_name=args.experiment,
            controllers=args.controllers,
            steps=args.steps,
            output_file=args.output,
            sumo_gui=args.gui,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 60)
        for controller, result in results.items():
            print(f"\n{controller.upper()}:")
            print(f"  Avg Waiting Time: {result.avg_waiting_time:.2f}s")
            print(f"  Avg Queue Length: {result.avg_queue_length:.2f}m")
            print(f"  Throughput: {result.throughput_veh_per_hour:.0f} veh/h")
            print(f"  Phase Changes: {result.phase_changes}")
            print(f"  Status: {result.status.value}")

        # Check improvement target
        if "fixed_time" in results and "max_pressure" in results:
            ft = results["fixed_time"]
            mp = results["max_pressure"]
            if ft.avg_waiting_time > 0:
                improvement = (ft.avg_waiting_time - mp.avg_waiting_time) / ft.avg_waiting_time * 100
                print(f"\nMax-Pressure vs Fixed-Time: {improvement:.1f}% waiting time reduction")
                if improvement >= 15.0:
                    print("✓ TARGET MET: >= 15% reduction achieved!")
                else:
                    print(f"✗ TARGET MISSED: Only {improvement:.1f}% reduction (need >= 15%)")

        return 0

    except Exception as e:
        logger.exception("Experiment failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())