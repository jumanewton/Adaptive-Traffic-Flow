# Adaptive Traffic Flow Lab

An open-source research and engineering project for understanding and improving
urban traffic flow with computer vision, traffic simulation, and adaptive
signal-control experiments.

The project starts safely with recorded video and simulation. It should not
control public traffic infrastructure until the algorithms have been validated
in simulation, hardware-in-the-loop tests, and an appropriately authorized
pilot.

## Project goals

- Detect and track vehicles and pedestrians from video.
- Estimate traffic state: queues, occupancy, speed, turning movements, and
  demand.
- Compare fixed-time and adaptive traffic-signal strategies.
- Stream traffic events for realtime analytics.
- Measure improvements using reproducible simulation experiments.
- Build a path toward realtime deployment without coupling perception directly
  to signal control.

## First milestone

Given a traffic video and a simulated intersection:

1. Detect and track road users with YOLO and ByteTrack.
2. Convert tracks into lane-level traffic-state measurements.
3. Publish those measurements as structured events.
4. Feed the measurements into a SUMO simulation.
5. Compare a fixed-time controller with a max-pressure controller.
6. Report waiting time, queue length, throughput, stops, and fairness.

The initial success target is a 15% reduction in average waiting time against a
fixed-time baseline while preserving signal, pedestrian, and phase-safety
constraints.

## Architecture

```text
Video / sensors
      |
      v
Perception: YOLO + ByteTrack
      |
      v
Traffic state estimation
      |
      +--------------------> Event stream
      |                         |
      v                         v
Traffic-control policy       Analytics storage
      |                         |
      v                         v
SUMO / authorized controller  Dashboard and reports
```

The perception layer answers **what is happening**. The control layer answers
**what should happen next**. Keeping those concerns separate allows models to
change without changing safety-critical control logic.

## Repository structure

```text
.
├── apps/
│   ├── vision_worker/       # Video ingestion, inference, and tracking
│   ├── state_service/       # Track-to-traffic-state aggregation
│   ├── control_service/     # Controller APIs and policy execution
│   └── api/                 # HTTP API for experiments and results
├── packages/
│   ├── domain/              # Shared entities, schemas, and event contracts
│   ├── vision/              # Detection, calibration, and model adapters
│   ├── tracking/            # ByteTrack and tracker interfaces
│   ├── traffic/             # Lanes, queues, phases, and traffic metrics
│   ├── control/             # Fixed-time, max-pressure, MPC, and RL adapters
│   ├── simulation/          # Simulator-independent interfaces
│   ├── streaming/           # Kafka-compatible producers and consumers
│   ├── storage/             # Database and object-storage adapters
│   └── observability/       # Metrics, logs, traces, and experiment reports
├── simulation/
│   └── sumo/                # SUMO networks, routes, scenarios, and runners
├── configs/                 # Environment and experiment configuration
├── data/
│   ├── raw/                 # Original videos and sensor exports
│   ├── annotations/         # CVAT exports and label definitions
│   └── processed/           # Derived clips, frames, and measurements
├── models/
│   ├── checkpoints/          # Training checkpoints; do not commit large files
│   └── exports/             # ONNX/TensorRT or serving-ready models
├── notebooks/               # Exploration and result analysis
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── infra/
│   ├── docker/              # Local services and compose files
│   └── kubernetes/          # Optional later deployment manifests
└── scripts/                 # Reproducible setup, training, and evaluation commands
```

## Submodule responsibilities

### `packages/domain`

Owns stable contracts shared by all services. Examples include `Detection`,
`Track`, `TrafficState`, `SignalPhase`, `TrafficEvent`, and `ExperimentResult`.
This package must not depend on OpenCV, Kafka, SUMO, or a particular database.

### `packages/vision`

Contains model loading, preprocessing, camera calibration, coordinate
transforms, and inference interfaces. It should expose detections rather than
traffic-control decisions.

### `packages/tracking`

Provides a tracker interface and ByteTrack implementation. It converts frame
detections into persistent object tracks and handles track lifecycle events.

### `packages/traffic`

Maps tracks to lanes and regions of interest, estimates queues and speeds, and
calculates traffic metrics. This is where camera geometry becomes traffic
meaning.

### `packages/control`

Contains interchangeable control policies:

- Fixed-time baseline
- Rule-based adaptive control
- Max-pressure control
- Model Predictive Control
- Reinforcement-learning experiments

Every policy should consume the same `TrafficState` input and produce a
validated signal recommendation. Safety and phase constraints belong here,
outside the ML model.

### `packages/simulation`

Defines a simulator-neutral interface so controllers can run against SUMO
without importing SUMO-specific code throughout the project.

### `packages/streaming`

Publishes and consumes versioned traffic events. Apache Kafka can be the first
backend; another Kafka-compatible backend can be added behind the same
interface.

### `packages/storage`

Stores raw artifacts, measurements, experiment results, and model metadata.
Keep object storage, analytical databases, and relational metadata behind
adapters.

### `apps/*`

Applications compose packages into deployable processes. Applications may
depend on packages; packages should not import application code.

## Data contracts

Example traffic-state event:

```json
{
  "schema_version": 1,
  "intersection_id": "intersection-01",
  "observed_at": "2026-01-01T12:00:00Z",
  "approaches": [
    {
      "approach_id": "northbound",
      "vehicle_count": 18,
      "queue_length_m": 74.0,
      "average_speed_kmh": 8.0,
      "pedestrian_count": 3
    }
  ]
}
```

Example controller recommendation:

```json
{
  "schema_version": 1,
  "intersection_id": "intersection-01",
  "current_phase": "north_south_green",
  "action": "extend_green",
  "duration_seconds": 8,
  "reason": "high_northbound_pressure",
  "simulation_only": true
}
```

Contracts should be versioned. A perception failure must result in an explicit
unknown or stale state, not a success-shaped default.

## Development roadmap

### Phase 0: Foundation

- Define Python package and service conventions.
- Add configuration and structured logging.
- Add domain schemas and contract tests.

### Phase 1: Perception

- Run inference on a recorded video.
- Add ByteTrack.
- Produce an annotated video and track export.
- Evaluate detection and tracking quality.

### Phase 2: Traffic state

- Define lanes and regions of interest.
- Estimate queue length, speed, occupancy, and line crossings.
- Validate measurements against manually reviewed clips.

### Phase 3: Simulation and control

- Create one SUMO intersection.
- Implement a fixed-time baseline.
- Implement max-pressure control.
- Compare reproducible experiments.

### Phase 4: Streaming and analytics

- Publish versioned traffic-state events.
- Store results in an analytical database.
- Add a dashboard for queues, delays, and controller decisions.

### Phase 5: Realtime deployment

- Export the model to a serving format.
- Add Triton or BentoML behind the inference interface.
- Add latency, dropped-frame, and stale-data monitoring.
- Test multiple connected intersections.

### Phase 6: Advanced research

- Model Predictive Control.
- Multi-intersection coordination.
- Emergency-vehicle priority.
- Emissions and fuel-consumption estimates.
- Reinforcement learning, evaluated only against safe baselines in simulation.

## Safety and ethics

- Use synthetic or properly authorized video.
- Minimize retention of identifiable imagery.
- Avoid face recognition and unnecessary identity tracking.
- Keep camera-derived data separate from personally identifying information.
- Treat controllers as advisory until independently validated.
- Enforce yellow, all-red, pedestrian, emergency, and minimum/maximum-green
  constraints.
- Provide a deterministic fallback when data is missing or stale.

## Licensing

Review the license of every dependency before distribution or commercial use.
In particular, verify the terms of the selected YOLO implementation, tracker,
streaming platform, model-serving runtime, and datasets. Keep third-party
notices with released artifacts.

## Current status

Project scaffolding and architecture definition. The next implementation task
is to add the domain schemas and a reproducible single-video perception
pipeline.
