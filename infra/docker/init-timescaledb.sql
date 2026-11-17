-- TimescaleDB initialization for Adaptive Traffic Flow Lab
-- Run as postgres superuser (default in timescaledb image)

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schema for traffic data
CREATE SCHEMA IF NOT EXISTS traffic;

-- Traffic state measurements hypertable
CREATE TABLE IF NOT EXISTS traffic.traffic_state (
    time        TIMESTAMPTZ       NOT NULL,
    intersection_id   TEXT        NOT NULL,
    approach_id       TEXT        NOT NULL,
    lane_id           TEXT,
    vehicle_count     INT         DEFAULT 0,
    queue_length_m    DOUBLE PRECISION DEFAULT 0,
    occupancy         DOUBLE PRECISION DEFAULT 0,
    avg_speed_kmh     DOUBLE PRECISION DEFAULT 0,
    pedestrian_count  INT         DEFAULT 0,
    vehicle_types     JSONB       DEFAULT '{}',
    perception_latency_ms DOUBLE PRECISION,
    tracking_quality  DOUBLE PRECISION,
    source            TEXT        DEFAULT 'camera',
    camera_ids        TEXT[]      DEFAULT '{}'
);

SELECT create_hypertable('traffic.traffic_state', 'time', if_not_exists => TRUE);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_traffic_state_intersection_time
    ON traffic.traffic_state (intersection_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_traffic_state_approach_time
    ON traffic.traffic_state (intersection_id, approach_id, time DESC);

-- Controller recommendations
CREATE TABLE IF NOT EXISTS traffic.controller_recommendation (
    time        TIMESTAMPTZ       NOT NULL,
    intersection_id   TEXT        NOT NULL,
    current_phase     TEXT        NOT NULL,
    action            TEXT        NOT NULL,
    duration_seconds  DOUBLE PRECISION NOT NULL,
    reason            TEXT,
    controller_type   TEXT        NOT NULL,
    simulation_only   BOOLEAN     DEFAULT TRUE,
    confidence        DOUBLE PRECISION DEFAULT 1.0
);

SELECT create_hypertable('traffic.controller_recommendation', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ctrl_rec_intersection_time
    ON traffic.controller_recommendation (intersection_id, time DESC);

-- Phase changes
CREATE TABLE IF NOT EXISTS traffic.phase_change (
    time        TIMESTAMPTZ       NOT NULL,
    intersection_id   TEXT        NOT NULL,
    from_phase        TEXT,
    to_phase          TEXT        NOT NULL,
    duration_seconds  DOUBLE PRECISION,
    triggered_by      TEXT        DEFAULT 'controller'
);

SELECT create_hypertable('traffic.phase_change', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_phase_change_intersection_time
    ON traffic.phase_change (intersection_id, time DESC);

-- Incidents
CREATE TABLE IF NOT EXISTS traffic.incident (
    time        TIMESTAMPTZ       NOT NULL,
    intersection_id   TEXT        NOT NULL,
    incident_type     TEXT        NOT NULL,
    description       TEXT,
    severity          TEXT        DEFAULT 'medium',
    resolved          BOOLEAN     DEFAULT FALSE,
    resolved_at       TIMESTAMPTZ
);

SELECT create_hypertable('traffic.incident', 'time', if_not_exists => TRUE);

-- Experiment results
CREATE TABLE IF NOT EXISTS traffic.experiment_result (
    experiment_id     UUID        NOT NULL,
    experiment_name   TEXT        NOT NULL,
    controller_type   TEXT        NOT NULL,
    status            TEXT        NOT NULL,
    started_at        TIMESTAMPTZ NOT NULL,
    completed_at      TIMESTAMPTZ,
    avg_waiting_time  DOUBLE PRECISION,
    max_queue_length  DOUBLE PRECISION,
    throughput_veh_per_hour DOUBLE PRECISION,
    avg_stops_per_veh DOUBLE PRECISION,
    avg_travel_time   DOUBLE PRECISION,
    total_emissions   DOUBLE PRECISION,
    approach_metrics  JSONB,
    time_series       JSONB,
    error             TEXT
);

CREATE INDEX IF NOT EXISTS idx_exp_result_controller
    ON traffic.experiment_result (controller_type, started_at DESC);

-- Intersection configs (static data, not a hypertable)
CREATE TABLE IF NOT EXISTS traffic.intersection_config (
    intersection_id   TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    config_json       JSONB NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Camera calibrations
CREATE TABLE IF NOT EXISTS traffic.camera_calibration (
    camera_id         TEXT PRIMARY KEY,
    intersection_id   TEXT REFERENCES traffic.intersection_config(intersection_id),
    calibration_json  JSONB NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Grant permissions
GRANT USAGE ON SCHEMA traffic TO traffic;
GRANT ALL ON ALL TABLES IN SCHEMA traffic TO traffic;
GRANT ALL ON ALL SEQUENCES IN SCHEMA traffic TO traffic;
ALTER DEFAULT PRIVILEGES IN SCHEMA traffic GRANT ALL ON TABLES TO traffic;