-- ClickHouse initialization for Adaptive Traffic Flow Lab
-- Run as default user in ClickHouse

-- Create database
CREATE DATABASE IF NOT EXISTS traffic_lab;

-- Traffic state measurements (MergeTree for analytical queries)
CREATE TABLE IF NOT EXISTS traffic_lab.traffic_state (
    time DateTime64(3),
    intersection_id String,
    approach_id String,
    lane_id String,
    vehicle_count UInt32,
    queue_length_m Float64,
    occupancy Float64,
    avg_speed_kmh Float64,
    pedestrian_count UInt32,
    vehicle_types Map(String, UInt32),
    perception_latency_ms Float64,
    tracking_quality Float64,
    source String,
    camera_ids Array(String)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (intersection_id, approach_id, time)
TTL time + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- Controller recommendations
CREATE TABLE IF NOT EXISTS traffic_lab.controller_recommendation (
    time DateTime64(3),
    intersection_id String,
    current_phase String,
    action String,
    duration_seconds Float64,
    reason String,
    controller_type String,
    simulation_only UInt8,
    confidence Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (intersection_id, time)
TTL time + INTERVAL 90 DAY;

-- Phase changes
CREATE TABLE IF NOT EXISTS traffic_lab.phase_change (
    time DateTime64(3),
    intersection_id String,
    from_phase String,
    to_phase String,
    duration_seconds Float64,
    triggered_by String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (intersection_id, time)
TTL time + INTERVAL 90 DAY;

-- Aggregated metrics for dashboards (pre-computed rollups)
CREATE TABLE IF NOT EXISTS traffic_lab.traffic_state_1min (
    time DateTime,
    intersection_id String,
    approach_id String,
    avg_vehicle_count Float64,
    max_queue_length_m Float64,
    avg_occupancy Float64,
    avg_speed_kmh Float64,
    total_pedestrians UInt32,
    sample_count UInt32
) ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (intersection_id, approach_id, time);

-- 5-minute rollup
CREATE TABLE IF NOT EXISTS traffic_lab.traffic_state_5min (
    time DateTime,
    intersection_id String,
    approach_id String,
    avg_vehicle_count Float64,
    max_queue_length_m Float64,
    avg_occupancy Float64,
    avg_speed_kmh Float64,
    total_pedestrians UInt32,
    sample_count UInt32
) ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (intersection_id, approach_id, time);

-- Hourly rollup
CREATE TABLE IF NOT EXISTS traffic_lab.traffic_state_1hour (
    time DateTime,
    intersection_id String,
    approach_id String,
    avg_vehicle_count Float64,
    max_queue_length_m Float64,
    avg_occupancy Float64,
    avg_speed_kmh Float64,
    total_pedestrians UInt32,
    sample_count UInt32
) ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (intersection_id, approach_id, time);

-- Materialized views for automatic rollups
CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_lab.mv_traffic_state_1min
TO traffic_lab.traffic_state_1min
AS SELECT
    toStartOfMinute(time) AS time,
    intersection_id,
    approach_id,
    avg(vehicle_count) AS avg_vehicle_count,
    max(queue_length_m) AS max_queue_length_m,
    avg(occupancy) AS avg_occupancy,
    avg(avg_speed_kmh) AS avg_speed_kmh,
    sum(pedestrian_count) AS total_pedestrians,
    count() AS sample_count
FROM traffic_lab.traffic_state
GROUP BY time, intersection_id, approach_id;

CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_lab.mv_traffic_state_5min
TO traffic_lab.traffic_state_5min
AS SELECT
    toStartOfFiveMinute(time) AS time,
    intersection_id,
    approach_id,
    avg(vehicle_count) AS avg_vehicle_count,
    max(queue_length_m) AS max_queue_length_m,
    avg(occupancy) AS avg_occupancy,
    avg(avg_speed_kmh) AS avg_speed_kmh,
    sum(pedestrian_count) AS total_pedestrians,
    count() AS sample_count
FROM traffic_lab.traffic_state
GROUP BY time, intersection_id, approach_id;

CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_lab.mv_traffic_state_1hour
TO traffic_lab.traffic_state_1hour
AS SELECT
    toStartOfHour(time) AS time,
    intersection_id,
    approach_id,
    avg(vehicle_count) AS avg_vehicle_count,
    max(queue_length_m) AS max_queue_length_m,
    avg(occupancy) AS avg_occupancy,
    avg(avg_speed_kmh) AS avg_speed_kmh,
    sum(pedestrian_count) AS total_pedestrians,
    count() AS sample_count
FROM traffic_lab.traffic_state
GROUP BY time, intersection_id, approach_id;

-- Experiment results
CREATE TABLE IF NOT EXISTS traffic_lab.experiment_result (
    experiment_id String,
    experiment_name String,
    controller_type String,
    status String,
    started_at DateTime64(3),
    completed_at DateTime64(3),
    avg_waiting_time Float64,
    max_queue_length Float64,
    throughput_veh_per_hour Float64,
    avg_stops_per_veh Float64,
    avg_travel_time Float64,
    total_emissions Float64,
    approach_metrics String,  -- JSON
    time_series String,       -- JSON
    error String
) ENGINE = MergeTree()
ORDER BY (controller_type, started_at)
TTL started_at + INTERVAL 365 DAY;

-- Grant permissions
GRANT SELECT, INSERT ON traffic_lab.* TO traffic;