"""Reinforcement Learning controller for traffic signals."""

from __future__ import annotations

import numpy as np
import pickle
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traffic_domain.schemas import (
    TrafficState,
    ControllerRecommendation,
    ControllerAction,
    SignalPhase,
    IntersectionConfig,
    ApproachConfig,
    LaneConfig,
    ApproachState,
    LaneState,
)
from traffic_config.settings import ControlSettings
from traffic_control.base import BaseController
from traffic_observability.logging import get_logger
from traffic_observability.metrics import get_metrics_registry


logger = get_logger(__name__)


@dataclass
class RLTransition:
    """Single RL transition (s, a, r, s')."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict[str, Any] = None


class TrafficEnv:
    """Traffic signal control environment for RL."""

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        settings: ControlSettings,
        max_steps: int = 3600,
    ):
        self._config = intersection_config
        self._settings = settings
        self._max_steps = max_steps
        self._current_step = 0
        self._state: TrafficState | None = None
        self._phase_idx = 0
        self._phase_elapsed = 0.0

        # Action space: 0=HOLD, 1=SWITCH_NEXT, 2=SWITCH_SPECIFIC (with phase index)
        self._n_phases = len(intersection_config.phases)
        self._action_space = 2 + self._n_phases  # hold, next, specific phases

        # Observation space: per-approach [queue, wait, speed, occupancy, phase_onehot]
        self._n_approaches = len(intersection_config.approaches)
        self._obs_dim = self._n_approaches * 5 + self._n_phases

    def reset(self, initial_state: TrafficState | None = None) -> np.ndarray:
        """Reset environment."""
        self._current_step = 0
        self._phase_idx = 0
        self._phase_elapsed = 0.0
        self._state = initial_state or self._create_empty_state()
        return self._get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute action and return (obs, reward, done, info)."""
        # Apply action
        self._apply_action(action)

        # Simulate traffic for one step (1 second)
        self._simulate_step()

        # Calculate reward
        reward = self._calculate_reward()

        # Check done
        self._current_step += 1
        done = self._current_step >= self._max_steps

        # Get observation
        obs = self._get_observation()

        total_wait = sum(self._approach_wait_times.get(a.approach_id, 0.0) for a in self._state.approaches)
        info = {
            "step": self._current_step,
            "phase": self._config.phases[self._phase_idx].value,
            "total_queue": sum(a.queue_length_m for a in self._state.approaches),
            "total_wait": total_wait,
        }

        return obs, reward, done, info

    def _apply_action(self, action: int) -> None:
        """Apply action to change phase."""
        if action == 0:  # HOLD
            pass
        elif action == 1:  # SWITCH_NEXT
            self._phase_idx = (self._phase_idx + 1) % self._n_phases
            self._phase_elapsed = 0.0
        elif action >= 2:  # SWITCH_SPECIFIC
            phase_idx = action - 2
            if 0 <= phase_idx < self._n_phases:
                self._phase_idx = phase_idx
                self._phase_elapsed = 0.0

    def _simulate_step(self) -> None:
        """Simulate one time step of traffic."""
        # This is a simplified simulation
        # In practice, this would interface with SUMO or a traffic flow model
        dt = 1.0
        self._phase_elapsed += dt

        # Maintain internal wait times since ApproachState is frozen
        if not hasattr(self, '_approach_wait_times'):
            self._approach_wait_times = {a.approach_id: 0.0 for a in self._state.approaches}

        # Build config lookups
        lane_configs = {}
        approach_configs = {}
        for approach_config in self._config.approaches:
            approach_configs[approach_config.approach_id] = approach_config
            lane_configs[approach_config.approach_id] = {lc.lane_id: lc for lc in approach_config.lanes}

        new_approaches = []
        for approach in self._state.approaches:
            # Arrivals (simplified) - count non-turning lanes from config
            arrival_rate = 0.1  # veh/s/lane
            approach_lane_configs = lane_configs.get(approach.approach_id, {})
            n_lanes = len([lc for lc in approach_lane_configs.values() if not lc.is_turning])
            arrivals = arrival_rate * n_lanes * dt

            # Departures if green
            current_phase = self._config.phases[self._phase_idx]
            approach_config = approach_configs.get(approach.approach_id)
            if approach_config and approach_config.signal_group in current_phase.value:
                saturation = 0.5  # veh/s/lane
                departures = saturation * n_lanes * dt
            else:
                departures = 0

            # Update queue (vehicles)
            queue_veh = approach.queue_length_m / 7.5
            queue_veh = max(0, queue_veh + arrivals - departures)
            new_queue_m = queue_veh * 7.5

            # Update wait time
            wait_time = self._approach_wait_times.get(approach.approach_id, 0.0)
            if new_queue_m > 0:
                wait_time += dt
            else:
                wait_time = max(0, wait_time - dt)
            self._approach_wait_times[approach.approach_id] = wait_time

            # Create new lane states
            new_lanes = {}
            for lane_id, lane in approach.lanes.items():
                new_lanes[lane_id] = LaneState(
                    lane_id=lane_id,
                    vehicle_count=lane.vehicle_count,
                    queue_length_m=new_queue_m / max(len(approach.lanes), 1),
                    occupancy=lane.occupancy,
                    average_speed_kmh=lane.average_speed_kmh,
                    measured_at=lane.measured_at,
                )

            # Create new approach state
            new_approaches.append(ApproachState(
                approach_id=approach.approach_id,
                vehicle_count=approach.vehicle_count + int(arrivals - departures),
                queue_length_m=new_queue_m,
                average_speed_kmh=approach.average_speed_kmh,
                occupancy=np.mean([l.occupancy for l in new_lanes.values()]) if new_lanes else 0.0,
                lanes=new_lanes,
                measured_at=approach.measured_at,
            ))

        self._state = TrafficState(
            intersection_id=self._state.intersection_id,
            observed_at=self._state.observed_at,
            approaches=new_approaches,
        )

    def _calculate_reward(self) -> float:
        """Calculate reward (negative of total delay)."""
        total_wait = sum(self._approach_wait_times.get(a.approach_id, 0.0) for a in self._state.approaches)
        total_queue = sum(a.queue_length_m for a in self._state.approaches)

        # Reward = negative weighted sum
        reward = -(total_wait * 0.1 + total_queue * 0.01)

        # Bonus for throughput
        throughput = sum(a.vehicle_count for a in self._state.approaches)
        reward += throughput * 0.01

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = []

        # Per-approach features
        for approach in self._state.approaches:
            # Estimate avg wait time from queue length and speed
            avg_wait = approach.queue_length_m / max(approach.average_speed_kmh / 3.6, 0.1)
            # Estimate occupancy from lanes
            avg_occupancy = np.mean([l.occupancy for l in approach.lanes.values()]) if approach.lanes else 0.0
            obs.extend([
                approach.queue_length_m / 100.0,  # Normalized queue
                avg_wait / 60.0,                  # Normalized wait
                approach.average_speed_kmh / 50.0,  # Normalized speed
                avg_occupancy,                     # Occupancy [0,1]
                approach.vehicle_count / 50.0,     # Normalized count
            ])

        # Current phase one-hot
        phase_onehot = np.zeros(self._n_phases)
        phase_onehot[self._phase_idx] = 1.0
        obs.extend(phase_onehot)

        return np.array(obs, dtype=np.float32)

    def _create_empty_state(self) -> TrafficState:
        """Create empty traffic state."""
        from traffic_domain.schemas import ApproachState, LaneState
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        approaches = []
        for approach_config in self._config.approaches:
            lanes = {}
            for lane_config in approach_config.lanes:
                lanes[lane_config.lane_id] = LaneState(
                    lane_id=lane_config.lane_id,
                    vehicle_count=0,
                    queue_length_m=0.0,
                    occupancy=0.0,
                    average_speed_kmh=0.0,
                    measured_at=now,
                )
            approaches.append(ApproachState(
                approach_id=approach_config.approach_id,
                vehicle_count=0,
                queue_length_m=0.0,
                average_speed_kmh=0.0,
                occupancy=0.0,
                lanes=lanes,
                measured_at=now,
            ))

        return TrafficState(
            intersection_id=self._config.intersection_id,
            observed_at=now,
            approaches=approaches,
        )


class RLAgent(ABC):
    """Abstract RL agent."""

    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action."""
        pass

    @abstractmethod
    def learn(self, transition: RLTransition) -> None:
        """Learn from transition."""
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save agent."""
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load agent."""
        pass


class QLearningAgent(RLAgent):
    """Tabular Q-learning agent (for small state spaces)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min

        # Discretize state space (simplified)
        self._q_table: dict[tuple, np.ndarray] = {}

    def _discretize(self, state: np.ndarray) -> tuple:
        """Discretize continuous state."""
        # Simple binning
        bins = np.linspace(-1, 2, 10)
        return tuple(np.digitize(state, bins))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        key = self._discretize(state)

        if key not in self._q_table:
            self._q_table[key] = np.zeros(self._action_dim)

        if training and np.random.random() < self._epsilon:
            return np.random.randint(self._action_dim)

        return int(np.argmax(self._q_table[key]))

    def learn(self, transition: RLTransition) -> None:
        """Q-learning update."""
        key = self._discretize(transition.state)
        next_key = self._discretize(transition.next_state)

        if key not in self._q_table:
            self._q_table[key] = np.zeros(self._action_dim)
        if next_key not in self._q_table:
            self._q_table[next_key] = np.zeros(self._action_dim)

        # Q-learning update
        td_target = transition.reward + self._gamma * np.max(self._q_table[next_key]) * (1 - transition.done)
        td_error = td_target - self._q_table[key][transition.action]
        self._q_table[key][transition.action] += self._alpha * td_error

        # Decay epsilon
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def save(self, path: str | Path) -> None:
        """Save Q-table."""
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": self._q_table,
                "epsilon": self._epsilon,
                "alpha": self._alpha,
                "gamma": self._gamma,
            }, f)

    def load(self, path: str | Path) -> None:
        """Load Q-table."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._q_table = data["q_table"]
            self._epsilon = data["epsilon"]
            self._alpha = data["alpha"]
            self._gamma = data["gamma"]


class DQNAgent(RLAgent):
    """Deep Q-Network agent (placeholder for PyTorch/TensorFlow implementation)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 1000,
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._batch_size = batch_size
        self._target_update = target_update
        self._step_count = 0

        # Replay buffer
        self._buffer: deque[RLTransition] = deque(maxlen=buffer_size)

        # Placeholder for neural network
        self._model = None
        self._target_model = None
        self._optimizer = None

        logger.warning("DQNAgent is a placeholder - implement with PyTorch/TensorFlow")

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action (placeholder)."""
        if training and np.random.random() < self._epsilon:
            return np.random.randint(self._action_dim)

        # Placeholder: return random action
        return np.random.randint(self._action_dim)

    def learn(self, transition: RLTransition) -> None:
        """Store transition and train (placeholder)."""
        self._buffer.append(transition)
        self._step_count += 1

        # Decay epsilon
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

        # Train periodically
        if self._step_count % self._target_update == 0:
            logger.info("DQN target network update (placeholder)")

    def save(self, path: str | Path) -> None:
        """Save model (placeholder)."""
        logger.warning("DQNAgent.save not implemented")

    def load(self, path: str | Path) -> None:
        """Load model (placeholder)."""
        logger.warning("DQNAgent.load not implemented")


class RLController(BaseController):
    """RL-based traffic signal controller."""

    def __init__(
        self,
        intersection_config: IntersectionConfig,
        settings: ControlSettings,
        agent: RLAgent | None = None,
        training: bool = False,
    ):
        super().__init__(intersection_config, settings)
        self._env = TrafficEnv(intersection_config, settings)
        self._agent = agent
        self._training = training
        self._current_obs: np.ndarray | None = None
        self._metrics = get_metrics_registry()

        # Create default agent if none provided
        if self._agent is None:
            self._agent = QLearningAgent(
                state_dim=self._env._obs_dim,
                action_dim=self._env._action_space,
            )

        # Initialize with empty state
        self._current_obs = self._env.reset()

    def compute_action(self, state: TrafficState) -> ControllerRecommendation:
        """Compute action using RL agent."""
        # Update environment with real state
        self._env._state = state

        # Get observation
        obs = self._env._get_observation()
        self._current_obs = obs

        # Select action
        action = self._agent.act(obs, training=self._training)

        # Map action to recommendation
        return self._action_to_recommendation(action)

    def _action_to_recommendation(self, action: int) -> ControllerRecommendation:
        """Map RL action to controller recommendation."""
        n_phases = len(self.config.phases)
        current_phase = self.current_phase or self.config.phases[0]
        from datetime import datetime, timezone

        if action == 0:  # HOLD
            return ControllerRecommendation(
                intersection_id=self.config.intersection_id,
                timestamp=datetime.now(timezone.utc),
                action=ControllerAction.HOLD_PHASE,
                current_phase=current_phase,
                duration_seconds=self.settings.min_green_time,
                confidence=0.7,
                reason="RL: HOLD",
                controller_type="rl",
            )
        elif action == 1:  # SWITCH_NEXT
            next_idx = (self.config.phases.index(current_phase) + 1) % n_phases
            next_phase = self.config.phases[next_idx]
            return ControllerRecommendation(
                intersection_id=self.config.intersection_id,
                timestamp=datetime.now(timezone.utc),
                action=ControllerAction.SWITCH_PHASE,
                current_phase=next_phase,
                duration_seconds=self.settings.min_green_time,
                confidence=0.8,
                reason="RL: SWITCH_NEXT",
                controller_type="rl",
            )
        else:  # SWITCH_SPECIFIC
            phase_idx = action - 2
            if 0 <= phase_idx < n_phases:
                target_phase = self.config.phases[phase_idx]
                return ControllerRecommendation(
                    intersection_id=self.config.intersection_id,
                    timestamp=datetime.now(timezone.utc),
                    action=ControllerAction.SWITCH_PHASE,
                    current_phase=target_phase,
                    duration_seconds=self.settings.min_green_time,
                    confidence=0.8,
                    reason=f"RL: SWITCH_TO_{target_phase.value}",
                    controller_type="rl",
                )

        # Fallback
        return ControllerRecommendation(
            intersection_id=self.config.intersection_id,
            timestamp=datetime.now(timezone.utc),
            action=ControllerAction.HOLD_PHASE,
            current_phase=current_phase,
            duration_seconds=self.settings.min_green_time,
            confidence=0.5,
            reason="RL: invalid action, holding",
            controller_type="rl",
        )

    def learn_from_transition(
        self,
        prev_state: TrafficState,
        action: int,
        reward: float,
        next_state: TrafficState,
        done: bool = False,
    ) -> None:
        """Learn from a transition (for online learning)."""
        if not self._training:
            return

        prev_obs = self._env._get_observation_from_state(prev_state)
        next_obs = self._env._get_observation_from_state(next_state)

        transition = RLTransition(
            state=prev_obs,
            action=action,
            reward=reward,
            next_state=next_obs,
            done=done,
        )

        self._agent.learn(transition)

        self._metrics.record_histogram(
            "rl_reward",
            reward,
            labels={"intersection": self._intersection_config.intersection_id},
        )

    def set_training(self, training: bool) -> None:
        """Set training mode."""
        self._training = training

    def save_agent(self, path: str | Path) -> None:
        """Save RL agent."""
        self._agent.save(path)
        logger.info("RL agent saved", path=path)

    def load_agent(self, path: str | Path) -> None:
        """Load RL agent."""
        self._agent.load(path)
        logger.info("RL agent loaded", path=path)


def create_rl_controller(
    intersection_config: IntersectionConfig,
    settings: ControlSettings,
    agent_type: str = "qlearning",
    training: bool = False,
    **kwargs,
) -> RLController:
    """Factory function to create RL controller."""
    if agent_type == "qlearning":
        agent = QLearningAgent(**kwargs)
    elif agent_type == "dqn":
        agent = DQNAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return RLController(intersection_config, settings, agent, training)


# Extension to TrafficEnv for RLController
def _get_observation_from_state(self, state: TrafficState) -> np.ndarray:
    """Get observation from a specific state."""
    obs = []
    for approach in state.approaches:
        obs.extend([
            approach.queue_length_m / 100.0,
            approach.avg_wait_time_s / 60.0,
            approach.avg_speed_kmh / 50.0,
            approach.occupancy,
            approach.vehicle_count / 50.0,
        ])
    phase_onehot = np.zeros(self._n_phases)
    phase_onehot[self._phase_idx] = 1.0
    obs.extend(phase_onehot)
    return np.array(obs, dtype=np.float32)


TrafficEnv._get_observation_from_state = _get_observation_from_state