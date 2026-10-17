import math

import numpy as np

from dog3588.observation import ObservationBuilder
from dog3588.types import ImuState, JointState


def state(value=0):
    return JointState(np.full(12, value, dtype=float), np.full(12, value * 2, dtype=float), np.ones(12))


def test_term_major_history_and_reset_repeat_first_sample(manifest):
    b = ObservationBuilder(manifest)
    imu = ImuState(np.array([1., 2., 3.]), np.array([0., 0., -1.]), 1.)
    gait = np.array([1.5, .575, .02, 0, .5, .5, 0])
    obs = b.build(state(1), imu, np.zeros(3), gait)
    assert obs.shape == (280,) and obs.dtype == np.float32
    np.testing.assert_array_equal(obs[:15], [1, 2, 3] * 5)
    np.testing.assert_allclose(obs[105:165], .1)  # joint velocity scale 0.05
    np.testing.assert_array_equal(obs[165:225], 0)
    np.testing.assert_array_equal(obs[260:], 0)  # no gait clock at rest
    b.advance(np.arange(12), np.zeros(3), gait)
    obs = b.build(state(2), ImuState(np.array([4., 5., 6.]), imu.gravity, 2.), np.zeros(3), gait)
    np.testing.assert_array_equal(obs[:15], [1, 2, 3] * 4 + [4, 5, 6])
    np.testing.assert_array_equal(obs[45:105], [1] * 48 + [2] * 12)
    np.testing.assert_array_equal(obs[165:225], [0] * 48 + list(range(12)))
    b.reset()
    obs = b.build(state(2), imu, np.zeros(3), gait)
    np.testing.assert_array_equal(obs[45:105], 2)


def test_gait_clock_duty_warp_and_stop_freeze(manifest):
    b = ObservationBuilder(manifest)
    gait = np.array([1.5, .6, .02, 0, .5, .5, 0])
    velocity = np.array([.1, 0, 0])
    b.advance(np.zeros(12), velocity, gait)
    assert math.isclose(b.phase, .03)
    obs = b.build(state(), ImuState(np.zeros(3), np.array([0, 0, -1]), 1.), velocity, gait)
    expected = [math.sin(math.pi * p / .6) for p in (.03, .53, .53, .03)]
    np.testing.assert_allclose(obs[-4:], expected, atol=1e-6)
    b.advance(np.zeros(12), np.zeros(3), gait)
    assert math.isclose(b.phase, .03)

