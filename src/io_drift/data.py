"""Reference-aligned EuRoC loading. Only gyro/accel/dt are model inputs."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Sequence:
    name: str
    time: np.ndarray
    imu: np.ndarray  # gyro xyz, specific force xyz, dt seconds
    position: np.ndarray
    velocity: np.ndarray  # reference world frame, labels only
    quaternion: np.ndarray  # scipy xyzw, labels/one-time initialization only
    body_velocity: np.ndarray


def validate_splits(config):
    groups = [config[k] for k in ('train', 'validation', 'test')]
    if any(not g or len(set(g)) != len(g) for g in groups):
        raise ValueError('Splits must be nonempty with no duplicates')
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Sequence leakage between splits')


def load_sequence(root, name, max_gap_s=.025):
    base = Path(root) / name / 'mav0'
    imu = pd.read_csv(base / 'imu0/data.csv')
    gt = pd.read_csv(base / 'state_groundtruth_estimate0/data.csv')
    ti, tg = imu.iloc[:, 0].to_numpy(np.int64), gt.iloc[:, 0].to_numpy(np.int64)
    if len(ti) < 2 or len(tg) < 2 or (np.diff(ti) <= 0).any() or (np.diff(tg) <= 0).any():
        raise ValueError(f'{name}: invalid timestamps')
    # Integer subtraction before float conversion preserves nanosecond precision.
    origin = min(ti[0], tg[0])
    ti, tg = (ti-origin)/1e9, (tg-origin)/1e9
    valid = (ti >= tg[0]) & (ti <= tg[-1])
    # Never interpolate labels across a large gap in reference coverage.
    right = np.searchsorted(tg, ti, side='left').clip(1, len(tg)-1)
    exact = np.isin(ti, tg)
    valid &= exact | ((tg[right] - tg[right-1]) <= max_gap_s)
    t = ti[valid]
    if len(t) < 2:
        raise ValueError(f'{name}: no supported reference coverage')
    raw = imu.iloc[:, 1:7].to_numpy(np.float64)[valid]
    reference = gt.iloc[:, 1:].to_numpy(np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(reference).all():
        raise ValueError(f'{name}: nonfinite data')
    q = reference[:, [4,5,6,3]]  # original wxyz -> xyzw
    if not np.allclose(np.linalg.norm(q, axis=1), 1., atol=.01):
        raise ValueError(f'{name}: invalid quaternions')
    rot = Slerp(tg, Rotation.from_quat(q))(t)
    interp = lambda array: np.column_stack([np.interp(t, tg, array[:, i]) for i in range(3)])
    position, velocity = interp(reference[:, :3]), interp(reference[:, 7:10])
    features = np.column_stack([raw, np.r_[0., np.diff(t)]])
    return Sequence(name, t, features, position, velocity, rot.as_quat(), rot.inv().apply(velocity))


def windows(sequence, window=200, stride=20, sample_step=2, max_gap_s=.025):
    if window < 2 or stride < 1 or sample_step < 1:
        raise ValueError('Invalid window configuration')
    t = sequence.time
    boundaries = np.r_[0, np.flatnonzero(np.diff(t) > max_gap_s)+1, len(t)]
    starts = np.concatenate([np.arange(a, b-window+1, stride, dtype=int)
                             for a,b in zip(boundaries[:-1], boundaries[1:])])
    if not len(starts):
        raise ValueError(f'{sequence.name}: no full windows')
    # Sample backwards from endpoint so the newest sample is always included.
    offsets = np.arange(window-1, -1, -sample_step)[::-1]
    endpoints = starts + window-1
    x = sequence.imu[starts[:, None]+offsets].astype(np.float32)
    y = sequence.body_velocity[endpoints].astype(np.float32)
    return x, y, endpoints
