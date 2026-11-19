import numpy as np


def velocity_metrics(true, pred):
    error = np.asarray(pred)-np.asarray(true)
    if error.ndim != 2 or error.shape[1] != 3 or not np.isfinite(error).all():
        raise ValueError('Expected finite Nx3 errors')
    rmse, mae = np.sqrt(np.mean(error**2, axis=0)), np.mean(abs(error), axis=0)
    return {'rmse_mean': float(rmse.mean()), 'mae_mean': float(mae.mean()),
            **{f'rmse_{a}':float(v) for a,v in zip('xyz',rmse)},
            **{f'mae_{a}':float(v) for a,v in zip('xyz',mae)}}


def trajectory_metrics(time, reference, estimate, horizon_s=1.):
    error = estimate-reference
    distance = np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()
    j = np.searchsorted(time, time+horizon_s)
    i = np.flatnonzero(j < len(time))
    # Translation RPE in the fixed reference world frame, not SE(3) pose RPE.
    relative = error[j[i]]-error[i]
    return {'position_rmse_m': float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
            'final_error_m': float(np.linalg.norm(error[-1])),
            'path_length_m': float(distance),
            'final_drift_percent': float(100*np.linalg.norm(error[-1])/distance) if distance>1e-6 else None,
            'translation_rpe_1s_m': float(np.sqrt(np.mean(np.sum(relative**2,axis=1)))) if len(i) else None}
