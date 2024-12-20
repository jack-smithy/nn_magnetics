import numpy as np
import torch

from nn_magnetics.utils.metrics import calculate_metrics, calculate_metrics_baseline


def validate(X_test, B_test, model):
    model.eval()

    avg_angle_errors_baseline = []
    avg_amp_errors_baseline = []
    avg_angle_errors = []
    avg_amp_errors = []

    with torch.no_grad():
        for X, B in zip(torch.from_numpy(X_test), torch.from_numpy(B_test)):
            B_pred = model(X)

            angle_errs_baseline, amp_errs_baseline = calculate_metrics_baseline(
                B.numpy()
            )
            avg_angle_errors_baseline.append(np.mean(angle_errs_baseline, axis=0))
            avg_amp_errors_baseline.append(np.mean(amp_errs_baseline, axis=0))

            angle_errs, amp_errs = calculate_metrics(B, B_pred)
            avg_angle_errors.append(np.mean(angle_errs, axis=0))
            avg_amp_errors.append(np.mean(amp_errs, axis=0))

    return {
        "angle_errors_baseline": avg_angle_errors_baseline,
        "amp_errors_baseline": avg_amp_errors_baseline,
        "angle_errors": avg_angle_errors,
        "amp_errors": avg_amp_errors,
    }
