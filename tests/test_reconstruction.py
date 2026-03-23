import numpy as np

from src.python.modal_lqr import build_model, reconstruct_field


def test_single_mode_reconstruction_vanishes_on_boundary():
    model = build_model(M=2)
    q = np.zeros(len(model.modes))
    q[0] = 1.0
    _, _, U = reconstruct_field(model, q, grid_size=31)
    assert np.allclose(U[0, :], 0.0)
    assert np.allclose(U[-1, :], 0.0)
    assert np.allclose(U[:, 0], 0.0)
    assert np.allclose(U[:, -1], 0.0)


def test_build_model_includes_viscous_damping():
    gamma = 0.3
    model = build_model(M=2, gamma=gamma)
    N = len(model.modes)
    assert model.gamma == gamma
    assert np.allclose(model.A[:N, N:], np.eye(N))
    assert np.allclose(model.A[N:, N:], -gamma * np.eye(N))

