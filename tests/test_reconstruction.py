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


def test_open_loop_energy_conservation_rk4():
    model = build_model(M=16, gamma=0.0)
    x0 = np.zeros(2 * len(model.modes))
    x0[0] = 1.0  # excite first q mode

    from src.python.modal_lqr import simulate_open_loop, compute_energy

    t, y = simulate_open_loop(model, x0, T=6.0, nt=800)
    E = compute_energy(model, y)

    rel_change = np.max(np.abs(E - E[0]) / E[0])
    assert rel_change < 0.01, f"energy drift too high: {rel_change:.5f}"


