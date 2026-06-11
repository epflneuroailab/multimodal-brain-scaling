import pickle

import numpy as np
import pytest


def test_metrics_import():
    torch = pytest.importorskip("torch")
    from mbs.metrics import CenteredKernelAlignmentTorch, RepresentationalSimilarityAnalysisTorch

    x = torch.eye(4)
    rsa = RepresentationalSimilarityAnalysisTorch()
    cka = CenteredKernelAlignmentTorch()
    assert torch.is_tensor(rsa(x, x))
    assert torch.is_tensor(cka(x, x))


def test_projection_seed_and_weight_initialization():
    torch = pytest.importorskip("torch")
    from mbs.modeling.projection import create_projector

    p1 = create_projector(input_dim=4, output_dim=2, random_seed=7)
    p2 = create_projector(input_dim=4, output_dim=2, random_seed=7)
    assert torch.allclose(
        p1.projection_layer.linear.weight,
        p2.projection_layer.linear.weight,
    )

    weights = torch.ones(2, 4)
    p3 = create_projector(input_dim=4, output_dim=2, projector_weights=weights)
    assert torch.allclose(p3.projection_layer.linear.weight, weights)


def test_metric_primitives_on_small_arrays():
    pytest.importorskip("torch")
    from mbs.metrics import RepresentationalSimilarityAnalysis

    x = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
            [1.0, 3.0, 0.0],
            [4.0, 1.0, 3.0],
        ]
    )
    rsa = RepresentationalSimilarityAnalysis()
    assert np.isfinite(rsa(x, x))


def test_linear_probe_loader_copies_numpy_weights_and_freezes_by_default(tmp_path):
    torch = pytest.importorskip("torch")
    from mbs.training.modeling.decoders import create_decoder
    from mbs.training.modeling.encoder_decoder import load_linear_probe

    decoder = create_decoder(input_dim=3, output_dim=2, num_hidden_layers=0, is_frozen=False)
    weights = np.arange(6, dtype=np.float32).reshape(2, 3)
    bias = np.array([0.5, -0.5], dtype=np.float32)
    probe_path = tmp_path / "roi_V1.pkl"
    with probe_path.open("wb") as f:
        pickle.dump({"W": weights, "b": bias}, f)

    load_linear_probe(decoder, probe_path)

    assert torch.allclose(decoder.fc1.weight, torch.as_tensor(weights))
    assert torch.allclose(decoder.fc1.bias, torch.as_tensor(bias))
    assert not decoder.fc1.weight.requires_grad
    assert not decoder.fc1.bias.requires_grad

    trainable_decoder = create_decoder(input_dim=3, output_dim=2, num_hidden_layers=0, is_frozen=True)
    load_linear_probe(trainable_decoder, probe_path, freeze=False)
    assert trainable_decoder.fc1.weight.requires_grad
    assert trainable_decoder.fc1.bias.requires_grad
