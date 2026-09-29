import numpy as np
import pytest
import torch

from instanseg.utils.utils import _torch_quantiles, percentile_normalize


@pytest.mark.parametrize("n", [1, 2, 7, 1000, 4097])
def test_torch_quantiles_matches_numpy(n: int) -> None:
    x = torch.rand(n)
    qs = [0.001, 0.5, 0.999]
    expected = np.quantile(x.numpy(), qs)
    got = torch.stack(_torch_quantiles(x, qs)).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_percentile_normalize_cuda_above_2_pow_24() -> None:
    # Each channel has > 2**24 elements, which torch.quantile rejects on CUDA (issue #146).
    img = torch.rand((4128, 4128, 2), device="cuda")
    out = percentile_normalize(img.clone())
    expected = percentile_normalize(img.cpu())
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)
