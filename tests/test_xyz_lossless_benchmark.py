import numpy as np
import pytest

from comparison_baselines.xyz_lossless_benchmark import METHODS, decode, encode


@pytest.mark.parametrize('method', METHODS)
def test_xyz_lossless_preserves_order_duplicates_and_grid_edges(method):
    xyz = np.array([[65535, 0, 32768], [65535, 0, 32768], [0, 65535, 0],
                    [0, 0, 0], [1, 3, 65535]], dtype=np.int32)
    data = encode(xyz, method)
    np.testing.assert_array_equal(decode(data, len(xyz)), xyz)
    with pytest.raises(ValueError, match='count'):
        decode(data, len(xyz) - 1)
    with pytest.raises(ValueError, match='CRC'):
        decode(data[:-1] + bytes([data[-1] ^ 1]), len(xyz))
