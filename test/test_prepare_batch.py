import numpy as np
import unittest

from teds.data import prepare_batch

class TestPrepareBatch(unittest.TestCase):
    def test_prepare_batch_simple(self):
        np.testing.assert_array_equal(
            [[1], [2], [3]],
            prepare_batch(
                samples=[np.array([1, 2, 3])],
                pad_token_id=0,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[1, 4], [2, 5], [3, 6]],
            prepare_batch(
                samples=[
                    np.array([1, 2, 3]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[1, 4], [2, 5], [0, 6]],
            prepare_batch(
                samples=[
                    np.array([1, 2]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[2, 6], [1, 5], [0, 4]],
            prepare_batch(
                samples=[
                    np.array([1, 2]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
                reverse=True,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[7, 7], [1, 4], [2, 5], [0, 6]],
            prepare_batch(
                samples=[
                    np.array([1, 2]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
                bos_token_id=7,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[1, 4], [2, 5], [7, 6], [0, 7]],
            prepare_batch(
                samples=[
                    np.array([1, 2]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
                eos_token_id=7,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[7, 7], [1, 4], [2, 5], [8, 6], [0, 8]],
            prepare_batch(
                samples=[
                    np.array([1, 2]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
                bos_token_id=7,
                eos_token_id=8,
            ).numpy()
        )

    def test_prepare_batch_place_pad_token_first(self):
        np.testing.assert_array_equal(
            [[1], [2], [3]],
            prepare_batch(
                samples=[np.array([1, 2, 3])],
                pad_token_id=0,
                place_pad_token_first=True,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[1, 4], [2, 5], [3, 6]],
            prepare_batch(
                samples=[
                    np.array([1, 2, 3]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
                place_pad_token_first=True,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[0, 4], [1, 5], [2, 6]],
            prepare_batch(
                samples=[
                    np.array([1, 2]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
                place_pad_token_first=True,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[0, 6], [2, 5], [1, 4]],
            prepare_batch(
                samples=[
                    np.array([1, 2]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
                reverse=True,
                place_pad_token_first=True,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[0, 7], [7, 4], [1, 5], [2, 6]],
            prepare_batch(
                samples=[
                    np.array([1, 2]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
                bos_token_id=7,
                place_pad_token_first=True,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[0, 4], [1, 5], [2, 6], [7, 7]],
            prepare_batch(
                samples=[
                    np.array([1, 2]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
                eos_token_id=7,
                place_pad_token_first=True,
            ).numpy()
        )
        np.testing.assert_array_equal(
            [[0, 7], [7, 4], [1, 5], [2, 6], [8, 8]],
            prepare_batch(
                samples=[
                    np.array([1, 2]),
                    np.array([4, 5, 6]),
                ],
                pad_token_id=0,
                bos_token_id=7,
                eos_token_id=8,
                place_pad_token_first=True,
            ).numpy()
        )

if __name__ == '__main__':
    unittest.main()
