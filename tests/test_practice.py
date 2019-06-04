import numpy
from chainer.dataset import concat_examples


def test_concat_example():
    print(concat_examples([([1, 2], [3, 4]), ([5, 6], [7, 8])]))
    print(concat_examples([([1, 2], [3, 4]), ([5, 6], [7, 8])]))
    res = concat_examples([
        {"a": [3, 4], "b": [5, 6]},
        {"a": [9, 10], "b": [11, 12]},
    ])
    print(res)
    assert res == {"a": numpy.array([[3, 4], [9, 10]]), "b": numpy.array([[5, 6], [11, 12]])}


if __name__ == '__main__':
    test_concat_example()