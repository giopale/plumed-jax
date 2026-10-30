"""Compare binary64 arrays without conflating tolerance with bitwise equality."""

import numpy as np


def compare(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError("Comparison shapes differ")
    a = actual.view(np.uint64)
    b = reference.view(np.uint64)
    finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)))
    if not finite:
        return dict(
            finite=False,
            bitwise_equal=bool(np.array_equal(a, b)),
            max_abs=None,
            max_ulp=None,
        )
    sign = np.uint64(1 << 63)
    ordered = lambda x: np.where(x & sign, ~x, x | sign)
    oa, ob = ordered(a), ordered(b)
    ulps = np.maximum(oa, ob) - np.minimum(oa, ob)
    ulps = np.where(actual == reference, np.uint64(0), ulps)
    error = np.abs(actual - reference)
    return dict(
        finite=True,
        bitwise_equal=bool(np.array_equal(a, b)),
        differing_components=int(np.sum(a != b)),
        components=int(a.size),
        max_abs=float(np.max(error, initial=0.0)),
        max_ulp=int(np.max(ulps, initial=np.uint64(0))),
    )
