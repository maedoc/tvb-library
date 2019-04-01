import time
import numpy as np
import numba


@numba.jit
def fn_aos(dx, x):
    a = numba.float32(1.05)
    t = numba.float32(3.0)
    r3 = numba.float32(1.0 / 3.0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1] / 2):
            i0 = j * 2
            i1 = j * 2 + 1
            dx[i, i0] = t * (x[i, i0] - x[i, i0]*x[i, i0]*x[i, i0] / r3 + x[i, i1])
            dx[i, i1] = (a - x[i, i0]) / t


@numba.jit
def fn_soa(dx, x):
    a = numba.float32(1.05)
    t = numba.float32(3.0)
    r3 = numba.float32(1.0 / 3.0)
    for j in range(x.shape[0] / 2):
        i0 = j * 2
        i1 = j * 2 + 1
        for i in range(x.shape[1]):
            dx[i0, i] = t * (x[i0, i] - x[i0, i]*x[i0, i]*x[i0, i] / r3 + x[i1, i])
            dx[i1, i] = (a - x[i0, i]) / t


@numba.jit
def fn_soa2(dx, x):
    a = numba.float32(1.05)
    t = numba.float32(3.0)
    r3 = numba.float32(1.0 / 3.0)
    for j in range(x.shape[0] / 2):
        i0 = j * 2
        i1 = j * 2 + 1
        dx[i0] = t * (x[i0] - x[i0]*x[i0]*x[i0] / r3 + x[i1])
        dx[i1] = (a - x[i0]) / t


def fn_soa3(dx, x):
    a = 1.05
    t = 3.0
    r3 = 1.0 / 3.0
    for j in range(x.shape[0] / 2):
        i0 = j * 2
        i1 = j * 2 + 1
        dx[i0] = t * (x[i0] - x[i0]*x[i0]*x[i0] / r3 + x[i1])
        dx[i1] = (a - x[i0]) / t



@numba.jit
def fn_aosoa(dx, x):
    a = numba.float32(1.05)
    t = numba.float32(3.0)
    r3 = numba.float32(1.0 / 3.0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            i0 = j * 2
            i1 = j * 2 + 1
            for k in range(x.shape[2]):
                dx[i, i0, k] = t * (x[i, i0, k] - x[i, i0, k] * x[i, i0, k] * x[i, i0, k] / r3 + x[i, i1, k])
                dx[i, i1, k] = (a - x[i, i0, k]) / t


def iterps(fn):
    tic = time.time()
    n = 0
    while (time.time() - tic) < 1.0:
        fn()
        n += 1
    return n


if __name__ == '__main__':
    n, m = 64, 8

    for n in [64, 128, 1024, 16384]:
        print n
        dx, x = np.zeros((2, n, m))
        print 'fn_aos', iterps(lambda : fn_aos(dx, x))

        dx, x = np.zeros((2, m, n))
        print 'fn_soa', iterps(lambda : fn_soa(dx, x))

        dx, x = np.zeros((2, m, n))
        print 'fn_soa2', iterps(lambda : fn_soa2(dx, x))

        dx, x = np.zeros((2, m, n))
        print 'fn_soa3', iterps(lambda : fn_soa3(dx, x))

        dx, x = np.zeros((2, n / 4, m, 4))
        print 'fn_aosoa', iterps(lambda : fn_aosoa(dx, x))

        print


    # soa is consistently faster than aos
    # aosoa is very poor performance
    # gufunc approach is aos, so there's room for improvement
    # soa w/ vector expressions is unfortunately slow but similar to aos (?)

    # should fold modes into svars, to avoid overhead of modes
    # provide some automation for fold/unfold operations