import os
if 0:
    os.environ['NUMBA_DUMP_IR'] = '1'
    os.environ['NUMBA_DUMP_OPTIMIZED'] = '1'
    os.environ['NUMBA_DUMP_ASSEMBLY'] = '1'


from numba import *

import numpy as np
import math
import time

dtype = float32
width = 4
npars = 14

a1d = dtype[:]
a2d = dtype[:, :]

signatures = []
signatures.append(
    (int64[:], ) + (a2d,) * 2 + (a1d,) * npars + (a2d,)
)

@guvectorize(signatures, '(), (n, w),(m, w)' + ',(w)'*npars + '->(n, w)', nopython=True)
def jr_simd(niter, y, c,
                   src,
                   nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu,
                   dx):
    one = dtype(1.0)
    two = dtype(2.0)
    dt = dtype(0.00000001)
    for t in range(niter[0]):
        for i in range(width):
            sigm_y1_y2 = two * nu_max[i] / (one + math.exp(r[i] * (v0[i] - (y[1, i] - y[2, i]))))
            sigm_y0_1 = two * nu_max[i] / (one + math.exp(r[i] * (v0[i] - (a_1[i] * J[i] * y[0, i]))))
            sigm_y0_3 = two * nu_max[i] / (one + math.exp(r[i] * (v0[i] - (a_3[i] * J[i] * y[0, i]))))
            dx[0, i] = y[3, i]
            dx[1, i] = y[4, i]
            dx[2, i] = y[5, i]
            dx[3, i] = A[i] * a[i] * sigm_y1_y2 - two * a[i] * y[3, i] - a[i] ** 2 * y[0, i]
            dx[4, i] = A[i] * a[i] * (mu[i] + a_2[i] * J[i] * sigm_y0_1 + c[0, i] + src[i]) - two * a[i] * y[4, i] - a[i] ** 2 * y[1, i]
            dx[5, i] = B[i] * b[i] * (a_4[i] * J[i] * sigm_y0_3) - two * b[i] * y[5, i] - b[i] ** 2 * y[2, i]
            y[0, i] += dt * dx[0, i]
            y[1, i] += dt * dx[1, i]
            y[2, i] += dt * dx[2, i]
            y[3, i] += dt * dx[3, i]
            y[4, i] += dt * dx[4, i]
            y[5, i] += dt * dx[5, i]


@guvectorize([(int64[:],) + (dtype[:],) * 17], '(),(n),(m)' + ',()'*14 + '->(n)', nopython=True)
def jr2(niter, y, c,
                   src,
                   nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu,
                   dx):
    one = dtype(1.0)
    two = dtype(2.0)
    dt = dtype(0.00000001)
    for t in range(niter[0]):
        sigm_y1_y2 = two * nu_max[0] / (one + math.exp(r[0] * (v0[0] - (y[1] - y[2]))))
        sigm_y0_1 = two * nu_max[0] / (one + math.exp(r[0] * (v0[0] - (a_1[0] * J[0] * y[0]))))
        sigm_y0_3 = two * nu_max[0] / (one + math.exp(r[0] * (v0[0] - (a_3[0] * J[0] * y[0]))))
        dx[0] = y[3]
        dx[1] = y[4]
        dx[2] = y[5]
        dx[3] = A[0] * a[0] * sigm_y1_y2 - two * a[0] * y[3] - a[0] ** 2 * y[0]
        dx[4] = A[0] * a[0] * (mu[0] + a_2[0] * J[0] * sigm_y0_1 + c[0] + src[0]) - two * a[0] * y[4] - a[0] ** 2 * y[1]
        dx[5] = B[0] * b[0] * (a_4[0] * J[0] * sigm_y0_3) - two * b[0] * y[5] - b[0] ** 2 * y[2]
        y[0] += dt * dx[0]
        y[1] += dt * dx[1]
        y[2] += dt * dx[2]
        y[3] += dt * dx[3]
        y[4] += dt * dx[4]
        y[5] += dt * dx[5]


nsvar = 6
nnode = 128
print(nnode)

cast = lambda arr: arr.astype(getattr(np, dtype.name))

y = cast(np.random.randn(nnode, nsvar))
c = cast(np.random.randn(nnode, 1))
dy = y.copy()
src, nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu = cast(np.random.randn(npars))

niter = 0
tic = time.time()
niniter = 64
while (time.time() - tic) < 1.0:
    for _ in range(4):
        jr2(niniter, y, c, src, nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu, dy)
        niter += niniter
print(niter)

nchunk = int(nnode / width)
y = y.reshape((nchunk, width, nsvar)).transpose((0, 2, 1)).copy()
c = c.reshape((nchunk, width, 1)).transpose((0, 2, 1)).copy()
dy = y.copy()
src, nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu = cast(np.random.randn(npars, width))

niter = 0
tic = time.time()
while (time.time() - tic) < 1.0:
    for _ in range(4):
        jr_simd(niniter, y, c, src, nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu, dy)
        niter += niniter
print(niter)