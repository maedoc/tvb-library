import os
if 0:
    os.environ['NUMBA_DUMP_LLVM'] = '1'
    #os.environ['NUMBA_DUMP_OPTIMIZED'] = '1'
    #os.environ['NUMBA_DUMP_ASSEMBLY'] = '1'


from numba import rewrites

@rewrites.register_rewrite('after-inference')
class MyRewrite(rewrites.Rewrite):

    def match(self, irobj, block, *args):
        if not hasattr(self, '_seen'):
            self._seen = set([])
        if block not in self._seen:
            self._seen.add(block)
            print(block)
            self.block = block
            return True
        return False

    def apply(self):
        return self.block


from numba import *

dtype = float32
width = int32(8)

@jit(nopython=True, nogil=True)
def f(x, y):
    acc = dtype(0.0)
    for i in range(width):
        a = (x[i] + y[i]) * x[i]
        b = (x[i] - y[i]) / y[i]
        c = x[i] * y[i] + x[i]
        d = x[i] * y[i] * a + b
        e = a / b * c - d
        acc += (a - b) / (c - d) * e
    return acc

import numpy as np

a, b = np.r_[:2*width].astype(
        getattr(np, dtype.name)).reshape((2, 8))

c = f(a, b)
