import numpy as np

# Reference: https://github.com/joschu/modular_rl
# Reference: ref: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
# Reference: ref: https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py


class AlgorithmRunManagement:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        """Constructor of AlgorithmRunManagement class for running state

        Args:
            shape: input shape of state
            demean: Defaults to True.
            destd: Defaults to True.
            clip: Defaults to 10.0.
        """
        self.demean = demean  # for mean deduction
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        """The __call__ method enables Python programmers to write classes where the instances behave
        like functions and can be called like a function.

        Args:
            x: input data
            update: enable update. Defaults to True.

        Returns:
            updated data
        """
        if update:
            self.rs.push(x)  # push new value to running state
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(
                x, -self.clip, self.clip
            )  # glip gradient to avoid overfit or underfit performance
        return x


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape
