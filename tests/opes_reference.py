"""Line-by-line Python transcription of PLUMED 2.10.0 OPESmetad.cpp (serial).

Used only as a test oracle. It mirrors the C++ control flow with Python lists
(including std::vector::erase), scalar loops and the first-step skip, so it is
structurally independent of the vectorized JAX implementation. Out of scope:
NLIST, EXPLORE, EXCLUDED_REGION and file I/O.
"""
import math


def pbc(x):
    return x - math.floor(x + 0.5)


class Reference:
    def __init__(self, *, sigma, barrier, pace, kT, periods=None, periodic_min=None,
                 bias_factor=None, epsilon=None, kernel_cutoff=None,
                 compression_threshold=1.0, sigma_min=(), adaptive_stride=None,
                 fixed_sigma=False, recursive_merge=True, no_zed=False,
                 calc_work=False, walkers=1, dimension=None):
        self.kbt = kT
        self.stride = pace
        self.walkers = walkers
        self.adaptive = sigma == "adaptive"
        self.ncv = dimension if self.adaptive else len(sigma)
        self.periods = list(periods or [0.0] * self.ncv)
        self.pmin = list(periodic_min or [-p / 2 for p in self.periods])
        self.biasfactor = barrier / kT if bias_factor is None else bias_factor
        self.prefactor = 1.0 if math.isinf(self.biasfactor) else 1 - 1 / self.biasfactor
        self.adaptive_stride = adaptive_stride or 10 * pace
        self.sigma0 = [[0.0] * self.ncv if self.adaptive else list(sigma) for _ in range(walkers)]
        self.av_cv = [[0.0] * self.ncv for _ in range(walkers)]
        self.av_M2 = [[0.0] * self.ncv for _ in range(walkers)]
        self.adaptive_counter = 0
        self.sigma_min = [list(sigma_min) for _ in range(walkers)]
        self.epsilon = math.exp(-barrier / self.prefactor / kT) if epsilon is None else epsilon
        self.sum_weights = self.epsilon ** self.prefactor
        self.sum_weights2 = self.sum_weights ** 2
        cutoff = math.sqrt(2 * barrier / self.prefactor / kT) if kernel_cutoff is None else kernel_cutoff
        self.cutoff2 = cutoff * cutoff
        self.val_at_cutoff = math.exp(-0.5 * self.cutoff2)
        self.threshold2 = compression_threshold ** 2
        self.no_zed = no_zed
        if no_zed:
            self.sum_weights = self.sum_weights2 = 1.0
        self.fixed_sigma = fixed_sigma
        self.recursive_merge = recursive_merge
        self.calc_work = calc_work
        self.work = [0.0] * walkers
        self.counter = 1
        self.zed = 1.0
        self.kernels = []  # [height, center, sigma]
        self.kde_norm = self.sum_weights
        self.first = True
        self.recursive_merges = 0

    def difference(self, i, a, b):
        d = b - a
        p = self.periods[i]
        return pbc(d / p) * p if p > 0 else d

    def bring_back(self, i, d):
        p = self.periods[i]
        middle = self.pmin[i] + p / 2
        return middle + self.difference(i, middle, d)

    def evaluate(self, kernel, x):
        h, c, s = kernel
        norm2 = 0.0
        for i in range(self.ncv):
            d = self.difference(i, c[i], x[i]) / s[i]
            norm2 += d * d
            if norm2 >= self.cutoff2:
                return 0.0
        return h * (math.exp(-0.5 * norm2) - self.val_at_cutoff)

    def bias(self, cv):
        prob = sum(self.evaluate(k, cv) for k in self.kernels) / self.kde_norm
        return self.kbt * self.prefactor * math.log(prob / self.zed + self.epsilon)

    def merge(self, k1, k2):
        h = k1[0] + k2[0]
        for i in range(self.ncv):
            periodic = self.periods[i] > 0
            if periodic:
                k1[1][i] = k2[1][i] + self.difference(i, k2[1][i], k1[1][i])
            c = (k1[0] * k1[1][i] + k2[0] * k2[1][i]) / h
            ss1 = k1[0] * (k1[2][i] ** 2 + k1[1][i] ** 2)
            ss2 = k2[0] * (k2[2][i] ** 2 + k2[1][i] ** 2)
            ss = (ss1 + ss2) / h - c * c
            k1[1][i] = self.bring_back(i, c) if periodic else c
            k1[2][i] = math.sqrt(ss)
        k1[0] = h

    def mergeable(self, center, giver):
        min_k, min_norm2 = len(self.kernels), self.threshold2
        for k, kernel in enumerate(self.kernels):
            if k == giver:
                continue
            norm2 = 0.0
            for i in range(self.ncv):
                d = self.difference(i, center[i], kernel[1][i]) / kernel[2][i]
                norm2 += d * d
                if norm2 >= min_norm2:
                    break
            if norm2 < min_norm2:
                min_norm2, min_k = norm2, k
        return min_k

    def add_kernel(self, height, center, sigma):
        if self.threshold2 != 0:
            taker = self.mergeable(center, len(self.kernels))
            if taker < len(self.kernels):
                self.merge(self.kernels[taker], [height, list(center), list(sigma)])
                if self.recursive_merge:
                    giver = taker
                    taker = self.mergeable(self.kernels[giver][1], giver)
                    while taker < len(self.kernels):
                        if taker > giver:
                            taker, giver = giver, taker
                        self.merge(self.kernels[taker], self.kernels[giver])
                        del self.kernels[giver]
                        self.recursive_merges += 1
                        giver = taker
                        taker = self.mergeable(self.kernels[giver][1], giver)
                return
        self.kernels.append([height, list(center), list(sigma)])

    def update(self, step, cvs, extra=None):
        """One completed MD step for all walkers; cvs is a list per walker."""
        biases = [self.bias(cv) for cv in cvs]  # calculate() with the old state
        if self.first:
            self.first = False
            return biases
        if self.adaptive:
            self.adaptive_counter += 1
            tau = min(self.adaptive_counter, self.adaptive_stride)
            for w, cv in enumerate(cvs):
                for i in range(self.ncv):
                    diff = self.difference(i, self.av_cv[w][i], cv[i])
                    self.av_cv[w][i] += diff / tau
                    self.av_M2[w][i] += diff * self.difference(i, self.av_cv[w][i], cv[i])
            if self.adaptive_counter < self.adaptive_stride and self.counter == 1:
                return biases
        if step % self.stride != 0:
            return biases
        log_weights = [(b + (0 if extra is None else extra[w])) / self.kbt
                       for w, b in enumerate(biases)]
        weights = [math.exp(x) for x in log_weights]
        self.counter += self.walkers
        self.sum_weights += sum(weights)
        self.sum_weights2 += sum(x * x for x in weights)
        neff = (1 + self.sum_weights) ** 2 / (1 + self.sum_weights2)
        self.kde_norm = self.sum_weights
        new = []
        for w in range(self.walkers):
            sigma = list(self.sigma0[w])
            if self.adaptive:
                factor = self.biasfactor
                if self.counter == 1 + self.walkers:
                    for i in range(self.ncv):
                        self.av_M2[w][i] *= self.biasfactor
                    for i in range(self.ncv):
                        self.sigma0[w][i] = math.sqrt(self.av_M2[w][i] / self.adaptive_counter / factor)
                    if not self.sigma_min[w]:
                        assert all(s > 1e-6 for s in self.sigma0[w])
                    else:
                        self.sigma0[w] = [max(a, b) for a, b in zip(self.sigma0[w], self.sigma_min[w])]
                sigma = [math.sqrt(m / self.adaptive_counter / factor) for m in self.av_M2[w]]
                if not self.sigma_min[w]:
                    for i in range(self.ncv):
                        if sigma[i] < 1e-6:
                            sigma[i] = 1e-6
                            self.sigma_min[w] = [1e-6] * self.ncv
                else:
                    sigma = [max(a, b) for a, b in zip(sigma, self.sigma_min[w])]
            if not self.fixed_sigma:
                rescaling = (neff * (self.ncv + 2.0) / 4.0) ** (-1.0 / (4 + self.ncv))
                sigma = [s * rescaling for s in sigma]
                if self.sigma_min[w]:
                    sigma = [max(a, b) for a, b in zip(sigma, self.sigma_min[w])]
            height = weights[w]
            for i in range(self.ncv):
                height *= self.sigma0[w][i] / sigma[i]
            new.append((height, list(cvs[w]), sigma))
        for kernel in new:
            self.add_kernel(*kernel)
        if not self.no_zed:
            total = sum(self.evaluate(kk, k[1]) for k in self.kernels for kk in self.kernels)
            self.zed = total / self.kde_norm / len(self.kernels)
        if self.calc_work:
            for w in range(self.walkers):
                self.work[w] += self.bias(cvs[w]) - biases[w]
        return biases
