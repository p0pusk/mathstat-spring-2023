import numpy as np
import scipy.stats as sps
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
from statsmodels.distributions.empirical_distribution import ECDF


def edf(data, cdf, x, title):
    fig, axes = plt.subplots(1, len(data), figsize=(12, 5))
    fig.suptitle(title)
    for i, inf in enumerate(data):
        y1 = ECDF(inf)(x)
        y2 = cdf(x)
        axes[i].plot(x, y1)
        # axes[i].set_xlabel("Плотность")
        axes[i].plot(x, y2)
        axes[i].set_title(f"n = {len(inf)}")
    plt.show()


def kde(data, pdf, x, title):
    scales = [0.5, 1.0, 2.0]
    fig, ax = plt.subplots(1, len(scales), figsize=(12, 4))
    fig.suptitle(f"{title}, n = {len(data)}")
    for i, scale in enumerate(scales):
        sns.kdeplot(data, ax=ax[i], bw_method="silverman", bw_adjust=scale, label="kde")
        ax[i].set_xlim([x[0], x[-1]])
        ax[i].set_ylim([0, 1])
        ax[i].plot(x, [pdf(xk) for xk in x], label="pdf")
        ax[i].set_ylabel("Плотность")
        ax[i].legend()
        ax[i].set_title(f"h={str(scale)}*$h_n$")
    plt.show()


def edfkde():
    edf(
        [
            np.random.standard_normal(20),
            np.random.standard_normal(60),
            np.random.standard_normal(100),
        ],
        sps.norm.cdf,
        np.linspace(-4, 4, 100),
        "Normal",
    )
    edf(
        [
            np.random.standard_cauchy(20),
            np.random.standard_cauchy(60),
            np.random.standard_cauchy(100),
        ],
        sps.cauchy.cdf,
        np.linspace(-4, 4, 100),
        "Cauchy",
    )
    edf(
        [
            np.random.laplace(loc=0, scale=1.0 / np.sqrt(2.0), size=20),
            np.random.laplace(loc=0, scale=1.0 / np.sqrt(2.0), size=60),
            np.random.laplace(loc=0, scale=1.0 / np.sqrt(2.0), size=100),
        ],
        lambda x: sps.laplace.cdf(x, loc=0, scale=1.0 / np.sqrt(2.0)),
        np.linspace(-4, 4, 100),
        "Laplace",
    )
    edf(
        [
            np.random.poisson(lam=10, size=20),
            np.random.poisson(lam=10, size=60),
            np.random.poisson(lam=10, size=100),
        ],
        lambda x: sps.poisson.cdf(x, 10),
        np.linspace(6, 14, 100),
        "Poisson",
    )
    edf(
        [
            np.random.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=20),
            np.random.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=60),
            np.random.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=100),
        ],
        lambda x: sps.uniform.cdf(x, -np.sqrt(3.0), 2 * np.sqrt(3.0)),
        np.linspace(-4, 4, 100),
        "Uniform",
    )

    for n in [20, 60, 100]:
        kde(
            np.random.standard_normal(n),
            sps.norm.pdf,
            np.linspace(-4, 4, 100),
            "Normal",
        )
    for n in [20, 60, 100]:
        kde(
            np.random.standard_cauchy(n),
            sps.cauchy.pdf,
            np.linspace(-4, 4, 100),
            "Cauchy",
        )
    for n in [20, 60, 100]:
        kde(
            np.random.laplace(loc=0, scale=1.0 / np.sqrt(2.0), size=n),
            lambda x: sps.laplace.pdf(x, loc=0, scale=1.0 / np.sqrt(2.0)),
            np.linspace(-4, 4, 100),
            "Laplace",
        )
    for n in [20, 60, 100]:
        kde(
            np.random.poisson(lam=10, size=n),
            lambda x: sps.poisson.pmf(x, 10),
            np.linspace(6, 14, 10),
            "Poisson",
        )
    for n in [20, 60, 100]:
        kde(
            np.random.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=n),
            lambda x: sps.uniform.pdf(x, -np.sqrt(3.0), 2 * np.sqrt(3.0)),
            np.linspace(-4, 4, 100),
            "Uniform",
        )


if __name__ == "__main__":
    edfkde()
