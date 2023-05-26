import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path


def dispersion_exp(sample):
    return np.mean(list(map(lambda x: x * x, sample))) - (np.mean(sample)) ** 2


def normal(size):
    return np.random.standard_normal(size=size)


def draw_results(x_set: list, m_all: float, s_all: list, number):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    ax1.set_ylim(0, 1)
    ax1.hist(
        x_set[0],
        density=True,
        histtype="stepfilled",
        alpha=0.3,
        label="N(0, 1) hyst n=20",
        color="black",
    )
    ax1.legend(loc="best", frameon=True)
    ax2.set_ylim(0, 1)
    ax2.hist(
        x_set[1],
        density=True,
        histtype="stepfilled",
        alpha=0.3,
        label="N(0, 1) hyst n=100",
        color="black",
    )
    ax2.legend(loc="best", frameon=True)

    ax3.set_ylim(0.9, 1.4)
    ax3.plot(m_all[0], [1, 1], label='"m" interval n = 20', color="lightcoral")
    ax3.plot(m_all[1], [1.1, 1.1], label='"m" interval n = 100', color="steelblue")
    ax3.legend()

    ax4.set_ylim(0.9, 1.4)
    ax4.plot(s_all[0], [1, 1], label="sigma interval n = 20", color="lightcoral")
    ax4.plot(s_all[1], [1.1, 1.1], label="sigma interval n = 100", color="steelblue")
    ax4.legend()

    fig.savefig(Path(rf"lab5-8/images/interval/8_{number}"), dpi=300)

    plt.show()


if __name__ == "__main__":
    n_set = [20, 100]
    x_20 = normal(20)
    x_100 = normal(100)
    x_set = [x_20, x_100]

    alpha = 0.05
    m_all = list()
    s_all = list()
    for i in range(len(n_set)):
        n = n_set[i]
        x = x_set[i]

        m = np.mean(x)
        s = np.sqrt(dispersion_exp(x))

        m1 = [
            m - s * (stats.t.ppf(1 - alpha / 2, n - 1)) / np.sqrt(n - 1),
            m + s * (stats.t.ppf(1 - alpha / 2, n - 1)) / np.sqrt(n - 1),
        ]
        s1 = [
            s * np.sqrt(n) / np.sqrt(stats.chi2.ppf(1 - alpha / 2, n - 1)),
            s * np.sqrt(n) / np.sqrt(stats.chi2.ppf(alpha / 2, n - 1)),
        ]

        m_all.append(m1)
        s_all.append(s1)

        print("n: %i" % (n))
        print("m: %.2f, %.2f" % (m1[0], m1[1]))
        print("sigma: %.2f, %.2f" % (s1[0], s1[1]))

    draw_results(x_set, m_all, s_all, 1)

    m_all = list()
    s_all = list()
    for i in range(len(n_set)):
        n = n_set[i]
        x = x_set[i]

        m = np.mean(x)
        s = np.sqrt(dispersion_exp(x))

        m_as = [
            m - stats.norm.ppf(1 - alpha / 2) / np.sqrt(n),
            m + stats.norm.ppf(1 - alpha / 2) / np.sqrt(n),
        ]
        e = (sum(list(map(lambda el: (el - m) ** 4, x))) / n) / s**4 - 3
        s_as = [
            s / np.sqrt(1 + stats.norm.ppf(1 - alpha / 2) * np.sqrt((e + 2) / n)),
            s / np.sqrt(1 - stats.norm.ppf(1 - alpha / 2) * np.sqrt((e + 2) / n)),
        ]

        m_all.append(m_as)
        s_all.append(s_as)

        print("m asymptotic: %.2f, %.2f" % (m_as[0], m_as[1]))
        print("sigma asymptotic: %.2f, %.2f" % (s_as[0], s_as[1]))
    draw_results(x_set, m_all, s_all, 2)
