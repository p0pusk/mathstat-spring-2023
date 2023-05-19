from pathlib import Path
import numpy as np
import scipy.stats as sps
import seaborn as sns
import matplotlib.pyplot as plt
from math import gamma


def normal_distr_graphics(
    loc: float = 0, scale: float = 1, sizes: tuple = (10, 50, 100)
) -> None:
    grid = np.linspace(-3, 3, 1000)
    plt.figure(figsize=(15, 5)).suptitle(
        r"Случайная величина $\xi \sim \mathcal{N}(0, 1)$"
    )

    for i in range(len(sizes)):
        normal_distr = np.random.standard_normal(size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.hist(
            normal_distr, bins=30, density=True, alpha=0.6, label="Гистограмма выборки"
        )
        plt.plot(
            grid,
            sps.norm.pdf(grid),
            color="red",
            lw=3,
            label="Плотность случайной величины",
        )
        plt.title(f"\nРазмер выборки: {sizes[i]}", fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.savefig(Path("lab1-4/images/histogram/normal.png"))
    # plt.show()


def cauchy_distr_graphics(
    loc: float = 0, scale: float = 1, sizes: tuple = (10, 50, 100)
) -> None:
    grid = np.linspace(-30, 30, 1000)
    plt.figure(figsize=(15, 5)).suptitle(
        r"Случайная величина $\xi \sim \mathcal{C}(0, 1)$"
    )

    for i in range(len(sizes)):
        # cauchy_distr = np.random.standard_cauchy(sizes[i])
        cauchy_distr = sps.cauchy.rvs(loc=0, scale=1, size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.xlim([-10, 10])
        # plt.hist(cauchy_distr, density=True,
        #         alpha=0.6, label='Гистограмма выборки')
        sns.histplot(cauchy_distr, kde=False, stat="density", label="samples")
        plt.plot(
            grid,
            sps.cauchy.pdf(grid),
            color="red",
            lw=3,
            label="Плотность случайной величины",
        )
        plt.title(f"\nРазмер выборки: {sizes[i]}", fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.savefig(Path("lab1-4/images/histogram/cauchy.png"))
    # plt.show()


def laplace_distr_graphics(
    loc: float = 0, scale: float = 1, sizes: tuple = (10, 50, 100)
) -> None:
    grid = np.linspace(-3, 3, 1000)
    plt.figure(figsize=(15, 5)).suptitle(
        r"Случайная величина $\xi \sim \mathcal{L}(0, 1/\sqrt{2})$"
    )

    for i in range(len(sizes)):
        cauchy_distr = np.random.laplace(loc=0, scale=1.0 / np.sqrt(2.0), size=sizes[i])
        plt.subplot(1, 3, i + 1)
        # plt.xlim([-30, 30])
        plt.hist(
            cauchy_distr, bins=30, density=True, alpha=0.6, label="Гистограмма выборки"
        )
        plt.plot(
            grid,
            sps.laplace.pdf(grid, loc=0, scale=1.0 / np.sqrt(2.0)),
            color="red",
            lw=3,
            label="Плотность случайной величины",
        )
        plt.title(f"\nРазмер выборки: {sizes[i]}", fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.savefig(Path("lab1-4/images/histogram/laplace.png"))
    # plt.show()


def poisson_distr_graphics(
    loc: float = 0, scale: float = 1, sizes: tuple = (10, 50, 100)
) -> None:
    grid = np.linspace(0, 20, 1000)
    plt.figure(figsize=(15, 5)).suptitle(
        r"Случайная величина $\xi \sim \mathcal{P}(10)$"
    )

    for i in range(len(sizes)):
        cauchy_distr = np.random.poisson(lam=10, size=sizes[i])
        plt.subplot(1, 3, i + 1)
        # plt.xlim([-30, 30])
        plt.hist(
            cauchy_distr, bins=30, density=True, alpha=0.6, label="Гистограмма выборки"
        )
        # plt.plot(grid, sps.poisson.pmf(grid, 10), color='red',
        #         lw=3, label='Плотность случайной величины')
        y = [(10**x * np.exp(-10) / gamma(x + 1)) for x in grid]
        plt.plot(grid, y, color="red", lw=3, label="Плотность случайной величины")
        plt.title(f"\nРазмер выборки: {sizes[i]}", fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.savefig(Path("lab1-4/images/histogram/poisson.png"))
    # plt.show()


def uniform_distr_graphics(
    loc: float = 0, scale: float = 1, sizes: tuple = (10, 50, 100)
) -> None:
    grid = np.linspace(-3, 3, 1000)
    plt.figure(figsize=(15, 5)).suptitle(
        r"Случайная величина $\xi \sim \mathcal{U}(-\sqrt{3}, \sqrt{3})$"
    )

    for i in range(len(sizes)):
        cauchy_distr = np.random.uniform(
            low=-np.sqrt(3.0), high=np.sqrt(3.0), size=sizes[i]
        )
        plt.subplot(1, 3, i + 1)
        # plt.xlim([-30, 30])
        plt.hist(
            cauchy_distr, bins=30, density=True, alpha=0.6, label="Гистограмма выборки"
        )
        plt.plot(
            grid,
            sps.uniform.pdf(grid, loc=-np.sqrt(3.0), scale=2 * np.sqrt(3.0)),
            color="red",
            lw=3,
            label="Плотность случайной величины",
        )
        plt.title(f"\nРазмер выборки: {sizes[i]}", fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.savefig(Path("lab1-4/images/histogram/uniform.png"))
    # plt.show()


def histograms():
    sample = (10, 50, 1000)
    normal_distr_graphics(sizes=sample)
    cauchy_distr_graphics(sizes=sample)
    laplace_distr_graphics(sizes=sample)
    poisson_distr_graphics(sizes=sample)
    uniform_distr_graphics(sizes=sample)


if __name__ == "__main__":
    histograms()
