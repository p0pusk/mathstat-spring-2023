import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import statistics
from tabulate import tabulate
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from pathlib import Path


def get_two_dim_normal_sample(size, rho, mu=[0, 0], d=[1.0, 1.0]):
    cov_matrix = [[d[0], rho], [rho, d[1]]]
    return stats.multivariate_normal.rvs(mu, cov_matrix, size=size)


def get_mix_two_dim_normal_sample(size, rho):
    return 0.9 * get_two_dim_normal_sample(size, 0.9) + 0.1 * get_two_dim_normal_sample(
        size, -0.9, d=[10, 10]
    )


def get_quadrant_coeff(x, y):
    size = len(x)
    median_x = np.median(x)
    median_y = np.median(y)
    n = [0, 0, 0, 0]
    for i in range(size):
        if x[i] >= median_x and y[i] >= median_y:
            n[0] += 1
        elif x[i] < median_x and y[i] >= median_y:
            n[1] += 1
        elif x[i] < median_x and y[i] < median_y:
            n[2] += 1
        elif x[i] >= median_x and y[i] < median_y:
            n[3] += 1
    return (n[0] + n[2] - n[1] - n[3]) / size


def get_correlation_coeffs(get_sample, size, rho, repeats):
    pearson, quadrant, spearman = [], [], []
    for i in range(repeats):
        sample = get_sample(size, rho)
        x, y = sample[:, 0], sample[:, 1]
        pearson.append(stats.pearsonr(x, y)[0])
        spearman.append(stats.spearmanr(x, y)[0])
        quadrant.append(get_quadrant_coeff(x, y))
    return pearson, spearman, quadrant


def create_table(pearson, spearman, quadrant, size, repeats, rho=None):
    if rho is not None:
        rows = [["rho = " + str(rho), "r", "r_{S}", "r_{Q}"]]
    else:
        rows = [["size = " + str(size), "r", "r_{S}", "r_{Q}"]]
    p = np.median(pearson)
    s = np.median(spearman)
    q = np.median(quadrant)
    rows.append(
        [
            "E(z)",
            np.around(p, decimals=3),
            np.around(s, decimals=3),
            np.around(q, decimals=3),
        ]
    )

    p = np.median([pearson[k] ** 2 for k in range(repeats)])
    s = np.median([spearman[k] ** 2 for k in range(repeats)])
    q = np.median([quadrant[k] ** 2 for k in range(repeats)])
    rows.append(
        [
            "E(z^2)",
            np.around(p, decimals=3),
            np.around(s, decimals=3),
            np.around(q, decimals=3),
        ]
    )

    p = statistics.variance(pearson)
    s = statistics.variance(spearman)
    q = statistics.variance(quadrant)
    rows.append(
        [
            "D(z)",
            np.around(p, decimals=3),
            np.around(s, decimals=3),
            np.around(q, decimals=3),
        ]
    )

    return tabulate(rows, [], tablefmt="latex")


def create_ellipse(x, y, ax, n_std=3.0, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)

    ellipse = Ellipse(
        (0, 0), width=rad_x * 2, height=rad_y * 2, facecolor="none", **kwargs
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def draw_ellipse(size, rhos):
    fig, ax = plt.subplots(1, 3)
    str_size = "n = " + str(size)
    titles = [
        str_size + r", $ \rho = 0$",
        str_size + r", $\rho = 0.5 $",
        str_size + r", $ \rho = 0.9$",
    ]
    for i in range(len(rhos)):
        num, rho = i, rhos[i]
        sample = get_two_dim_normal_sample(size, rho)
        x, y = sample[:, 0], sample[:, 1]
        create_ellipse(x, y, ax[num], edgecolor="navy")
        ax[num].grid()
        ax[num].scatter(x, y, s=5)
        ax[num].set_title(titles[num])
        fig.savefig(Path(f"lab5-8/images/ellipse/5_{size}"))
    # plt.show()


if __name__ == "__main__":
    sizes = [20, 60, 100]
    rhos = [0, 0.5, 0.9]
    repeats = 1000

    for size in sizes:
        for ro in rhos:
            pearson, spearman, quadrant = get_correlation_coeffs(
                get_two_dim_normal_sample, size, ro, repeats
            )
            print(
                "\n"
                + str(size)
                + "\n"
                + str(create_table(pearson, spearman, quadrant, size, repeats, ro))
            )

        pearson, spearman, quadrant = get_correlation_coeffs(
            get_mix_two_dim_normal_sample, size, 0, repeats
        )
        print(
            "\n"
            + str(size)
            + "\n"
            + str(create_table(pearson, spearman, quadrant, size, repeats))
        )
        draw_ellipse(size, rhos)
