import numpy as np
from pprint import pprint
from numpy.core.fromnumeric import mean
from numpy.lib import median
from tabulate import tabulate


def get_characteristics(generator, sample_size: tuple):
    iters = 1000
    characteristics = dict()
    for num in sample_size:
        characteristics[num] = dict()
        mean = []
        median = []
        z_R = []
        z_Q = []
        z_tr = []
        for _ in range(iters):
            data = generator(num)
            data.sort()

            mean.append(data.mean())
            median.append(np.median(data))
            z_R.append((data[0] + data[-1]) / 2)
            z_Q.append((np.quantile(data, 0.25) + np.quantile(data, 0.75)) / 2)
            r = num // 4
            z_tr.append(sum(data[r:-r]) / (num - 2 * r))

        characteristics[num]["mean"] = round(np.mean(mean), 4)
        characteristics[num]["median"] = round(np.mean(median), 4)
        characteristics[num]["z_R"] = round(np.mean(z_R), 4)
        characteristics[num]["z_Q"] = round(np.mean(z_Q), 4)
        characteristics[num]["z_tr"] = round(np.mean(z_tr), 4)
        characteristics[num]["d_mean"] = round(np.std(mean) ** 2, 4)
        characteristics[num]["d_median"] = round(np.std(median) ** 2, 4)
        characteristics[num]["d_z_R"] = round(np.std(z_R) ** 2, 4)
        characteristics[num]["d_z_Q"] = round(np.std(z_Q) ** 2, 4)
        characteristics[num]["d_z_tr"] = round(np.std(z_tr) ** 2, 4)

        characteristics[num]["mean+"] = (
            "["
            + str(round(np.mean(mean) - np.std(mean), 4))
            + "; "
            + str(round(np.mean(mean) + np.std(mean), 4))
            + "]"
        )
        characteristics[num]["median+"] = (
            "["
            + str(round(np.mean(median) - np.std(median), 4))
            + "; "
            + str(round(np.mean(median) + np.std(median), 4))
            + "]"
        )
        characteristics[num]["z_R+"] = (
            "["
            + str(round(np.mean(z_R) - np.std(z_R), 4))
            + "; "
            + str(round(np.mean(z_R) + np.std(z_R), 4))
            + "]"
        )
        characteristics[num]["z_Q+"] = (
            "["
            + str(round(np.mean(z_Q) - np.std(z_Q), 4))
            + "; "
            + str(round(np.mean(z_Q) + np.std(z_Q), 4))
            + "]"
        )
        characteristics[num]["z_tr+"] = (
            "["
            + str(round(np.mean(z_tr) - np.std(z_tr), 4))
            + "; "
            + str(round(np.mean(z_tr) + np.std(z_tr), 4))
            + "]"
        )
    return characteristics


def chars() -> dict[dict]:
    sample_size = (10, 100, 1000)
    chars = dict()
    chars["normal"] = get_characteristics(np.random.standard_normal, sample_size)

    chars["cauchy"] = get_characteristics(np.random.standard_cauchy, sample_size)
    chars["laplace"] = get_characteristics(
        lambda n: np.random.laplace(loc=0, scale=1.0 / np.sqrt(2.0), size=n),
        sample_size,
    )
    chars["poisson"] = get_characteristics(
        lambda n: np.random.poisson(lam=10, size=n), sample_size
    )
    chars["uniform"] = get_characteristics(
        lambda n: np.random.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=n),
        sample_size,
    )
    return chars


if __name__ == "__main__":
    characts = chars()
    for dist in ["normal", "cauchy", "laplace", "poisson", "uniform"]:
        print(dist)
        print(r"\begin{table}")
        print(rf"\caption{{dist}}")
        for n in (10, 100, 1000):
            v = characts[dist][n]
            print(r"\begin{adjustbox}{width=\textwidth}")
            print(r"\begin{tabular}{| c | c | c | c | c | c |}")
            print(r"\hline")
            print(
                f"n = {n}"
                + r" & $\bar{x}$ & $med x$ & $z_R$ & $z_Q$ & $z_{tr}$ \\\hline"
            )
            print("$E(x)$ &", end=" ")
            print(
                f"{v['mean']} & {v['median']} & {v['z_R']} & {v['z_Q']} & {v['z_tr']}",
                end=" ",
            )
            print(r"\\\hline")
            print("$D(x)$ &", end=" ")
            print(
                (
                    f"{v['d_mean']} & {v['d_median']} & {v['d_z_R']} & {v['d_z_Q']} &"
                    f" {v['d_z_tr']}"
                ),
                end=" ",
            )
            print(r"\\\hline")
            print(r"$E(x) \pm \sqrt{D(x)}$ &", end=" ")
            print(
                (
                    f"{v['mean+']} & {v['median+']} & {v['z_R+']} & {v['z_Q+']} &"
                    f" {v['z_tr+']}"
                    r"\\\hline"
                )
            )
            print(r"\end{tabular}")
            print(r"\end{adjustbox}")
            print()
        print(r"\end{table}")
