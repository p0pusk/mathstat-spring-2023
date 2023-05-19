import numpy as np
import pprint


def get_characteristics(generator, sample_size: tuple):
    iters = 1000
    characteristics = dict()
    for num in sample_size:
        characteristics[f"{num}"] = dict()
    for num in sample_size:
        mean = 0
        mean_2 = 0
        median = 0
        median_2 = 0
        z_R = 0
        z_R_2 = 0
        z_Q = 0
        z_Q_2 = 0
        z_tr = 0
        z_tr_2 = 0
        for j in range(iters):
            data = generator(num)
            data.sort()

            temp = data.mean()
            mean += temp
            mean_2 += temp**2

            temp = np.median(data)
            median += temp
            median_2 += temp**2

            temp = (data[0] + data[-1]) / 2
            z_R += temp
            z_R_2 += temp**2

            temp = (np.quantile(data, 0.25) + np.quantile(data, 0.75)) / 2
            z_Q += temp
            z_Q_2 += temp**2

            r = num // 4
            temp = sum(data[r:-r]) / (num - 2 * r)
            z_tr += temp
            z_tr_2 += temp**2

        mean /= iters
        mean_2 /= iters
        median /= iters
        median_2 /= iters
        z_R /= iters
        z_R_2 /= iters
        z_Q /= iters
        z_Q_2 /= iters
        z_tr /= iters
        z_tr_2 /= iters

        d_mean = mean_2 - mean**2
        d_median = median_2 - median**2
        d_z_R = z_R_2 - z_R**2
        d_z_Q = z_Q_2 - z_Q**2
        d_z_tr = z_tr_2 - z_tr**2

        characteristics[f"{num}"]["mean"] = round(mean, 4)
        characteristics[f"{num}"]["median"] = round(median, 4)
        characteristics[f"{num}"]["z_R"] = round(z_R, 4)
        characteristics[f"{num}"]["z_Q"] = round(z_Q, 4)
        characteristics[f"{num}"]["z_tr"] = round(z_tr, 4)
        characteristics[f"{num}"]["d_mean"] = round(d_mean, 4)
        characteristics[f"{num}"]["d_median"] = round(d_median, 4)
        characteristics[f"{num}"]["d_z_R"] = round(d_z_R, 4)
        characteristics[f"{num}"]["d_z_Q"] = round(d_z_Q, 4)
        characteristics[f"{num}"]["d_z_tr"] = round(d_z_tr, 4)

        characteristics[f"{num}"]["mean+"] = (
            "["
            + str(round(mean - np.sqrt(d_mean), 4))
            + "; "
            + str(round(mean + np.sqrt(d_mean), 4))
            + "]"
        )
        characteristics[f"{num}"]["median+"] = (
            "["
            + str(round(median - np.sqrt(d_median), 4))
            + "; "
            + str(round(median + np.sqrt(d_median), 4))
            + "]"
        )
        characteristics[f"{num}"]["z_R+"] = (
            "["
            + str(round(z_R - np.sqrt(d_z_R), 4))
            + "; "
            + str(round(z_R + np.sqrt(d_z_R), 4))
            + "]"
        )
        characteristics[f"{num}"]["z_Q+"] = (
            "["
            + str(round(z_Q - np.sqrt(d_z_Q), 4))
            + "; "
            + str(round(z_Q + np.sqrt(d_z_Q), 4))
            + "]"
        )
        characteristics[f"{num}"]["z_tr+"] = (
            "["
            + str(round(z_tr - np.sqrt(d_z_tr), 4))
            + "; "
            + str(round(z_tr + np.sqrt(d_z_tr), 4))
            + "]"
        )
    return characteristics


def lab2() -> dict[dict]:
    sample_size = (10, 100, 1000)
    chars = dict()
    print("Processing...")
    print("[ ] Getting characteristics of normal distribution...")
    chars["normal"] = get_characteristics(np.random.standard_normal, sample_size)
    print("[+] Characteristics recieved.")
    print("[ ] Getting characteristics of Cauchy distribution...")
    chars["cauchy"] = get_characteristics(np.random.standard_cauchy, sample_size)
    print("[+] Characteristics recieved.")
    print("[ ] Getting characteristics of Laplace distribution...")
    chars["laplace"] = get_characteristics(
        lambda n: np.random.laplace(loc=0, scale=1.0 / np.sqrt(2.0), size=n),
        sample_size,
    )
    print("[+] Characteristics recieved.")
    print("[ ] Getting characteristics of Poisson distribution...")
    chars["poisson"] = get_characteristics(
        lambda n: np.random.poisson(lam=10, size=n), sample_size
    )
    print("[+] Characteristics recieved.")
    print("[ ] Getting characteristics of uniform distribution...")
    chars["uniform"] = get_characteristics(
        lambda n: np.random.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=n),
        sample_size,
    )
    print("[+] Characteristics recieved.")
    return chars


if __name__ == "__main__":
    pprint.pprint(lab2())
