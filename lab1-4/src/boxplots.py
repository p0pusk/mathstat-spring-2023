from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pprint


def boxplot(data, title):
    nums = [str(len(inf)) for inf in data]
    fig, ax = plt.subplots(1, 1)
    sns.boxplot(data=data, orient="h", ax=ax)
    ax.set(xlabel="x", ylabel="n")
    ax.set(yticklabels=nums)
    ax.set_title(title)
    plt.savefig(Path(f"lab1-4/images/boxplots/{title}.png"))
    plt.show()


def outlier(data):
    iters = 1000
    num = 0
    for i in range(iters):
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        diff = q3 - q1
        x1 = q1 - 1.5 * diff
        x2 = q1 + 1.5 * diff
        num += np.count_nonzero((data < x1) | (data > x2)) / len(data)
    return round(num / iters, 2)


def boxplots():
    print("Processing...")
    outliers = dict()

    print("[ ] Getting boxplot and outliers of normal distribution...")
    data20, data100 = np.random.standard_normal(20), np.random.standard_normal(100)
    boxplot([data20, data100], "Normal")
    outliers["Normal"] = {"20": outlier(data20), "100": outlier(data100)}
    print("[+] Data recieved.")

    print("[ ] Getting boxplot and outliers of Cauchy distribution...")
    data20, data100 = np.random.standard_cauchy(20), np.random.standard_cauchy(100)
    boxplot([data20, data100], "Cauchy")
    outliers["Cauchy"] = {"20": outlier(data20), "100": outlier(data100)}
    print("[+] Data recieved.")

    print("[ ] Getting boxplot and outliers of Laplace distribution...")
    data20, data100 = np.random.laplace(
        loc=0, scale=1.0 / np.sqrt(2.0), size=20
    ), np.random.laplace(loc=0, scale=1.0 / np.sqrt(2.0), size=100)
    boxplot([data20, data100], "Laplace")
    outliers["Laplace"] = {"20": outlier(data20), "100": outlier(data100)}
    print("[+] Data recieved.")

    print("[ ] Getting boxplot and outliers of Poisson distribution...")
    data20, data100 = np.random.poisson(lam=10, size=20), np.random.poisson(
        lam=10, size=100
    )
    boxplot([data20, data100], "Poisson")
    outliers["Poisson"] = {"20": outlier(data20), "100": outlier(data100)}
    print("[+] Data recieved.")

    print("[ ] Getting boxplot and outliers of uniform distribution...")
    data20, data100 = np.random.uniform(
        low=-np.sqrt(3.0), high=np.sqrt(3.0), size=20
    ), np.random.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=100)
    boxplot([data20, data100], "Uniform")
    outliers["Uniform"] = {"20": outlier(data20), "100": outlier(data100)}
    print("[+] Data recieved.")

    pprint.pprint(outliers)


if __name__ == "__main__":
    boxplots()
