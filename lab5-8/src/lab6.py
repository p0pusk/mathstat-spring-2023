import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as opt
from pathlib import Path


def func(x):
    return 2 + 2 * x


def noise_func(x):
    y = []
    for i in x:
        y.append(func(i) + stats.norm.rvs(0, 1))
    return y


def LMM(parameters, x, y):
    alpha_0, alpha_1 = parameters
    sum = 0
    for i in range(len(x)):
        sum += abs(y[i] - alpha_0 - alpha_1 * x[i])
    return sum


def get_MNK_params(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (
        np.mean(x * x) - np.mean(x) ** 2
    )
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1


def get_MNM_params(x, y):
    beta_0, beta_1 = get_MNK_params(x, y)
    result = opt.minimize(LMM, [beta_0, beta_1], args=(x, y), method="SLSQP")
    coeffs = result.x
    alpha_0, alpha_1 = coeffs[0], coeffs[1]
    return alpha_0, alpha_1


def MNK(x, y):
    beta_0, beta_1 = get_MNK_params(x, y)
    print("beta_0 = " + str(beta_0), "beta_1 = " + str(beta_1))
    y_new = [beta_0 + beta_1 * _x for _x in x]
    return y_new


def MNM(x, y):
    alpha_0, alpha_1 = get_MNM_params(x, y)
    print("alpha_0= " + str(alpha_0), "alpha_1 = " + str(alpha_1))
    y_new = [alpha_0 + alpha_1 * _x for _x in x]
    return y_new


def get_dist(y_model, y_regr):
    arr = [(y_model[i] - y_regr[i]) ** 2 for i in range(len(y_model))]
    dist_y = sum(arr)
    return dist_y


def plot_lin_regression(text, x, y, number):
    y_mnk = MNK(x, y)
    y_mnm = MNM(x, y)
    y_dist_mnk = get_dist(y, y_mnk)
    y_dist_mnm = get_dist(y, y_mnm)
    print("MNK distance:", y_dist_mnk)
    print("mnm distance:", y_dist_mnm)
    plt.scatter(x, y, label="Выборка", color="black", marker=".", linewidths=0.7)
    plt.plot(x, func(x), label="Модель", color="lightcoral")
    plt.plot(x, y_mnk, label="МНК", color="steelblue")
    plt.plot(x, y_mnm, label="МНМ", color="lightgreen")
    plt.xlim([-1.8, 2])
    plt.grid()
    plt.legend()
    plt.savefig(Path(rf"lab5-8/images/regression/6_{number}"))
    plt.show()


if __name__ == "__main__":
    x = np.arange(-1.8, 2, 0.2)
    y = noise_func(x)
    plot_lin_regression("NoPerturbations", x, y, 1)

    x = np.arange(-1.8, 2, 0.2)
    y = noise_func(x)
    y[0] += 10
    y[-1] -= 10
    plot_lin_regression("Perturbations", x, y, 2)
