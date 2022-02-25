import numpy as np
from scipy.stats import uniform, norm, describe, moment, skew, kurtosis
from scipy.special import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_kappa(skewness_, kurtosis_):
    try:

        kappa = (skewness_ ** 2 * ((kurtosis_ + 3) ** 2)) / (4 * (2 * kurtosis_ -
                                                                  3 * skewness_ ** 2 - 6) * (4 * kurtosis_ - 3 * skewness_ ** 2))

    except ZeroDivisionError:

        kappa = 0

    return kappa


def probability_density_function_1(sigma, skewness_, kurtosis_):

    r = (6 * (kurtosis_ - skewness_ ** 2 - 1)) / \
        (6 + 3 * skewness_ ** 2 - 2 * kurtosis_)
    a1_p_a2 = (sigma / 2) * (skewness_ ** 2 *
                             (r + 2) ** 2 + 16 * (r + 1)) ** 0.5

    m1 = 0.5 * (r - 2 + r * (r + 2) * (((skewness_ ** 2) /
                ((skewness_ ** 2) * ((r + 2) ** 2) + 16 * (r + 1))) ** 0.5))
    m2 = 0.5 * (r - 2 - r * (r + 2) * (((skewness_ ** 2) /
                ((skewness_ ** 2) * ((r + 2) ** 2) + 16 * (r + 1))) ** 0.5))

    if skewness_ > 0:
        tmp_ = m2
        m2 = m1
        m1 = tmp_

    ye = (1 / (a1_p_a2)) * ((((m1 + 1) ** m1) * (m2 + 1) ** m2) / ((m1 + m2 + 2)
                                                                   ** (m1 + m2))) * ((gamma(m1 + m2 + 2)) / ((gamma(m1 + 1)) * (gamma(m2 + 1))))
    a2 = a1_p_a2 / (1 + ((m1 + 1) / (m2 + 1)))
    a1 = a1_p_a2 - a2
    return lambda z: ye * ((1 + z / a1) ** m1) * ((1 - z / a2) ** m2)


def probability_density_function_2(sigma, skewness_, kurtosis_):

    r = (6 * (kurtosis_ - skewness_ ** 2 - 1)) / \
        (-6 - 3 * skewness_ ** 2 + 2 * kurtosis_)
    v = ((-r) * (r - 2) * (skewness_)) / \
        (((16 * (r - 1)) - ((skewness_ ** 2) * ((r - 2) ** 2))) ** 0.5)
    a = (sigma/4) * (((16 * (r - 1)) - ((skewness_ ** 2) * ((r - 2) ** 2))) ** 0.5)
    i, _ = quad(lambda phi: np.exp(-v * np.pi * 0.5) *
                np.power(np.sin(phi), r) * np.exp(v * phi), 0, np.pi)
    y0 = 1 / (a * i)

    me = 0.5 * (r + 2)

    return lambda z: y0 * ((1 + (z / a - v / r) ** 2) ** (-me - v * np.arctan(z / a - v / r)))


def probability_density_function_3(sigma, skewness_, kurtosis_):

    r = (6 * (kurtosis_ - skewness_ ** 2 - 1)) / \
        (6 + 3 * skewness_ ** 2 - 2 * kurtosis_)
    a = (sigma / 2) * (skewness_ ** 2 * (r + 2) ** 2 + 16 * (r + 1)) ** 0.5
    q2 = ((r - 2) / (2)) + ((r * (r + 2)) / (2)) * (((skewness_ ** 2) /
                                                     (((skewness_ ** 2) * ((r + 2) ** 2)) + (16 * (r + 1)))) ** 0.5)
    q1 = (((r - 2) / (2)) - ((r * (r + 2)) / (2)) * (((skewness_ ** 2) /
          (((skewness_ ** 2) * ((r + 2) ** 2)) + (16 * (r + 1)))) ** 0.5))
    a1 = (a * (q1 - 1)) / ((q1 - 1) - (q2 + 1))
    a2 = (a * (q2 + 1)) / ((q1 - 1) - (q2 + 1))
    ye = (((q2 + 1) ** q2) * ((q1 - q2 - 2) ** (q1 - q2)) * (gamma(q1))) / \
        ((a) * ((q1 - 1) ** q1) * (gamma(q1 - q2 - 1)) * (gamma(q1 + 1)))

    return lambda z: ye * ((1 + z / a1) ** -q1) * ((1 + z / a2) ** q2)


def probability_density_function_4(sigma, skewness_, kurtosis_):

    y0 = 1 / (sigma * np.sqrt(np.pi * 2))

    return lambda z: y0 * np.exp((-(z ** 2)) / (2 * sigma ** 2))


def probability_density_function_5(sigma, skewness_, kurtosis_):

    m = (5 * kurtosis_ - 9) / (2 * (3 - kurtosis_))
    a = np.sqrt((2 * (sigma ** 2) * kurtosis_) / (3 - kurtosis_))
    y0 = (1 / (a * np.sqrt(np.pi))) * ((gamma(m + (3 / 2))) / (gamma(m)))

    return lambda z: y0 * ((1 - ((z ** 2) / (a ** 2))) ** m)


def probability_density_function_6(sigma, skewness_, kurtosis_):

    m = (5 * kurtosis_ - 9) / (2 * (kurtosis_ - 3))
    a = np.sqrt((2 * (sigma ** 2) * kurtosis_) / (kurtosis_ - 3))
    y0 = (1 / (a * np.sqrt(np.pi))) * ((gamma(m)) / (gamma(m - 0.5)))

    return lambda z: y0 * ((1 + ((z ** 2) / (a ** 2))) ** (-m))


def calculate_probability_density_function(skewness_, kurtosis_, sigma=1):

    kappa = calculate_kappa(skewness_, kurtosis_)

    if kappa < 0:
        print("Type = {}".format(1))
        return probability_density_function_1(sigma, skewness_, kurtosis_)

    elif 0 < kappa < 1:
        print("Type = {}".format(2))
        return probability_density_function_2(sigma, skewness_, kurtosis_)

    elif kappa > 1:
        print("Type = {}".format(3))
        return probability_density_function_3(sigma, skewness_, kurtosis_)

    elif kappa == 0 and skewness_ == 0 and kurtosis_ == 3:
        print("Type = {}".format(4))
        return probability_density_function_4(sigma, skewness_, kurtosis_)

    elif kappa == 0 and skewness_ == 0 and kurtosis_ < 3:
        print("Type = {}".format(5))
        return probability_density_function_5(sigma, skewness_, kurtosis_)

    elif kappa == 0 and skewness_ == 0 and kurtosis_ > 3:
        print("Type = {}".format(6))
        return probability_density_function_6(sigma, skewness_, kurtosis_)

def calculate_roughness(data):

    return np.sum(np.abs(data - np.mean(data))) / len(data)

def sample_pdf(pdf, x1=-4, x2=4, n1=100000, n2=100000):

    x = np.linspace(x1, x2, n1)
    p = pdf(x)
    p = np.nan_to_num(p)

    return np.random.choice(x, n2, p=p / np.sum(p))


def print_summary(pdf=None, data=None):

    if pdf is not None and data is None:

        data = sample_pdf(pdf=pdf)

    print("Roughness: {}".format(calculate_roughness(data)))
    print("Mean = {}".format(np.mean(data)))
    print("Median = {}".format(np.median(data)))
    print("Variance = {}".format(np.var(data)))
    print("Standard Deviation = {}".format(np.std(data)))
    print("Skewness = {}".format(skew(data)))
    print("Kurtosis = {}".format(kurtosis(data)))
    print("Min-Max = {}".format(np.min(data), np.max(data)))


def print_moments(pdf=None, data=None, n = 4):

    if pdf is not None and data is None:

        data = sample_pdf(pdf=pdf)

    for k in range(1, n + 1):

        print("{}th moment = {}".format(k, moment(data, k)))



def plot_pdf(pdf=None, data=None):

    if pdf is not None and data is None:

        data = sample_pdf(pdf=pdf)
                                                
    sns.histplot(data,
                      bins=500,
                      kde=True,
                      color='red',
                      fill=False,
                      stat='probability',
                      line_kws={'linewidth': 10})




