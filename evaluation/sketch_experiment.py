import functools

import numpy as np
import collections
from sketch import (
    RecursiveHadamardResponse,
    OptimizedCountMeanSketch,
    HadamardSketch,
    CountMeanSketchHadamardEncoding,
)
import matplotlib.pyplot as plt
import matplotlib

COLORS = list(matplotlib.colors.TABLEAU_COLORS.values())

def get_zipf_distribution(a, b, distribution_size):
    frequencies = []
    for rank in range(1, distribution_size + 1):
        frequency = 1 / (rank + b) ** a
        frequencies.append(frequency)
    frequencies = np.array(frequencies)
    frequencies = frequencies / np.sum(frequencies)
    return frequencies


def generate_zipf_dataset(a, b, distribution_size, dataset_size, words=None):
    frequencies = get_zipf_distribution(a, b, distribution_size)
    if not words:
        words = list(range(distribution_size))
    assert len(words) == len(frequencies)

    dataset = []
    for word, frequency in zip(words, frequencies):
        count = int(round(dataset_size * frequency))
        if count < 1:
            count = 1
        if len(dataset) + count >= dataset_size:
            dataset += [word] * (dataset_size - len(dataset))
            break
        else:
            dataset += [word] * count
    return dataset


def parse_kosark():
    counter = collections.Counter()
    with open("kosarak_sequences.txt", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            keys = list(map(int, line[:-1].split(" ")))
            for key in keys:
                counter[key] += 1
    return counter


@functools.cache
def get_mini_kosark():
    counter = parse_kosark()
    mini_counter = collections.Counter()
    for k, v in counter.items():
        v = int(round(v / 100))
        if v == 0:
            continue
        mini_counter[k] = v
    # print(mini_counter)
    # print(sum(mini_counter.values()), len(mini_counter.keys()), min(mini_counter.keys()), max(mini_counter.keys()))
    #
    # first_1000_counts = 0
    # for i in range(1000):
    #     first_1000_counts += mini_counter[i - 2]
    # print(first_1000_counts / sum(mini_counter.values()))
    data_list = []
    for k, v in mini_counter.items():
        data_list += [k + 2] * v
    return data_list

def kosark_overview():
    counter = parse_kosark()
    total = sum(counter.values())
    print(total)
    sorted_pairs = sorted(
        list([(k, v / total) for k, v in counter.items()]), key=lambda x: -x[1]
    )
    for k, v in sorted_pairs[:100]:
        print(k, v)


def get_filename(sketch, epsilon, dataset_name):
    if isinstance(sketch, RecursiveHadamardResponse):
        class_name = "rhr"
    elif isinstance(sketch, OptimizedCountMeanSketch):
        if sketch.goal == OptimizedCountMeanSketch.OPTIMIZED_FOR_MSE:
            if sketch.f_star == 0:
                class_name = "ocms-f0"
            else:
                class_name = "ocms"
        elif sketch.goal == OptimizedCountMeanSketch.OPTIMIZED_FOR_L1L2:
            class_name = "l-ocms"
        else:
            raise ValueError("unknown optimization goal")
    elif isinstance(sketch, HadamardSketch):
        class_name = "he"
    elif isinstance(sketch, CountMeanSketchHadamardEncoding):
        class_name = "apple_cms"
    else:
        raise TypeError("sketch has an unknown type")
    return f"data/cms/{dataset_name}_{class_name}_eps_{epsilon}.csv"


def run_zipf_experiment_once(clazz, epsilon):
    dict_size = 1000000

    rhr_b = int(np.ceil(np.log2(np.exp(1)) * epsilon))
    rhr_base = 2 ** (rhr_b - 1)
    distribution_size = 100
    words = [i * rhr_base for i in range(distribution_size)]
    data_list = generate_zipf_dataset(2, 0, distribution_size, 10000, words=words)

    sketch = clazz(data_list, epsilon, dict_size)
    estimated_frequencies = sketch.batch_query(words)
    # print(estimated_frequencies)
    filename = get_filename(sketch, epsilon, "zipf")

    row = (
        ",".join(["{:.9f}".format(frequency) for frequency in estimated_frequencies])
        + "\n"
    )
    with open(filename, "a") as f:
        f.write(row)
    print('append one row: ', filename)


def compute_worst_case_mse(data, frequencies):
    mse = np.mean(np.square(data - frequencies), axis=0)
    return np.max(mse)


def fetch_worst_mse_by_epsilons(name, epsilons, filename_format: str, frequencies):
    result = []
    for epsilon in epsilons:
        filename = filename_format.format(name, epsilon)
        data = np.genfromtxt(filename, delimiter=",")
        mse = np.mean(np.square(data - frequencies), axis=0)
        max_mse = np.max(mse)
        result.append(max_mse)
    return result


def fetch_l1_l2_by_epsilons(name, epsilons, filename_format: str, frequencies):
    l1s = []
    l2s = []
    for epsilon in epsilons:
        filename = filename_format.format(name, epsilon)
        data = np.genfromtxt(filename, delimiter=",")
        l1 = np.mean(np.sum(np.abs(data - frequencies), axis=1))
        l2 = np.mean(np.sum(np.square(data - frequencies), axis=1))
        l1s.append(l1)
        l2s.append(l2)
    return l1s, l2s


def evaluate_zipf_mse():
    dataset_size = 10000
    distribution_size = 100
    frequencies = [0] * distribution_size
    data_list = generate_zipf_dataset(2, 0, distribution_size, dataset_size)
    for word in data_list:
        frequencies[word] += 1
    frequencies = np.array(frequencies, dtype=np.float64)
    frequencies /= np.sum(frequencies)

    epsilons = range(1, 6)
    filename_format = "data/cms/zipf_{0}_eps_{1}.csv"
    ocms = fetch_worst_mse_by_epsilons("ocms", epsilons, filename_format, frequencies)
    rhr = fetch_worst_mse_by_epsilons("rhr", epsilons, filename_format, frequencies)
    apple_cms = fetch_worst_mse_by_epsilons(
        "apple_cms", epsilons, filename_format, frequencies
    )
    he = fetch_worst_mse_by_epsilons("he", epsilons, filename_format, frequencies)

    calculated_ocms_ub = [
        np.max(
            calculate_ocms_mse(
                epsilon,
                len(data_list),
                frequencies,
                f_star=1,
            )
        )
        * max_mse_estimator_upper_bound_factor(100, 100)
        for epsilon in epsilons
    ]

    calculated_ocms = [
        np.max(
            calculate_ocms_mse(
                epsilon,
                len(data_list),
                frequencies,
                f_star=1,
            )
        )
        for epsilon in epsilons
    ]

    matplotlib.rcParams.update({"font.size": 18})
    plt.rcParams["text.usetex"] = True

    plt.plot(epsilons, ocms, c=COLORS[0], marker='o', label="OCMS")
    plt.plot(epsilons, rhr, c=COLORS[1], marker='^', label="RHR")
    plt.plot(epsilons, apple_cms, c=COLORS[2], marker='s', label="CMS+HE")
    plt.plot(epsilons, he, c=COLORS[3], marker='v', label="HE")

    plt.plot(epsilons, calculated_ocms, '--', c=COLORS[8], label="calculated OCMS")
    plt.plot(epsilons, calculated_ocms_ub, '--', c='y')

    plt.ylabel("worst-case MSE")
    plt.xlabel("privacy factor $\\epsilon$")
    plt.yscale("log")
    # plt.ylim([4e-6, 1.5e-3])
    plt.legend()
    plt.legend(fontsize=12)

    plt.savefig("images/cms/zipf_mse.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def evaluate_zipf_l1_l2():
    dataset_size = 10000
    distribution_size = 100
    frequencies = [0] * distribution_size
    data_list = generate_zipf_dataset(2, 0, distribution_size, dataset_size)
    for word in data_list:
        frequencies[word] += 1
    frequencies = np.array(frequencies, dtype=np.float64)
    frequencies /= np.sum(frequencies)

    epsilons = range(1, 6)
    filename_format = "data/cms/zipf_{0}_eps_{1}.csv"
    ocms_l1, ocms_l2 = fetch_l1_l2_by_epsilons(
        "l-ocms", epsilons, filename_format, frequencies
    )
    rhr_l1, rhr_l2 = fetch_l1_l2_by_epsilons(
        "rhr", epsilons, filename_format, frequencies
    )
    apple_cms_l1, apple_cms_l2 = fetch_l1_l2_by_epsilons(
        "apple_cms", epsilons, filename_format, frequencies
    )
    he_l1, he_l2 = fetch_l1_l2_by_epsilons("he", epsilons, filename_format, frequencies)

    calculated_l1s = []
    calculated_l2s = []
    for epsilon in range(1, 6):
        l1, l2 = calculate_ocms_l1l2(epsilon, dataset_size, 1000000, 100)
        calculated_l1s.append(l1)
        calculated_l2s.append(l2)


    matplotlib.rcParams.update({"font.size": 18})
    plt.rcParams["text.usetex"] = True
    plt.plot(epsilons, ocms_l1, marker='o', label="OCMS")
    plt.plot(epsilons, rhr_l1, marker='^', label="RHR")
    plt.plot(epsilons, apple_cms_l1, marker='s', label="CMS+HE")
    plt.plot(epsilons, he_l1, marker='v', label="HE")
    plt.plot(epsilons, calculated_l1s, '--', c=COLORS[8], label="calculated OCMS")

    plt.ylabel("$l_1$ loss")
    plt.xlabel("privacy factor $\\epsilon$")
    plt.yscale("log")
    plt.ylim([1e-1, 2.5])
    plt.legend()
    plt.legend(fontsize=12)
    plt.savefig("images/cms/zipf_l1.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    plt.plot(epsilons, ocms_l2, marker='o', label="OCMS")
    plt.plot(epsilons, rhr_l2, marker='^', label="RHR")
    plt.plot(epsilons, apple_cms_l2, marker='s', label="CMS+HE")
    plt.plot(epsilons, he_l2, marker='v', label="HE")
    plt.plot(epsilons, calculated_l2s, '--', c=COLORS[8], label="calculated OCMS")

    plt.ylabel("$l_2$ loss")
    plt.xlabel("privacy factor $\\epsilon$")
    plt.yscale("log")
    plt.legend()
    plt.legend(fontsize=12)
    plt.savefig("images/cms/zipf_l2.pdf", format="pdf", bbox_inches="tight")
    plt.show()


@functools.cache
def sample_normal_distribution():
    rand_state = np.random.get_state()
    np.random.seed(317)
    data_list = np.random.normal(1000, 50, 10000)
    data_list = [int(max(0, value)) for value in data_list]
    np.random.set_state(rand_state)
    return data_list


def run_norm_experiment_once(clazz, epsilon):
    dict_size = 10000

    data_list = sample_normal_distribution()
    sketch = clazz(data_list, epsilon, dict_size)
    estimated_frequencies = sketch.batch_query(list(range(10000)))
    # print(estimated_frequencies)
    filename = get_filename(sketch, epsilon, "norm")

    row = (
        ",".join(["{:.9f}".format(frequency) for frequency in estimated_frequencies])
        + "\n"
    )
    with open(filename, "a") as f:
        f.write(row)
    print('append one row: ', filename)


def calculate_ocms_mse(epsilon, dataset_size, frequencies, f_star=1):
    if f_star >= 0.5:
        m = int(round(1 + np.exp(epsilon / 2)))
    elif f_star == 0:
        m = int(round(1 + np.exp(epsilon)))
    else:
        raise ValueError("not implemented")
    var_rr_aa = np.exp(epsilon) * (m - 1) / (np.exp(epsilon) - 1) ** 2
    var_rr_ab = (np.exp(epsilon) + m - 2) / (np.exp(epsilon) - 1) ** 2
    A = (1 - frequencies) * (var_rr_aa + (m - 1) * var_rr_ab + (m - 1) / m)
    B = frequencies * (m * var_rr_aa)

    return m / (dataset_size * (m - 1) ** 2) * (A + B)

def calculate_ocms_l1l2(epsilon, dataset_size, dict_size, num_values_of_interest):
    delta = np.exp(epsilon / 2) * np.sqrt(
        (np.exp(epsilon) + dict_size - 1)
        * ((dict_size - 1) * np.exp(epsilon) + 1)
    )
    m = int(round(1 + delta / (np.exp(epsilon) + dict_size - 1)))

    var_rr_aa = np.exp(epsilon) * (m - 1) / (np.exp(epsilon) - 1) ** 2
    var_rr_ab = (np.exp(epsilon) + m - 2) / (np.exp(epsilon) - 1) ** 2

    A = (num_values_of_interest - 1) * (var_rr_aa + (m - 1) * var_rr_ab + (m - 1) / m)
    B =  (m * var_rr_aa)

    l2 = m / (dataset_size * (m - 1) ** 2) * (A + B)
    l1 = np.sqrt(num_values_of_interest * l2)

    return l1, l2


def max_mse_estimator_upper_bound_factor(size_of_batch_query, experiment_rounds):
    return 1 + 2 * (np.log(20 * size_of_batch_query) + np.sqrt(experiment_rounds * np.log(20 * size_of_batch_query))) / experiment_rounds


def evaluate_norm_mse():
    data_list = sample_normal_distribution()
    frequencies = [0] * 10000
    for value in data_list:
        frequencies[value] += 1
    frequencies = np.array(frequencies, dtype=np.float64)
    frequencies /= np.sum(frequencies)

    epsilons = range(1, 6)
    filename_format = "data/cms/norm_{0}_eps_{1}.csv"
    ocms = fetch_worst_mse_by_epsilons(
        "ocms-f0", epsilons, filename_format, frequencies
    )
    rhr = fetch_worst_mse_by_epsilons("rhr", epsilons, filename_format, frequencies)
    apple_cms = fetch_worst_mse_by_epsilons(
        "apple_cms", epsilons, filename_format, frequencies
    )
    he = fetch_worst_mse_by_epsilons("he", epsilons, filename_format, frequencies)

    calculated_ocms_ub = [
        np.max(
            calculate_ocms_mse(
                epsilon,
                len(data_list),
                frequencies,
                f_star=0,
            )
        )
        * max_mse_estimator_upper_bound_factor(10000, 100)
        for epsilon in epsilons
    ]

    calculated_ocms = [
        np.max(
            calculate_ocms_mse(
                epsilon,
                len(data_list),
                frequencies,
                f_star=0,
            )
        )
        for epsilon in epsilons
    ]

    matplotlib.rcParams.update({"font.size": 18})
    plt.rcParams["text.usetex"] = True
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.plot(epsilons, ocms, c=colors[0], marker='o', label="OCMS")
    plt.plot(epsilons, rhr, c=colors[1], marker='^', label="RHR")
    plt.plot(epsilons, apple_cms, c=colors[2], marker='s', label="CMS+HE")
    plt.plot(epsilons, he, c=colors[3], marker='v', label="HE")

    plt.plot(epsilons, calculated_ocms, '--', c='y', label="calculated OCMS")
    plt.plot(epsilons, calculated_ocms_ub, '--', c='y')

    plt.ylabel("worst-case MSE")
    plt.xlabel("privacy factor $\\epsilon$")
    plt.yscale("log")
    # plt.ylim([4e-6, 1.5e-3])
    plt.legend()
    plt.legend(fontsize=12)

    plt.savefig("images/cms/norm_mse.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def evaluate_norm_l1_l2():
    data_list = sample_normal_distribution()
    frequencies = [0] * 10000
    for value in data_list:
        frequencies[value] += 1
    frequencies = np.array(frequencies, dtype=np.float64)
    frequencies /= np.sum(frequencies)

    epsilons = range(1, 6)
    filename_format = "data/cms/norm_{0}_eps_{1}.csv"
    ocms_l1, ocms_l2 = fetch_l1_l2_by_epsilons(
        "ocms-f0", epsilons, filename_format, frequencies
    )
    rhr_l1, rhr_l2 = fetch_l1_l2_by_epsilons(
        "rhr", epsilons, filename_format, frequencies
    )
    apple_cms_l1, apple_cms_l2 = fetch_l1_l2_by_epsilons(
        "apple_cms", epsilons, filename_format, frequencies
    )
    he_l1, he_l2 = fetch_l1_l2_by_epsilons("he", epsilons, filename_format, frequencies)

    calculated_l1s = []
    calculated_l2s = []
    for epsilon in range(1, 6):
        l1, l2 = calculate_ocms_l1l2(epsilon, len(data_list), 10000, 10000)
        calculated_l1s.append(l1)
        calculated_l2s.append(l2)

    matplotlib.rcParams.update({"font.size": 18})
    plt.rcParams["text.usetex"] = True
    plt.plot(epsilons, ocms_l1, marker='o', label="OCMS")
    plt.plot(epsilons, rhr_l1, marker='^', label="RHR")
    plt.plot(epsilons, apple_cms_l1, marker='s', label="CMS+HE")
    plt.plot(epsilons, he_l1, marker='v', label="HE")
    plt.plot(epsilons, calculated_l1s, '--', color=COLORS[8], label="calculated OCMS")

    plt.ylabel("$l_1$ loss")
    plt.xlabel("privacy factor $\\epsilon$")
    plt.yscale("log")
    # plt.ylim([3e-1, 2.5])
    plt.legend(fontsize=12)
    plt.savefig("images/cms/norm_l1.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    plt.plot(epsilons, ocms_l2, marker='o', label="OCMS")
    plt.plot(epsilons, rhr_l2, marker='^', label="RHR")
    plt.plot(epsilons, apple_cms_l2, marker='s', label="CMS+HE")
    plt.plot(epsilons, he_l2, marker='v', label="HE")
    plt.plot(epsilons, calculated_l2s, '--', color=COLORS[8], label="calculated OCMS")

    plt.ylabel("$l_2$ loss")
    plt.xlabel("privacy factor $\\epsilon$")
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.savefig("images/cms/norm_l2.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def ocms_mse_histogram():
    epsilon = 3
    dataset_size = 10000
    distribution_size = 100
    frequencies = [0] * distribution_size
    data_list = generate_zipf_dataset(2, 0, distribution_size, dataset_size)
    for word in data_list:
        frequencies[word] += 1
    frequencies = np.array(frequencies, dtype=np.float64)
    frequencies /= np.sum(frequencies)

    calculated_ocms_mse = calculate_ocms_mse(epsilon, dataset_size, frequencies)

    filename = f"data/cms/zipf_ocms_eps_{epsilon}.csv"
    ocms = np.genfromtxt(filename, delimiter=",")
    mse = np.mean(np.square(ocms - frequencies), axis=0)
    plt.plot(range(distribution_size), mse)
    plt.plot(range(distribution_size), calculated_ocms_mse)
    plt.show()


def run_mini_kosarak_experiment_once(clazz, epsilon):
    dict_size = 26000
    data_list = get_mini_kosark()
    sketch = clazz(data_list, epsilon, dict_size)
    estimated_frequencies = sketch.batch_query(list(range(1000)))
    # print(estimated_frequencies)
    filename = get_filename(sketch, epsilon, "kosarak")

    row = (
        ",".join(["{:.9f}".format(frequency) for frequency in estimated_frequencies])
        + "\n"
    )
    with open(filename, "a") as f:
        f.write(row)
    print('append one row: ', filename)


def evaluate_kosarak_mse():
    data_list = get_mini_kosark()
    frequencies = [0] * 1000
    for value in data_list:
        if value >= 1000:
            continue
        frequencies[value] += 1
    frequencies = np.array(frequencies, dtype=np.float64)
    frequencies /= len(data_list)

    epsilons = range(1, 6)
    filename_format = "data/cms/kosarak_{0}_eps_{1}.csv"
    ocms = fetch_worst_mse_by_epsilons(
        "ocms", epsilons, filename_format, frequencies
    )
    rhr = fetch_worst_mse_by_epsilons("rhr", epsilons, filename_format, frequencies)
    apple_cms = fetch_worst_mse_by_epsilons(
        "apple_cms", epsilons, filename_format, frequencies
    )
    he = fetch_worst_mse_by_epsilons("he", epsilons, filename_format, frequencies)

    calculated_ocms = [
        np.max(
            calculate_ocms_mse(
                epsilon,
                len(data_list),
                frequencies,
            )
        )
        for epsilon in epsilons
    ]

    calculated_ocms_ub = [
        np.max(
            calculate_ocms_mse(
                epsilon,
                len(data_list),
                frequencies,
            )
        )
        * max_mse_estimator_upper_bound_factor(1000, 100)
        for epsilon in epsilons
    ]

    matplotlib.rcParams.update({"font.size": 18})
    plt.rcParams["text.usetex"] = True
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.plot(epsilons, ocms, c=colors[0], marker='o', label="OCMS")
    plt.plot(epsilons, rhr, c=colors[1], marker='^', label="RHR")
    plt.plot(epsilons, apple_cms, c=colors[2], marker='s', label="CMS+HE")
    plt.plot(epsilons, he, c=colors[3], marker='v', label="HE")

    plt.plot(epsilons, calculated_ocms, '--', c='y', label="calculated OCMS")
    plt.plot(epsilons, calculated_ocms_ub, '--', c='y')

    plt.ylabel("worst-case MSE")
    plt.xlabel("privacy factor $\\epsilon$")
    plt.yscale("log")
    # plt.ylim([4e-6, 1.5e-3])
    plt.legend()
    plt.legend(fontsize=12)

    plt.savefig("images/cms/kosarak_mse.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def evaluate_kosarak_l1_l2():
    data_list = get_mini_kosark()
    frequencies = [0] * 1000
    for value in data_list:
        if value >= 1000:
            continue
        frequencies[value] += 1
    frequencies = np.array(frequencies, dtype=np.float64)
    frequencies /= len(data_list)

    epsilons = range(1, 6)
    filename_format = "data/cms/kosarak_{0}_eps_{1}.csv"
    ocms_l1, ocms_l2 = fetch_l1_l2_by_epsilons(
        "l-ocms", epsilons, filename_format, frequencies
    )
    rhr_l1, rhr_l2 = fetch_l1_l2_by_epsilons(
        "rhr", epsilons, filename_format, frequencies
    )
    apple_cms_l1, apple_cms_l2 = fetch_l1_l2_by_epsilons(
        "apple_cms", epsilons, filename_format, frequencies
    )
    he_l1, he_l2 = fetch_l1_l2_by_epsilons("he", epsilons, filename_format, frequencies)

    calculated_l1s = []
    calculated_l2s = []
    for epsilon in range(1, 6):
        l1, l2 = calculate_ocms_l1l2(epsilon, len(data_list), 26000, 1000)
        calculated_l1s.append(l1)
        calculated_l2s.append(l2)

    matplotlib.rcParams.update({"font.size": 18})
    plt.rcParams["text.usetex"] = True
    plt.plot(epsilons, ocms_l1, marker='o', label="OCMS")
    plt.plot(epsilons, rhr_l1, marker='^', label="RHR")
    plt.plot(epsilons, apple_cms_l1, marker='s', label="CMS+HE")
    plt.plot(epsilons, he_l1, marker='v', label="HE")
    plt.plot(epsilons, calculated_l1s, '--', color=COLORS[8], label="calculated OCMS")

    plt.ylabel("$l_1$ loss")
    plt.xlabel("privacy factor $\\epsilon$")
    plt.yscale("log")
    plt.ylim([3e-1, 2.5])
    plt.legend(fontsize=12)
    plt.savefig("images/cms/kosarak_l1.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    plt.plot(epsilons, ocms_l2, marker='o', label="OCMS")
    plt.plot(epsilons, rhr_l2, marker='^', label="RHR")
    plt.plot(epsilons, apple_cms_l2, marker='s', label="CMS+HE")
    plt.plot(epsilons, he_l2, marker='v', label="HE")
    plt.plot(epsilons, calculated_l2s, '--', color=COLORS[8], label="calculated OCMS")

    plt.ylabel("$l_2$ loss")
    plt.xlabel("privacy factor $\\epsilon$")
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.savefig("images/cms/kosarak_l2.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def run_kosarak():
    for epsilon in range(1, 6):
        for _ in range(100):
            run_mini_kosarak_experiment_once(OptimizedCountMeanSketch, epsilon)
            run_mini_kosarak_experiment_once(OptimizedCountMeanSketch.optimize_l1l2, epsilon)
            run_mini_kosarak_experiment_once(RecursiveHadamardResponse, epsilon)
            run_mini_kosarak_experiment_once(HadamardSketch, epsilon)
            run_mini_kosarak_experiment_once(CountMeanSketchHadamardEncoding.apple_cms, epsilon)


if __name__ == "__main__":
    evaluate_norm_mse()
    evaluate_norm_l1_l2()






