from sketch import HadamardSketch, HadamardEncoding, OptimizedCountMeanSketch, hadamard_row_hashing
from sketch_experiment import calculate_ocms_l1l2
import numpy as np


def testHadamardMatrix():
	h_size = 8
	matrix = [[0] * h_size for _ in range(h_size)]
	for i in range(h_size):
		for j in range(h_size):
			v = HadamardSketch.get_hadamard_element(i, j)
			matrix[i][j] = v
	matrix = np.array(matrix)
	print(matrix)
	print(np.sum(matrix, axis=0))


def testHadamardEncoding():
	he = HadamardEncoding(1, 10)
	x_list = list(range(10))
	counters = np.zeros(10)
	data_list = [0] * 10000 + [5] * 20000 + [8] * 30000
	for value in data_list:
		res = he.perturb(value)
		counters += he.batch_decode(res, x_list)
	print(counters / len(data_list))

	print(np.sqrt(1/10 / len(data_list)))


def testHadamardEncoding2():
	he = HadamardEncoding(1, 10)
	counter = 0
	data_list = [0] * 10000
	print(he.keep_prob)
	for value in data_list:
		res = he.perturb(value)
		hashed_v, row = res
		counter += (hashed_v == hadamard_row_hashing(row, 0))
	print(counter, counter / len(data_list))

def test_verify_ocms_mse():
	data_list = [0] * 10000 + [1] * 20000 + [2] * 30000
	true_f = [0.0] * 10
	true_f[0] = 1/6
	true_f[1] = 2/6
	true_f[2] = 3/6
	epsilon = 10

	print('expected var: ', np.exp(epsilon / 2) / (len(data_list) * (np.exp(epsilon / 2) - 1)**2) )
	mse = np.zeros(10)
	epoches = 100
	for _ in range(epoches):
		sketch = OptimizedCountMeanSketch(data_list, epsilon, 100000)
		estimators = sketch.batch_query(list(range(10)))
		mse += np.square((estimators - true_f) )
	mse /= epoches
	print(mse)

def test_ocms_hash_range():
	sketch = OptimizedCountMeanSketch([], np.log(4), 100000, f_star=0)
	print(sketch.hash_range)

	sketch = OptimizedCountMeanSketch([], np.log(4), 100000, f_star=1)
	print(sketch.hash_range)


def test_calculated_ocms_l1l2():
	epsilon = 1
	calculated = []
	approxs = []
	for d in [10, 100, 1000, 10000]:
		approx = 4 * d * np.exp(epsilon) / (np.exp(epsilon) - 1) ** 2
		approxs.append(approx)
		l1, l2 = calculate_ocms_l1l2(epsilon, 1, d, d)
		calculated.append(l2)

	print('')
	print(calculated)
	print(approxs)