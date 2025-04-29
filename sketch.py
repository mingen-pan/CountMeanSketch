import random
import numpy as np
import scipy
import os, sys


def generate_rappor_response(bit_id, partition_size, epsilon):
	e = np.exp(epsilon) / 2
	prob = 1 / (e + 1)
	arr = np.random.binomial(1, prob, partition_size)
	arr[bit_id] = 1 - arr[bit_id]
	return arr


class Sketch:
	def __init__(
		self,
		data_list,
		hash_function_generator,
		hash_range,
		ldp_process,
		collision_prob=None,
	):
		self.data_list = data_list
		self.hash_function_generator = hash_function_generator
		self.hash_range = hash_range
		self.ldp_process = ldp_process
		self.collision_prob = collision_prob
		self.perturbed_values = []
		self.assigned_hash_functions = []
		self.prepare()

	def process_object(self, original_value):
		hash_func = self.hash_function_generator()
		hashed_value = hash_func(original_value)
		perturbed_value = self.ldp_process.perturb(hashed_value)
		return perturbed_value, hash_func

	def prepare(self):
		for original_value in self.data_list:
			perturbed_value, hash_func = self.process_object(original_value)
			self.perturbed_values.append(perturbed_value)
			self.assigned_hash_functions.append(hash_func)

	def batch_query(self, x_list):
		counts = np.zeros(len(x_list))
		for perturbed_value, hash_func in zip(
			self.perturbed_values, self.assigned_hash_functions
		):
			hashed_x_list = list(map(hash_func, x_list))
			estimators = self.ldp_process.batch_decode(perturbed_value, hashed_x_list)
			counts += estimators
		if self.collision_prob:
			return counts / (
				len(self.data_list) * (1 - self.collision_prob)
			) - self.collision_prob / (1 - self.collision_prob)
		else:
			return self.hash_range / (
				len(self.data_list) * (self.hash_range - 1)
			) * counts - 1 / (self.hash_range - 1)

	def query(self, x):
		count = 0
		for perturbed_value, hash_func in zip(
			self.perturbed_values, self.assigned_hash_functions
		):
			hashed_x = hash_func(x)
			count += self.ldp_process.decode(perturbed_value, x)
		if self.collision_prob:
			return count / (
				len(self.data_list) * (1 - self.collision_prob)
			) - self.collision_prob / (1 - self.collision_prob)
		else:
			return self.hash_range / (
				len(self.data_list) * (self.hash_range - 1)
			) * count - 1 / (self.hash_range - 1)


class RandomizedResponse:
	def __init__(self, epsilon, m):
		self.epsilon = epsilon
		self.m = m
		probs = np.ones((m, m))
		e = np.exp(epsilon)
		np.fill_diagonal(probs, e)
		self.probs = probs / (e + m - 1)
		self.p_inv = np.linalg.inv(self.probs)
		self.candidates = list(range(m))

	def perturb(self, value):
		return np.random.choice(self.candidates, p=self.probs[value])

	def batch_decode(self, perturbed_value, x_list):
		return self.p_inv[perturbed_value][x_list]

	def decode(self, perturbed_value, x):
		return self.p_inv[perturbed_value][x]


def get_hadamard_element(row, col):
	num = row & col
	return 1 - num.bit_count() % 2


def hadamard_row_hashing(row, value):
	return get_hadamard_element(row, value + 1)


class HadamardEncoding:
	def __init__(self, epsilon, m):
		self.epsilon = epsilon
		self.h_size = 2 ** int(np.ceil(np.log2(m + 1)))
		probs = np.ones((2, 2))
		e = np.exp(epsilon)
		np.fill_diagonal(probs, e)
		self.keep_prob = e / (e + 1)
		self.p_inv = np.array([e / (e - 1), -1 / (e - 1)])

	def perturb(self, value):
		row = random.randrange(self.h_size)
		hashed_value = hadamard_row_hashing(row, value)
		if random.uniform(0, 1) > self.keep_prob:
			hashed_value = 1 - hashed_value
		return hashed_value, row

	def batch_decode(self, perturbed_struct, x_list):
		hashed_value, row = perturbed_struct

		estimators = []
		for x in x_list:
			hashed_x = hadamard_row_hashing(row, x)
			if hashed_x == hashed_value:
				estimator = self.p_inv[0]
			else:
				estimator = self.p_inv[1]
			estimators.append(2 * estimator - 1)
		return estimators

	def decode(self, perturbed_struct, x):
		return self.batch_decode(perturbed_struct, [x])[0]


class HadamardSketch(Sketch):
	def __init__(self, data_list, epsilon, dict_size):
		self.rr = RandomizedResponse(epsilon, 2)
		self.dict_size = dict_size
		self.h_size = 2 ** int(np.ceil(np.log2(dict_size + 1)))
		super().__init__(data_list, self.hash_function_generator, 2, self.rr)

	def hash_function_generator(self):
		row = random.randrange(self.h_size)
		hash_func = lambda col: hadamard_row_hashing(row, col)
		return hash_func


PRIME_NUMBER = 18361375334787046697


def pairwise_hash_function(a_0, a_1, x):
	return (a_0 + a_1 * x) % PRIME_NUMBER


class CountMeanSketchHadamardEncoding(Sketch):
	def __init__(self, data_list, epsilon, dict_size, hashing_family_size, hash_range):
		self.he = HadamardEncoding(epsilon, hash_range)
		self.dict_size = dict_size
		self.hashing_family_size = hashing_family_size
		self.hash_functions = self.sample_all_hash_functions()
		super().__init__(data_list, self.hash_function_generator, hash_range, self.he)

	def sample_hash_function(self):
		a_0 = random.randrange(1, PRIME_NUMBER)
		a_1 = random.randrange(1, PRIME_NUMBER)
		return lambda x: pairwise_hash_function(a_0, a_1, x) % self.hash_range

	def sample_all_hash_functions(self):
		hash_functions = []
		for _ in range(self.hashing_family_size):
			hash_functions.append(self.sample_hash_function())
		return hash_functions

	def hash_function_generator(self):
		return self.hash_functions[random.randrange(self.hashing_family_size)]

	@classmethod
	def apple_cms(cls, data_list, epsilon, dict_size):
		return cls(data_list, epsilon, dict_size, 2048, 1024)


class OptimizedCountMeanSketch(Sketch):
	OPTIMIZED_FOR_MSE = 1
	OPTIMIZED_FOR_L1L2 = 2

	def __init__(self, data_list, epsilon, dict_size, goal=OPTIMIZED_FOR_MSE, f_star=1):
		self.dict_size = dict_size
		if goal == self.OPTIMIZED_FOR_MSE:
			hash_range = self.compute_hash_range_mse(epsilon, f_star)
		elif goal == self.OPTIMIZED_FOR_L1L2:
			delta = np.exp(epsilon / 2) * np.sqrt(
				(np.exp(epsilon) + dict_size - 1)
				* ((dict_size - 1) * np.exp(epsilon) + 1)
			)
			hash_range = int(round(1 + delta / (np.exp(epsilon) + dict_size - 1)))
		else:
			raise ValueError('unsupported optimization goal')
		self.goal = goal
		self.rr = RandomizedResponse(epsilon, hash_range)
		self.f_star = f_star
		super().__init__(data_list, self.hash_function_generator, hash_range, self.rr)

	@staticmethod
	def compute_hash_range_mse(epsilon, f_star):
		if f_star >= 0.5:
			return int(round(1 + np.exp(epsilon / 2)))
		e = np.exp(epsilon)
		delta = ((1 - f_star) * e + f_star) * (f_star * e + 1 - f_star)
		delta = np.sqrt(e * delta)
		return 1 + int(round(delta / (f_star * e + 1 - f_star)))

	def hash_function_generator(self):
		a_0 = random.randrange(1, PRIME_NUMBER)
		a_1 = random.randrange(1, PRIME_NUMBER)
		return lambda x: pairwise_hash_function(a_0, a_1, x) % self.hash_range

	@classmethod
	def optimize_l1l2(cls, data_list, epsilon, dict_size):
		return cls(data_list, epsilon, dict_size, goal=cls.OPTIMIZED_FOR_L1L2)

	@classmethod
	def optimize_mse_f0(cls, data_list, epsilon, dict_size):
		return cls(data_list, epsilon, dict_size, goal=cls.OPTIMIZED_FOR_MSE, f_star=0)


class RecursiveHadamardResponse(Sketch):
	def __init__(self, data_list, epsilon, dict_size):
		self.dict_size = dict_size
		self.b = int(np.ceil(np.log2(np.exp(1)) * epsilon))
		self.base = 2 ** (self.b - 1)
		self.h_size = 2 ** int(np.ceil(np.log2(dict_size // self.base + 1)))
		hash_range = 2**self.b
		self.rr = RandomizedResponse(epsilon, hash_range)
		super().__init__(data_list, self.hash_function_generator, hash_range, self.rr)

	def hash_function(self, row, value):
		a = value // self.base
		b = value % self.base
		hashed = hadamard_row_hashing(row, a)
		return b * 2 + hashed

	def hash_function_generator(self):
		row = random.randrange(self.h_size)
		return lambda x: self.hash_function(row, x)

	@staticmethod
	def flip_last_bit(x):
		x_front = x // 2
		x_last = x % 2
		return x_front * 2 + (1 - x_last)

	def query(self, x):
		collision_count = 0
		same_a_count = 0
		for perturbed_value, hash_func in zip(
			self.perturbed_values, self.assigned_hash_functions
		):
			hashed_x = hash_func(x)
			collision_estimator = self.ldp_process.decode(perturbed_value, hashed_x)
			diff_last_bit_estimator = self.ldp_process.decode(
				perturbed_value, self.flip_last_bit(hashed_x)
			)
			collision_count += collision_estimator
			same_a_count += collision_estimator + diff_last_bit_estimator

		return 1 / len(self.data_list) * (2 * collision_count - same_a_count)

	def batch_query(self, x_list):
		collision_counts = np.zeros(len(x_list))
		same_a_counts = np.zeros(len(x_list))
		for perturbed_value, hash_func in zip(
			self.perturbed_values, self.assigned_hash_functions
		):
			hashed_x_list = list(map(hash_func, x_list))
			flip_bit_hashed_x_list = list(map(self.flip_last_bit, hashed_x_list))
			collision_estimators = self.ldp_process.batch_decode(
				perturbed_value, hashed_x_list
			)
			diff_last_bit_estimators = self.ldp_process.decode(
				perturbed_value, flip_bit_hashed_x_list
			)
			collision_counts += collision_estimators
			same_a_counts += collision_estimators + diff_last_bit_estimators
		return 1 / len(self.data_list) * (2 * collision_counts - same_a_counts)


def get_new_seed():
	return int.from_bytes(os.urandom(16), sys.byteorder)


if __name__ == "__main__":
	data_list = [0] * 10000 + [1] * 20000 + [2] * 30000
	epsilon = 5
	dict_size = 100000

	# idx = np.array([129, 192, 257, 320, 385, 448, 513, 576, 641, 704, 769, 832, 897, 960])
	idx = list(range(100))

	sketch = OptimizedCountMeanSketch(data_list, epsilon, dict_size)
	print(sketch.batch_query(idx))

	sketch = RecursiveHadamardResponse(data_list, epsilon, dict_size)
	print(sketch.batch_query(idx))

