import random
import numpy as np

PRIME_NUMBER = 18361375334787046697


def pairwise_hash_function(a_0, a_1, x):
	return (a_0 + a_1 * x) % PRIME_NUMBER


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

	def batch_decode(self, perturbed_object, queried_values):
		return self.p_inv[perturbed_object][queried_values]

	def decode(self, perturbed_object, queried_value):
		return self.p_inv[perturbed_object][queried_value]


class CountMeanSketchRandomizedResponseClient:
	def __init__(self, epsilon, dict_size, hash_range):
		"""
		Create a CMS+RR client
		:param epsilon: privacy guarantee factor
		:param dict_size: dictionary size (number of possible values)
		:param hash_range: the range a hash function will return, i.e., an integer of [0, hash_range)
		"""
		self.epsilon = epsilon
		self.dict_size = dict_size
		self.hash_range = hash_range
		self.rr = RandomizedResponse(epsilon, hash_range)

	def perturb(self, original_object):
		"""
		Randomly sample a hash function, hash the original value, and perturb the hashed value.
		:param original_object: the original value of an object
		:return the perturbed hashed value of the object, parameters of the hash function
		"""
		hash_param = self._generate_hash_param()
		hashed_object = pairwise_hash_function(*hash_param, original_object) % self.hash_range
		perturbed_hashed_object = self.rr.perturb(hashed_object)
		return perturbed_hashed_object, hash_param

	@staticmethod
	def _generate_hash_param():
		a_0 = random.randrange(1, PRIME_NUMBER)
		a_1 = random.randrange(1, PRIME_NUMBER)
		return a_0, a_1


class CountMeanSketchRandomizedResponseServer:
	def __init__(self, epsilon, dict_size, hash_range):
		"""
		Create a CMS+RR server
		:param epsilon: privacy guarantee factor
		:param dict_size: dictionary size (number of possible values)
		:param hash_range: the range a hash function will return, i.e., an integer of [0, hash_range)
		"""
		self.epsilon = epsilon
		self.dict_size = dict_size
		self.hash_range = hash_range
		self.rr = RandomizedResponse(epsilon, hash_range)
		self.perturbed_hashed_objects = []
		self.hash_params = []

	def reset(self):
		self.perturbed_hashed_objects = []
		self.hash_params = []

	def receive(self, perturbed_hashed_object, hash_param):
		"""
		Receive the output from a client
		:param perturbed_hashed_object: the perturbed hashed value of an object
		:param hash_param: parameters of the in-use hash function
		"""
		self.perturbed_hashed_objects.append(perturbed_hashed_object)
		self.hash_params.append(hash_param)

	# return the estimated frequencies of the input values
	def batch_query(self, values):
		"""
		Estimate the frequencies of the given values
		:param values: the values whose frequencies will be estimated
		:return an array of frequencies corresponding to the given values
		"""
		counts = np.zeros(len(values))
		for perturbed_hash_object, hash_param in zip(
				self.perturbed_hashed_objects, self.hash_params
		):
			def hash_func(x):
				return pairwise_hash_function(*hash_param, x) % self.hash_range

			hashed_values = list(map(hash_func, values))
			estimators = self.rr.batch_decode(perturbed_hash_object, hashed_values)
			counts += estimators
		return self.hash_range / (
				len(self.perturbed_hashed_objects) * (self.hash_range - 1)
		) * counts - 1 / (self.hash_range - 1)


def build_ocms_rr_optimized_for_l1l2(epsilon, dict_size):
	"""
	Build the server and client of CMS+RR optimized for the l1 / l2 losses
	:param epsilon: privacy guarantee factor
	:param dict_size: dictionary size (number of possible values)
	:return: server and client
	"""
	delta = np.exp(epsilon / 2) * np.sqrt(
		(np.exp(epsilon) + dict_size - 1)
		* ((dict_size - 1) * np.exp(epsilon) + 1)
	)
	hash_range = int(round(1 + delta / (np.exp(epsilon) + dict_size - 1)))
	return (CountMeanSketchRandomizedResponseServer(epsilon, dict_size, hash_range),
			CountMeanSketchRandomizedResponseClient(epsilon, dict_size, hash_range))


def build_ocms_rr_optimized_for_mse(epsilon, dict_size, f_star=1):
	"""
	Build the server and client of CMS+RR optimized for the worst-case MSE
	:param epsilon: privacy guarantee factor
	:param dict_size: dictionary size (number of possible values)
	:param f_star: upper bound of the maximum frequency among all values
	:return: server and client
	"""
	if f_star >= 0.5:
		hash_range = int(round(1 + np.exp(epsilon / 2)))
	else:
		e = np.exp(epsilon)
		delta = ((1 - f_star) * e + f_star) * (f_star * e + 1 - f_star)
		delta = np.sqrt(e * delta)
		hash_range = 1 + int(round(delta / (f_star * e + 1 - f_star)))
	return (CountMeanSketchRandomizedResponseServer(epsilon, dict_size, hash_range),
			CountMeanSketchRandomizedResponseClient(epsilon, dict_size, hash_range))


if __name__ == "__main__":
	original_objects = [0] * 4000 + [25] * 3000 + [50] * 2000 + [75] * 1000
	dict_size = 1000000
	epsilon = 3

	values_of_interest = list(range(100))
	original_frequencies = np.zeros(len(values_of_interest))
	original_frequencies[0] = 0.4
	original_frequencies[25] = 0.3
	original_frequencies[50] = 0.2
	original_frequencies[75] = 0.1

	server, client = build_ocms_rr_optimized_for_mse(epsilon, dict_size)
	for original_object in original_objects:
		perturbed_value, hash_param = client.perturb(original_object)
		server.receive(perturbed_value, hash_param)
	estimated_frequencies = server.batch_query(values_of_interest)
	print(estimated_frequencies[[0, 25, 50, 75]])
	print(np.sum(np.square(estimated_frequencies - original_frequencies)))

	for original_object in original_objects:
		perturbed_value, hash_param = client.perturb(original_object)
		server.receive(perturbed_value, hash_param)
	estimated_frequencies = server.batch_query(values_of_interest)
	print(estimated_frequencies[[0, 25, 50, 75]])
	print(np.max(np.square(estimated_frequencies - original_frequencies)))
