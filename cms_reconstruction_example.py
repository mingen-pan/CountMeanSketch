import numpy as np


def example():
	np.random.seed(17)
	np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

	V_size = 3
	U_size = 5

	P = np.random.rand(U_size, V_size)
	P = P / P.sum(axis=0)
	print(P)
	print('rank: ', np.linalg.matrix_rank(P))

	Q = np.linalg.inv(P.T @ P) @ P.T
	print(Q)

	V_distribution = np.array([0.2, 0.3, 0.5])[:, np.newaxis]
	print((P @ V_distribution).T)
	print(Q @ (P @ V_distribution))

if __name__ == '__main__':
	example()
