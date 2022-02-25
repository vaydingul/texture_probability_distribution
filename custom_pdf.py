from scipy.stats.sampling import SimpleRatioUniforms, NumericalInversePolynomial
import numpy as np

class CustomPDF(object):

	def __init__(self, pdf=None):

		self.pdf = pdf

	def __call__(self):

		self.rng = SimpleRatioUniforms(self, random_state = 42)


	def pdf(self, x: float) -> float:

		# note that the normalization constant isn't required

		return self.pdf(x)

	def sample(self, n: int = 10000) -> np.ndarray:

		return self.rng.rvs(n)
