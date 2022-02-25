from scipy.stats.sampling import SimpleRatioUniforms
import numpy as np

class CustomPDF(object):

	def __init__(self, pdf=None):

		self.pdf = pdf

	def pdf(self, x: float) -> float:

		# note that the normalization constant isn't required

		return self.pdf(x)

	def sample(self, n: int) -> np.ndarray:

		self.rng = SimpleRatioUniforms(self.pdf)
		return self.rng.rvs(n)
