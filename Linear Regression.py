from numpy import random, array, dot, sum, ndarray

class LinearRegression:
	def __init__(self, learning_rate:float = 0.01, max_iterations:int = int (1e6)):
		self.learning_rate = learning_rate
		self.max_iterations = max_iterations
		self.coefficient = None
		self.intercept = None

	def fit(self, x: ndarray, y: ndarray):
		n_samples, n_features = x.shape
		self.coefficient = random.rand(n_features)
		self.intercept = 0

		for i in range(self.max_iterations):
			y_pred = dot(x, self.coefficient) + self.intercept

			gradient_coefficient = (1 / n_samples) * dot(x.T, (y_pred - y))
			gradient_intercept = (1 / n_samples) * sum(y_pred - y)

			self.coefficient -= self.learning_rate * gradient_coefficient
			self.intercept -= self.learning_rate * gradient_intercept

		return self

	def predict(self, x: ndarray):
		return dot(x, self.coefficient) + self.intercept


feature = array([[1], [2], [3], [4], [5]])
target = array([3, 5, 7, 9, 11])
model = LinearRegression(learning_rate = 0.01, max_iterations = int (1e6))
model.fit(feature, target)
print("Weights:", model.coefficient)
print("Bias:", model.intercept)
test_data = array([[6], [7], [8]])
prediction = model.predict(test_data)
print(prediction)
