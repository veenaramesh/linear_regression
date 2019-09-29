# NOTE: You may not use the library Linear Regression, but implement your own!
# REMEMBER to place self.attribute = 0 with value from your implementation

class MyLinearRegression:
  """
  Define what a linear regressor can do
  """
  def __init__ (self, learning_rate=0.005):
    """
    Initialize the regressor
    """
    self.theta = 0; # parameter vector;
    self.alpha = learning_rate; # learning rate
    self.cost  = 0; # cost function

  def gradientDescent(self, X_train, y_train, theta, alpha, iters):
    """
    Implementatation of the gradient descent
    INPUT:
    alpha: the learning rate
    iters: number of iterations

    OUTPUT:
    theta: updated value for theta
    cost: value of the cost function
    """
    # implementation code here
    X = np.concatenate([np.ones((len(X_train), 1)), X_train], axis=1)
    transpose = X.transpose()
    costs = []
    weights = np.zeros(X.shape[1], dtype=int)

    for i in range(iters):
      h = np.dot(X, weights)  # find the hypothesis/predictions
      c = h - y_train
      g = np.dot(transpose, c)
      g = g / len(y_train)
      weights = weights - alpha*g

      current_cost = np.absolute(c).sum()
      costs.append(current_cost)

    return weights, costs

  def fitUsingGradientDescent(self, X_train, y_train):
    """
    Train the regressor using gradient descent
    """
    # implementation code here

    t, c = self.gradientDescent(X_train, y_train, self.theta, self.alpha, iters=1000)
    self.theta = t
    self.cost = c

    return self

  def fitUsingNormalEquation(self, X_train, y_train):
    """
    Training using the Normal (close form) equation
    """
    # implementation code here for Task 4.

    X = np.concatenate([np.ones((len(X_train), 1)), X_train], axis=1)
    transpose_x = X.transpose()
    optimal_theta = np.dot(np.linalg.inv(np.dot(transpose_x, X)), np.dot(transpose_x, y_train))
    self.theta = optimal_theta

    return self

  def fit(self, X_train, y_train, model_type="gd"):
    # to make Pipeline work smoooothly
    if model_type == "gd":
      return self.fitUsingGradientDescent(X_train, y_train)
    else:
      return self.fitUsingNormalEquation(X_train, y_train)

  def predict(self, X_test):
    """
    Predicting the label
    """
    # implementation code here

    X = np.concatenate([np.ones((len(X_test), 1)), X_test], axis=1)
    y_predict = np.dot(X, self.theta)

    return y_predict

  def __str__(self):
    """
    Print out the parameter out when call print()
    """
    return("Parameter vector is %f" % self.theta)
