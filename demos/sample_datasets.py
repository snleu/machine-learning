# Using sample datasets from scikit learn library
# These data sets can be used as training data to test 

from sklearn import datasets
from sklearn.externals import joblib


# iris is a data set consisting of 4 pieces of information on 150 irises
# iris_data is a 2D array of (n_samples, n_features)
iris = datasets.load_iris()
clf_iris = joblib.load('clf_iris.pkl')

# digits is a series of 8x8 image arrays with grayscale values
# digits_data is an array of the images; each image is a handwritten digit and will be used to test recognition of handwritten digits
digits = datasets.load_digits()
clf_digits = joblib.load('clf_digits.pkl')

def svm_digits_demo():

	digits_test = digits.data[-1:]
	prediction = clf_digits.predict(digits_test)
	target = digits.target[-1]

	print 'prediciton of digit: ', prediction
	print 'labeled value of digit: ', target

def svm_irirs_demo():

	iris_test = iris.data[-1:]
	prediction = clf_iris.predict(iris_test)
	target = iris.target[-1]

	print 'prediciton of iris: ', prediction
	print 'labeled value of iris: ', target

svm_digits_demo()
svm_irirs_demo()

