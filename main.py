import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename):
    return np.genfromtxt(filename, delimiter=',')

def make_plot(data, title, b0, b1):
    x_vals = np.zeros(len(data))
    y_vals = np.zeros(len(data))

    x = np.linspace(-2, 2, 100)
    y = b0 + b1*x

    for i in range(len(data)):
        x_vals[i] = data[i][0]
        y_vals[i] = data[i][1]

    plt.plot(x_vals, y_vals, 'bo')
    plt.plot(x, y, 'r', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()

def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat

def main():
    syn1 = read_csv('./data/synthetic-1.csv')
    syn2 = read_csv('./data/synthetic-2.csv')
    syn3 = read_csv('./data/synthetic-3.csv')

    coef = [0.4, 0.8]

    make_plot(syn1, "Synthetic 1", coef[0], coef[1])
    #make_plot(syn2, "Synthetic 2")
    #make_plot(syn3, "Synthetic 3")

    for row in syn1:
        yhat = predict(row, coef)
        print("Expected=%.3f, Predicted=%.3f" % (row[-1], yhat))

if __name__== "__main__": main()
