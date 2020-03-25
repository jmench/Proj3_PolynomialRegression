import numpy as np
import matplotlib.pyplot as plt

# read csv file into np array
def read_csv(filename):
    return np.genfromtxt(filename, delimiter=',')

# plot all the data points and the corresponding line
def make_plot(data, title, coef):
    x_vals = np.zeros(len(data))
    y_vals = np.zeros(len(data))

    x = np.linspace(-2, 2, 100)
    y = 0
    for i in range(len(coef)):
        y += coef[i] * (x**i)

    for i in range(len(data)):
        x_vals[i] = data[i][0]
        y_vals[i] = data[i][1]

    plt.plot(x_vals, y_vals, 'bo')
    plt.plot(x, y, 'r', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()

def get_error(coef, row):
    #set prediciton to intercept value to begin
    prediction = 0
    x = row[0]
    actual = row[1]
    for i in range(len(coef)):
        prediction += coef[i] * (x ** i)
    error = prediction - actual
    return error

def update_coef(data, learn_rate, epochs, coef):
    error_arr = np.zeros(epochs)
    m = 1/len(data)
    for epoch in range(epochs):
        sum_error = 0
        for row in data:
            # Get predicted value here
            err = get_error(coef, row)
            #update all coef
            for i in range(len(coef)):
                coef[i] = coef[i] - learn_rate * (1/m) * err * (row[0]**i)
            sum_error += err**2
        error_arr[epoch] = sum_error
    mse = round(np.mean(error_arr), 3)
    print('The MSE of this model is ' + str(mse))
    return coef

def main():
    syn1 = read_csv('./data/synthetic-1.csv')
    syn2 = read_csv('./data/synthetic-2.csv')
    syn3 = read_csv('./data/synthetic-3.csv')

    coef1 = [0.0, 0.0]
    coef2 = [0.0, 0.0, 0.0]
    coef4 = [0.0, 0.0, 0.0, 0.0, 0.0]
    coef7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    coef_arr = [[coef1, '1st order'], [coef2, '2nd order'], [coef4, '4th order'], [coef7, '7th order']]

    for i in range(len(coef_arr)):
        if i < 3:
            alpha = .00001
        else:
            alpha = .000001
        coefs = update_coef(syn1, alpha, 100, coef_arr[i][0])
        print('Weight results for syn1 and ' + coef_arr[i][1] + ':')
        weights = np.array(coefs)
        np.around(weights, 3)
        print(weights)
        print()
        make_plot(syn1, "Synthetic 1 - " + coef_arr[i][1], weights)

        coefs = update_coef(syn2, alpha, 100, coef_arr[i][0])
        print('Weight results for syn2 and ' + coef_arr[i][1] + ':')
        weights = np.array(coefs)
        np.around(weights, 3)
        print(weights)
        print()
        make_plot(syn2, "Synthetic 2 - " + coef_arr[i][1], weights)

        coefs = update_coef(syn3, alpha, 100, coef_arr[i][0])
        print('Weight results for syn3  and ' + coef_arr[i][1] + ':')
        weights = np.array(coefs)
        np.around(weights, 3)
        print(weights)
        print()
        make_plot(syn3, "Synthetic 3 - " + coef_arr[i][1], weights)

if __name__== "__main__": main()
