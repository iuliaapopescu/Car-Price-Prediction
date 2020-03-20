import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

# define models
linear_regression_model = LinearRegression()

# load training data
training_data = np.load('training_data.npy')
prices = np.load('prices.npy')

# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)


# 3 - fold => 3 samples per fold
num_samples_fold = len(training_data) // 3

# Split train in 3 folds
training_data_1, prices_1 = training_data[:num_samples_fold], \
                            prices[:num_samples_fold]

training_data_2, prices_2 = training_data[num_samples_fold: 2 * num_samples_fold], \
                            prices[num_samples_fold: 2 * num_samples_fold]

training_data_3, prices_3 = training_data[2 * num_samples_fold:], \
                            prices[2 * num_samples_fold:]


def normalize(train_data, test_data):
    scaler = preprocessing.StandardScaler()

    scaler.fit(train_data)
    scaler_train = scaler.transform(train_data)
    scaler_test = scaler.transform(test_data)

    return scaler_train, scaler_test


def step(train_data, train_labels, test_data, test_labels, model):

    norm_train, norm_test = normalize(train_data, test_data)

    reg = model.fit(norm_train, train_labels)
    predict = reg.predict(norm_test)
    mae = mean_absolute_error(test_labels, predict)
    mse = mean_squared_error(test_labels, predict)

    return mae, mse


model = linear_regression_model

# Run 1

mae_1, mse_1 = step(np.concatenate((training_data_1, training_data_3)),
                    np.concatenate((prices_1, prices_3)),
                    training_data_2,
                    prices_2,
                    model)

# Run 2

mae_2, mse_2 = step(np.concatenate((training_data_1, training_data_2)),
                    np.concatenate((prices_1, prices_2)),
                    training_data_3,
                    prices_3,
                    model)

# Run 3

mae_3, mse_3 = step(np.concatenate((training_data_2, training_data_3)),
                    np.concatenate((prices_2, prices_3)),
                    training_data_1,
                    prices_1,
                    model)

# Mean mean_absolute_error and mean mean_squared_error for linear regression

print("Values for linear regression are: \n")

mean_mae = (mae_1 + mae_2 + mae_3) / 3
mean_mse = (mse_1 + mse_2 + mse_3) / 3

print("Mean absolute error 1 is: %f" % mae_1)
print("Mean absolute error 2 is: %f" % mae_2)
print("Mean absolute error 3 is: %f" % mae_3)

print("Mean squared error 1 is: %f" % mse_1)
print("Mean squared error 2 is: %f" % mse_2)
print("Mean squared error 3 is: %f" % mse_3)

print("Overall mean absolute error is: %f" % mean_mae)
print("Overall mean squared error is: %f\n" % mean_mse)


for alpha_ in [1, 10, 100, 1000]:
    model = Ridge(alpha=alpha_)

    print("Values for ridge regression with alpha = %d are: \n" % alpha_)

    # Run 1

    mae_1, mse_1 = step(np.concatenate((training_data_1, training_data_3)),
                        np.concatenate((prices_1, prices_3)),
                        training_data_2,
                        prices_2,
                        model)

    # Run 2

    mae_2, mse_2 = step(np.concatenate((training_data_1, training_data_2)),
                        np.concatenate((prices_1, prices_2)),
                        training_data_3,
                        prices_3,
                        model)

    # Run 3

    mae_3, mse_3 = step(np.concatenate((training_data_2, training_data_3)),
                        np.concatenate((prices_2, prices_3)),
                        training_data_1,
                        prices_1,
                        model)

    mean_mae = (mae_1 + mae_2 + mae_3) / 3
    mean_mse = (mse_1 + mse_2 + mse_3) / 3

    print("Mean absolute error 1 is: %f" % mae_1)
    print("Mean absolute error 2 is: %f" % mae_2)
    print("Mean absolute error 3 is: %f" % mae_3)

    print("Mean squared error 1 is: %f" % mse_1)
    print("Mean squared error 2 is: %f" % mse_2)
    print("Mean squared error 3 is: %f" % mse_3)

    print("Overall mean absolute error is: %f" % mean_mae)
    print("Overall mean squared error is: %f\n" % mean_mse)


print("Ridge regression best performs with alpha = 10.\n")
print("Rapport for ridge regression of parameter alpha = 10: \n")

model = Ridge(10)
scaler = preprocessing.StandardScaler()
scaler.fit(training_data)
norm_train = scaler.transform(training_data)
model.fit(norm_train, prices)

print("The coefficients for the regression are: \n", model.coef_, "\n")
print("The bias for the regression is: \n", model.intercept_, "\n")

features = ["Year",
            "Kilometers Driven",
            "Mileage",
            "Engine",
            "Power",
            "Seats",
            "Owner Type",
            "Fuel Type",
            "Transmission"]

max_index = np.argmax(np.abs(model.coef_))
most_significant_feature = features[int(max_index)]

second_most_significant_feature = features[(max_index + 1)]

min_index = np.argmin(np.abs(model.coef_))
least_significant_feature = features[int(min_index)]

print("Features are: ", features, "\n")

print("The most significant feature is: %s\n" % most_significant_feature)
print("The second most significant feature is: %s\n" % second_most_significant_feature)
print("The least significant feature is: %s\n" % least_significant_feature)
