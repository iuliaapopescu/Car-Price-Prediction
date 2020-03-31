- Split training data in 3 folds

```
training_data_1, prices_1 = training_data[:num_samples_fold], \
                            prices[:num_samples_fold]

training_data_2, prices_2 = training_data[num_samples_fold: 2 * num_samples_fold], \
                            prices[num_samples_fold: 2 * num_samples_fold]

training_data_3, prices_3 = training_data[2 * num_samples_fold:], \
                            prices[2 * num_samples_fold:]
```

- Define normalizing method using the standard scaler from sklearn

```
def normalize(train_data, test_data):

scaler = preprocessing.StandardScaler()
```

- Define function that trains the model depending on the regression, calculates predictions based on the normalized data and returns the mean absolute error and mean square error

```
reg = model.fit(norm_train, train_labels)

predict = reg.predict(norm_test)

mae = mean_absolute_error(test_labels, predict)

mse = mean_squared_error(test_labels, predict)
```

- Calculate values for the ridge regression with alpha taking 4 values: 1, 10, 100, 1000

- For the best performing alpha calculate values for the ridge regression on the whole training data

```
model = Ridge(10)

scaler = preprocessing.StandardScaler()

scaler.fit(training_data)

norm_train = scaler.transform(training_data)

model.fit(norm_train, prices)
```
