from python import Python


fn main() raises:
    # pairplot()
    train()


def pairplot():
    pl = Python.import_module("polars")
    sns = Python.import_module("seaborn")
    plt = Python.import_module("matplotlib.pyplot")

    q = pl.scan_csv("data/Battery_RUL.csv")
    df = q.collect().to_pandas()

    sns.pairplot(df)
    plt.show()


def calculate_metrics(model_name: String, y_test: PythonObject, y_test_pred: PythonObject):
    sklearn = Python.import_module("sklearn")

    r2 = sklearn.metrics.r2_score(y_test, y_test_pred)
    rmse = sklearn.metrics.root_mean_squared_error(y_test, y_test_pred)
    
    print("Model: ", model_name)
    print("r2 score: ", r2)
    print("Root Mean Squared Error: ", rmse)
    print("")


def train():
    np = Python.import_module("numpy")
    pl = Python.import_module("polars")
    ensemble = Python.import_module("sklearn.ensemble")
    linear_model = Python.import_module("sklearn.linear_model")

    q = pl.scan_csv("data/Battery_RUL.csv")
    df = q.collect().to_pandas()

    # create device_id
    df["split"] = np.isclose(df["Cycle_Index"], 1.0)
    df["device_id"] = df["split"].cumsum()
    devices = df["device_id"].unique()
    np.random.shuffle(devices)

    num_train_devices = 10
    # slice the num_train_devices from the shuffled list
    train_devices = np.take(devices, np.arange(0, num_train_devices))
    test_devices = np.take(devices, np.arange(num_train_devices, len(devices)))
    
    train = df[df['device_id'].isin(train_devices)]
    test = df[df['device_id'].isin(test_devices)]

    # drop unecessary columns
    train = train.drop(["device_id", "Cycle_Index", "split"], axis=1)
    test = test.drop(["device_id", "Cycle_Index", "split"], axis=1)

    # # split into features and labels X, y
    X_train = train.drop(train.columns[-1], axis=1)
    y_train = train[train.columns[-1]]
    X_test = test.drop(test.columns[-1], axis=1)
    y_test = test[test.columns[-1]]
    
    random_forest_model = ensemble.RandomForestRegressor(n_estimators = 100, random_state = 0)
    random_forest_model.fit(X_train, y_train)
    y_test_pred_random_forest = random_forest_model.predict(X_test)

    linear_model = linear_model.LinearRegression()
    linear_model.fit(X_train, y_train)
    y_test_pred_linear = linear_model.predict(X_test)

    calculate_metrics("random forest", y_test, y_test_pred_random_forest)
    calculate_metrics("linear", y_test, y_test_pred_linear)
