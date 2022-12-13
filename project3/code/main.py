from functions import *
from grid_search import *
sns.set_style("darkgrid")
tf.keras.utils.set_random_seed(1)


def assign_labels(data, words):
    for i in range(len(words)):
        data.replace(words[i], i, inplace=True)
    return data


def split_grid(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


def import_dataset(df):
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    wind_directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE",
                       "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    wind_directions = sorted(df["WindDir3pm"].unique())
    df["WindDir3pm"] = assign_labels(df["WindDir3pm"], wind_directions)
    df["WindDir9am"] = assign_labels(df["WindDir9am"], wind_directions)
    df["WindGustDir"] = assign_labels(df["WindGustDir"], wind_directions)
    df["RainTomorrow"] = assign_labels(df["RainTomorrow"], ["No", "Yes"])
    df["RainToday"] = assign_labels(df["RainToday"], ["No", "Yes"])

    df["RainTomorrow"] = assign_labels(df["RainTomorrow"], ["No", "Yes"])
    target = df["RainTomorrow"]
    return df, target


def correlation_plot(df, savename, idx):
    corr = (df.iloc[idx, 2:]).corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, cmap="coolwarm")
    plt.savefig("../figures/correlation_heatmap_%s.png" % (savename),
                dpi=300, bbox_inches='tight')
    plt.show()

    corr_tomorrow = pd.DataFrame({"RainTomorrow": corr["RainTomorrow"]})
    print(corr_tomorrow)
    sns.scatterplot(x=corr_tomorrow.index, y=corr_tomorrow["RainTomorrow"])
    plt.xticks(rotation=90)
    plt.savefig("../figures/correlation_plot_%s.png" % (savename),
                dpi=300, bbox_inches='tight')

    plt.show()


def get_averages(df, location):
    return df.loc[df["Location"] == location].mean()


def scale_and_split(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_loc_test_loc(df, model, idx_1, idx_2, randomforest=False):

    # Training and test set of locations
    X_train, y_train = shuffle(
        df.iloc[idx_1, 2:-1],  df["RainTomorrow"].iloc[idx_1], random_state=1)
    X_test, y_test = shuffle(
        df.iloc[idx_2, 2:-1], df["RainTomorrow"].iloc[idx_2], random_state=1)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if randomforest:
        model.fit(X_train, y_train, verbose=0)
    else:
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    scores = model.evaluate(X_test, y_test)
    return scores


def split_each_loc(df, test_size):
    train = pd.DataFrame()
    test = pd.DataFrame()

    print(np.shape(df))
    for loca in df["Location"].unique():
        train_loc, test_loc = train_test_split(
            df.loc[df["Location"] == loca], test_size=test_size, random_state=1)
        train = train.append(train_loc)  # (train.iloc[:, 2:-1])
        test = test.append(test_loc)  # (test.iloc[:, 2:-1])

    train = shuffle(train, random_state=1)
    test = shuffle(test, random_state=1)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    return train, test


def grid_search_location(df, target, location):

    # Set up data for chosen location
    location_idx = df.index[df["Location"] == location]
    y = target[location_idx]  # np.ravel(target.ilo)
    X = df.iloc[location_idx, 2:-1]
    # Correlation plots
    # correlation_plot(df, location, location_idx)

    # Train-Test-split and standard scale
    X_train, X_test, y_train, y_test = scale_and_split(X, y)

    # Logistic Regression
    print(np.shape(X_train), np.shape(y_train))
    eta = np.logspace(-4, 0, 5)
    lamda = np.logspace(-6, 0, 7)
    method = ["momentum", "ADAM"]
    for m in method:
        grid_search_logreg(X_train,
                           y_train.to_numpy().reshape(len(y_train), 1),
                           X_test,
                           y_test.to_numpy().reshape(len(y_test), 1),
                           gradient="SGD",
                           lamda=lamda,
                           eta=eta,
                           method=m,
                           iterations=100,
                           batch_size=32,
                           mom=0.3,
                           savename="logreg_%s_%s" % (m, location),
                           n_B=None)
    plt.show()
    # Perform grid search for NN

    # Neural Network
    grid_search_layers([10, 20, 30, 40, 50], [1, 2, 3, 4, 5], X_train,
                       X_test, y_train, y_test, optimizer="ADAM", n_B=10, savename="NN_grid_ADAM_bootstrap_%s" % (location))

    # Random Forest
    trees = [10, 50, 100, 200, 500]
    depth = [5, 10, 15, 20, 25, 30]
    grid_search_trees_depth(
        trees, depth, X_train, X_test, y_train, y_test, n_B=10, savename="RF_grid_bootstrap_%s" % (location))
    plt.show()


def average_plots(df, average):
    loc_df = pd.DataFrame({"Location": df["Location"].unique()}).T
    loc_df.columns = average.columns
    total_average = average.mean(axis=1)
    average_diff = (average.T - total_average.T).T
    average_diff = loc_df.append(average_diff)

    plt.figure(figsize=(10, 8))
    plt.title("Humidity3pm difference from mean")
    sns.scatterplot(
        data=average_diff.T, x="Humidity3pm", y="Location")
    plt.ylabel("Location")
    plt.xlabel("Relative Humidity (%)")
    plt.savefig("../figures/Relative_humidity3pm.png",
                dpi=300, bbox_inches='tight')
    plt.figure(figsize=(10, 8))
    plt.title("Sunshine difference from mean")
    sns.scatterplot(data=average_diff.T, y="Location", x="Sunshine")
    plt.ylabel("Location")
    plt.xlabel("Sunhine (hours)")
    plt.savefig("../figures/Sunshine.png",
                dpi=300, bbox_inches='tight')
    plt.show()


def main():
    run_gridsearch = False
    # Import Dataset
    df_full = pd.read_csv("data/weatherAUS.csv")
    df, target = import_dataset(df_full)
    # Print number of measurements at each station
    n_locations = 0

    average = pd.DataFrame()
    for loca in df["Location"].unique():
        print("\n%s size: " % (loca), len(df.loc[df["Location"] == loca]))
        average.insert(0, loca, get_averages(df, loca), True)
        n_locations += 1

    # average_plots(df, average)
    # correlation_plot(df, "full", df.index)

    # Train and test data on Cobar
    print("GRIDSEARCH ")
    # grid_search_location(df, target, location="Cobar")

    # Training on one location, test on another
    loc_1 = "Cobar"
    loc_2 = "CoffsHarbour"
    loc_3 = "Darwin"
    idx_1 = df.index[df["Location"] == loc_1]
    idx_2 = df.index[df["Location"] == loc_2]
    idx_3 = df.index[df["Location"] == loc_3]

    # Run Neural Network
    print("\n\n\nNeural Network")
    model = create_neural_network_keras([20, 20])
    scores = train_loc_test_loc(df, model, idx_1, idx_2)
    print("\n", loc_1, "trained Neural Network accuracy on predicting", loc_2)
    print(scores[1])

    scores = train_loc_test_loc(df, model, idx_1, idx_3)
    print("\n", loc_1, "trained Neural Network accuracy on predicting", loc_3)
    print(scores[1])

    # Run Random forest
    print("\n\n\nRandom Forest")
    model_rf = tfdf.keras.RandomForestModel(
        num_trees=100, max_depth=10, verbose=0)
    model_rf. compile(
        metrics=["accuracy"])
    scores = train_loc_test_loc(df, model_rf, idx_1, idx_2, randomforest=True)
    print("\n", loc_1, "trained Random Forest accuracy on predicting", loc_2)
    print(scores[1])

    scores = train_loc_test_loc(df, model_rf, idx_1, idx_3, randomforest=True)
    print("\n", loc_1, "trained Random Forest accuracy on predicting", loc_3)
    print(scores[1])

    # Grid search random forest
    if run_gridsearch:
        trees = [10, 50, 100, 200, 500]
        depth = [5, 10, 15, 20, 25, 30]
        grid_search_trees_depth(
            trees, depth, X_train, X_test, y_train, y_test, n_B=10, savename="RF_grid_all")

    # Run whole dataset
    print("\n\n\nRUNNING WHOLE DATASET")

    train, test = split_each_loc(df, test_size=0.2)
    X_train, X_test = train.iloc[:, 2:-
                                 1].to_numpy(), test.iloc[:, 2:-1].to_numpy()
    y_train, y_test = train.iloc[:, -
                                 1:].to_numpy(), test.iloc[:, -1:].to_numpy()

    print(np.shape(y_train))
    print(np.where(
        y_train == 1))
    print(len(np.where(
        y_train == 1)))

    print("Fraction of all data with no rain tomorrow:", (len(np.where(
        y_train == 1)[0]) + len(np.where(y_test == 1)[0]))/(len(y_train) + len(y_test)))

    trees_best = 500
    depth_best = 20
    train_cobar = train.loc[train["Location"] == "Cobar"]
    test_cobar = test.loc[test["Location"] == "Cobar"]
    train_darwin = train.loc[train["Location"] == "Darwin"]
    test_darwin = test.loc[test["Location"] == "Darwin"]
    X_test_cobar = test_cobar.iloc[:, 2:-1].to_numpy()
    y_test_cobar = test_cobar.iloc[:, -1:].to_numpy()
    X_test_darwin = test_darwin.iloc[:, 2:-1].to_numpy()
    y_test_darwin = test_darwin.iloc[:, -1:].to_numpy()

    model_rf = tfdf.keras.RandomForestModel(
        num_trees=100, max_depth=10, verbose=0)
    model_rf. compile(
        metrics=["accuracy"])
    model_rf.fit(X_train, y_train)
    scores_cobar = model_rf.evaluate(X_test_cobar, y_test_cobar)
    scores_darwin = model_rf.evaluate(X_test_darwin, y_test_darwin)

    print("\n\nFull data accuracy score on Cobar test data:", scores_cobar[1])
    print("Full data accuracy score on Darwin test data:", scores_darwin[1])
    print("\n")

    model = create_neural_network_keras([20, 20])
    model.fit(X_train, y_train)
    scores_cobar = model.evaluate(X_test_cobar, y_test_cobar)
    scores_darwin = model.evaluate(X_test_darwin, y_test_darwin)

    print("\n\nFull data accuracy score on Cobar test data:", scores_cobar[1])
    print("Full data accuracy score on Darwin test data:", scores_darwin[1])
    # Grid search on all data
    trees = [10, 50, 100, 200, 500]
    depth = [5, 10, 15, 20, 25, 30]

    # grid_search_trees_depth(
    #    trees, depth, X_train, X_test, y_train, y_test, n_B=None, savename="RF_grid_all")

    print("STARTING GRID SEARCH FOR FULL DATASET")
    grid_search_layers([10, 20, 30, 40, 50], [1, 2, 3, 4, 5], X_train,
                       X_test, y_train, y_test, optimizer="ADAM", n_B=None, epochs=100, batch_size=320, savename="NN_grid_ADAM_all")
    print("FINISHED")


if __name__ == "__main__":
    main()
