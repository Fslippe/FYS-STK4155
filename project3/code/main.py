from functions import *
from grid_search import *
sns.set_style("whitegrid")
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
    sns.heatmap(corr)
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
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()
    print(np.shape(df))
    for loca in df["Location"].unique():
        train, test = train_test_split(
            df.loc[df["Location"] == loca], test_size=test_size, random_state=1)
        X_train = X_train.append(train.iloc[:, 2:-1])
        X_test = X_test.append(test.iloc[:, 2:-1])
        y_train = y_train.append(train.iloc[:, -1:])
        y_test = y_test.append(test.iloc[:, -1:])

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


def main():
    run_gridsearch = False
    # Import Dataset
    df_full = pd.read_csv("data/weatherAUS.csv")
    df, target = import_dataset(df_full)
    print("RUNNING WHOLE DATASET")
    X_train, X_test, y_train, y_test = split_each_loc(df, test_size=0.2)
    # Print number of measurements at each station
    n_locations = 0

    for loca in df["Location"].unique():
        print("\n%s size: " % (loca), len(df.loc[df["Location"] == loca]))
        print(get_averages(df, loca))
        n_locations += 1

    # Set up data for chosen location
    location_idx = df.index[df["Location"] == "Cobar"]
    y = target[location_idx]  # np.ravel(target.ilo)
    X = df.iloc[location_idx, 2:-1]
    print(y)
    print(X)
    # Correlation plots
    correlation_plot(df, "cobar", location_idx)

    # Train-Test-split and standard scale
    X_train, X_test, y_train, y_test = scale_and_split(X, y)

    # Perform grid search for NN
    if run_gridsearch:
        grid_search_layers([10, 20, 30, 40, 50], [1, 2, 3, 4, 5], X_train,
                           X_test, y_train, y_test, optimizer="ADAM", n_B=10, savename="NN_grid_ADAM_bootstrap_cobar")
        plt.show()

    # Run NN
    loc_1 = "Cobar"
    loc_2 = "CoffsHarbour"
    loc_3 = "Darwin"
    idx_1 = df.index[df["Location"] == loc_1]
    idx_2 = df.index[df["Location"] == loc_2]
    idx_3 = df.index[df["Location"] == loc_3]

    print(idx_1.append(idx_2))
    print()
    print("\n\n\nNeural Network")
    model = create_neural_network_keras([20, 20])
    scores = train_loc_test_loc(df, model, idx_1, idx_2)
    print("\n", loc_1, "trained model accuracy on predicting", loc_2)
    print(scores[1])

    scores = train_loc_test_loc(df, model, idx_1, idx_3)
    print("\n", loc_1, "trained model accuracy on predicting", loc_3)
    print(scores[1])

    # Grid search random forest
    if run_gridsearch:
        trees = [10, 50, 100, 200, 500]
        depth = [5, 10, 15, 20, 25, 30]
        grid_search_trees_depth(
            trees, depth, X_train, X_test, y_train, y_test, n_B=10, savename="RF_grid_all")

    # Set up Random forest model for best parameters
    print("\n\n\nRandom Forest")
    print(X_train, y_train)
    model_rf = tfdf.keras.RandomForestModel(
        num_trees=100, max_depth=10, verbose=0)
    model_rf. compile(
        metrics=["accuracy"])
    scores = train_loc_test_loc(df, model_rf, idx_1, idx_2, randomforest=True)
    print("\n", loc_1, "trained model accuracy on predicting", loc_2)
    print(scores[1])

    scores = train_loc_test_loc(df, model_rf, idx_1, idx_3, randomforest=True)
    print("\n", loc_1, "trained model accuracy on predicting", loc_3)
    print(scores[1])

    # Run whole dataset
    print("RUNNING WHOLE DATASET")
    X_train, X_test, y_train, y_test = split_each_loc(df, test_size=0.2)
    print("\n\n\n\n\n\n\n\RAINTOMORROW", X_train, y_train)

    # y_all = target[:]  # np.ravel(target.ilo)
    #X_all = df.iloc[:, 2:-1]
    #X_train, X_test, y_train, y_test = scale_and_split(X_all, y_all)
    model_rf = tfdf.keras.RandomForestModel(
        num_trees=100, max_depth=10, verbose=0)
    model_rf. compile(
        metrics=["accuracy"])
    model_rf.fit(X_train, y_train)
    scores = model_rf.evaluate(X_test, y_test)
    print(np.shape(X_train))
    print("Full data accuracy score on test data:", scores[1])
    trees = [10, 50, 100, 200, 500]
    depth = [5, 10, 15, 20, 25, 30]

    print("\n\n\n\n\n\n\n\RAINTOMORROW", y_train)
    grid_search_trees_depth(
        trees, depth, X_train, X_test, y_train, y_test, n_B=None, savename="RF_grid_all")


if __name__ == "__main__":
    main()
