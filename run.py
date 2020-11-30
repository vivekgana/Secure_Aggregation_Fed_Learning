



if __name__ == '__main__':
    config = {
        'n_clients': 5,
        'key_length': 1024,
        'n_iter': 50,
        'eta': 1.5,
    }
    # load data, train/test split and split training data between clients
    X, y, X_test, y_test = get_data(n_clients=config['n_clients'])
    # first each hospital learns a model on its respective dataset for comparison.
    local_learning(X, y, X_test, y_test, config)
    # and now the full glory of federated learning
    federated_learning(X, y, X_test, y_test, config)