seed = 43
np.random.seed(seed)


def mean_square_error(y_pred, y):
    """ 1/m * \sum_{i=1..m} (y_pred_i - y_i)^2 """
    return np.mean((y - y_pred) ** 2)


def encrypt_vector(public_key, x):
    return [public_key.encrypt(i) for i in x]


def decrypt_vector(private_key, x):
    return np.array([private_key.decrypt(i) for i in x])


def sum_encrypted_vectors(x, y):
    if len(x) != len(y):
        raise ValueError('Encrypted vectors must have the same size')
    return [x[i] + y[i] for i in range(len(x))]


def federated_learning(X, y, X_test, y_test, config):
    n_clients = config['n_clients']
    n_iter = config['n_iter']
    names = ['Hospital {}'.format(i) for i in range(1, n_clients + 1)]

    # Instantiate the server and generate private and public keys
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    server = Server(key_length=config['key_length'])

    # Instantiate the clients.
    # Each client gets the public key at creation and its own local dataset
    clients = []
    for i in range(n_clients):
        clients.append(Client(names[i], X[i], y[i], server.pubkey))

    # The federated learning with gradient descent
    print('Running distributed gradient aggregation for {:d} iterations'
          .format(n_iter))
    for i in range(n_iter):

        # Compute gradients, encrypt and aggregate
        encrypt_aggr = clients[0].encrypted_gradient(sum_to=None)
        for c in clients[1:]:
            encrypt_aggr = c.encrypted_gradient(sum_to=encrypt_aggr)

        # Send aggregate to server and decrypt it
        aggr = server.decrypt_aggregate(encrypt_aggr, n_clients)

        # Take gradient steps
        for c in clients:
            c.gradient_step(aggr, config['eta'])

    print('Error (MSE) that each client gets after running the protocol:')
    for c in clients:
        y_pred = c.predict(X_test)
        mse = mean_square_error(y_pred, y_test)
        print('{:s}:\t{:.2f}'.format(c.name, mse))


def local_learning(X, y, X_test, y_test, config):
    n_clients = config['n_clients']
    names = ['Hospital {}'.format(i) for i in range(1, n_clients + 1)]

    # Instantiate the clients.
    # Each client gets the public key at creation and its own local dataset
    clients = []
    for i in range(n_clients):
        clients.append(Client(names[i], X[i], y[i], None))

    # Each client trains a linear regressor on its own data
    print('Error (MSE) that each client gets on test set by '
          'training only on own local data:')
    for c in clients:
        c.fit(config['n_iter'], config['eta'])
        y_pred = c.predict(X_test)
        mse = mean_square_error(y_pred, y_test)
        print('{:s}:\t{:.2f}'.format(c.name, mse))
