# -*- coding: utf-8 -*-
import math
import numpy as np


class Server(object):
    def __init__(self, args):
        self.args = args
        self.model_shared_params = {}

    def update_model_shared_params(self, clients, random_cids):
        """update parameters of models shared by all active clients at each
        epoch.

        Args:
            clients: A list of clients instances.
            random_cids: Randomly selected client ID in each training round.
        """
        for c_id in random_cids:
            # Obtain current client's parameters
            self.model_shared_params[c_id] = clients[c_id].get_params()

    def choose_clients(self, n_clients, ratio=1.0):
        """Randomly chooses some clients.
        """
        choose_num = math.ceil(n_clients * ratio)
        return np.random.permutation(n_clients)[:choose_num]

    def get_model_shared_params(self):
        return self.model_shared_params
