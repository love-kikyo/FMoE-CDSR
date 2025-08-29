# -*- coding: utf-8 -*-
import logging
from tqdm import tqdm
from utils.train_utils import EarlyStopping, LRDecay


def evaluation_logging(eval_logs, round, weights, mode="valid"):
    if mode == "valid":
        logging.info("Epoch%d Valid:" % round)
    else:
        logging.info("Test:")
    domain_list = list(eval_logs.keys())
    avg_eval_log = {}
    avg_eval_log_shared = {}
    for metric_name in list(eval_logs.values())[0][1].keys():
        avg_eval_val = 0
        avg_eval_val_shared = 0
        for i in range(len(domain_list)):
            eval_list, eval_shared = eval_logs[domain_list[i]]
            avg_eval_val += eval_list[i][metric_name] * weights[domain_list[i]]
            avg_eval_val_shared += eval_shared[metric_name] * \
                weights[domain_list[i]]
        avg_eval_log[metric_name] = avg_eval_val
        avg_eval_log_shared[metric_name] = avg_eval_val_shared

    for i in range(len(eval_logs)):
        logging.info("domain:%s", domain_list[i])
        eval_log = eval_logs[domain_list[i]]
        # for j in range(len(eval_log[0])):
        #     if i != j:
        #         logging.info("model_%s:", j)
        #     else:
        #         logging.info("model_local:")
        #     logging.info("MRR: %.4f" % eval_log[0][j]["MRR"])
            # logging.info("HR @1|5|10: %.4f \t %.4f \t %.4f \t" %
            #     (eval_log[0][j]["HR_1"], eval_log[0][j]["HR_5"],
            #     eval_log[0][j]["HR_10"]))
            # logging.info("NDCG @5|10: %.4f \t %.4f" %
            #     (eval_log[0][j]["NDCG_5"], eval_log[0][j]["NDCG_10"]))
        logging.info("model_moe:")
        logging.info("MRR: %.4f" % eval_log[1]["MRR"])
        logging.info("HR @1|5|10: %.4f \t %.4f \t %.4f \t" %
            (eval_log[1]["HR_1"], eval_log[1]["HR_5"],
            eval_log[1]["HR_10"]))
        logging.info("NDCG @5|10: %.4f \t %.4f" %
                    (eval_log[1]["NDCG_5"], eval_log[1]["NDCG_10"]))

    # logging.info("\tavg_model_local:")
    # logging.info("MRR: %.4f" % avg_eval_log["MRR"])
    # logging.info("HR @1|5|10: %.4f \t %.4f \t %.4f \t" %
    #     (avg_eval_log["HR_1"], avg_eval_log["HR_5"],
    #         avg_eval_log["HR_10"]))
    # logging.info("NDCG @5|10: %.4f \t %.4f" %
    #     (avg_eval_log["NDCG_5"], avg_eval_log["NDCG_10"]))

    logging.info("\tavg_model_moe:")
    logging.info("MRR: %.4f" % avg_eval_log_shared["MRR"])
    logging.info("HR @1|5|10: %.4f \t %.4f \t %.4f \t" %
        (avg_eval_log_shared["HR_1"], avg_eval_log_shared["HR_5"],
            avg_eval_log_shared["HR_10"]))
    logging.info("NDCG @5|10: %.4f \t %.4f" %
        (avg_eval_log_shared["NDCG_5"], avg_eval_log_shared["NDCG_10"]))

    return avg_eval_log_shared


def load_and_eval_model(n_clients, clients, args):
    eval_logs = {}
    for c_id in tqdm(range(n_clients), ascii=True):
        clients[c_id].load_params()
        eval_log = clients[c_id].evaluation(mode="test")
        eval_logs[clients[c_id].domain] = eval_log
    weights = dict((client.domain, client.test_weight) for client in clients)
    evaluation_logging(eval_logs, 0, weights, mode="test")


def run_fl(clients, server, args):
    n_clients = len(clients)
    if args.do_eval:
        load_and_eval_model(n_clients, clients, args)
    else:
        early_stopping = EarlyStopping(
            args.checkpoint_dir, patience=args.es_patience, verbose=True)
        lr_decay = LRDecay(args.lr, args.decay_epoch,
                           args.optimizer, args.lr_decay,
                           patience=args.ld_patience, verbose=True)
        for round in range(1, args.epochs + 1):
            random_cids = server.choose_clients(n_clients, args.frac)

            # Train with these clients
            for c_id in tqdm(random_cids, ascii=True):
                # Restore global parameters to client's model
                clients[c_id].set_shared_params(
                    server.get_model_shared_params())

                # Train one client
                clients[c_id].train_epoch(
                    round, args)

            server.update_model_shared_params(clients, random_cids)

            if round % args.eval_interval == 0:
                eval_logs = {}
                for c_id in tqdm(range(n_clients), ascii=True):
                    clients[c_id].set_shared_params(
                        server.get_model_shared_params())
                    if c_id in random_cids:
                        eval_log_list, eval_log_shared = clients[c_id].evaluation(
                            mode="valid")
                    else:
                        eval_log_list, eval_log_shared = clients[c_id].get_old_eval_log(
                        )
                    eval_logs[clients[c_id].domain] = eval_log_list, eval_log_shared

                weights = dict((client.domain, client.valid_weight)
                               for client in clients)
                avg_eval_log = evaluation_logging(
                    eval_logs, round, weights, mode="valid")

                # Early Stopping. Here only compare the current results with
                # the best results
                early_stopping(avg_eval_log, clients)
                if early_stopping.early_stop:
                    logging.info("Early stopping")
                    break

                # Learning rate decay. Here only compare the current results
                # with the latest results
                lr_decay(round, avg_eval_log, clients)

        load_and_eval_model(n_clients, clients, args)
