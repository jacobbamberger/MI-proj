# starting point tutorial: https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-cross-validation/train-cross-validation.py
# I do not understand this code fully

#!/usr/bin/env python

import wandb
import os
import multiprocessing
import collections
import random

from util.train import GNN
from datasets import split_data

SWEEP_NAME = "no-augmentation-CoordToCnc" # for human understandability: this should be equal to sweep name in yaml
JOB_NAME = "hyper_param_options"
# RUN_NAME = "option-nb" # did not manage to make this more human-understandable...

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "config")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("val_f1score"))


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(num, sweep_q, worker_q, train_set, val_set, test_set):
    reset_wandb_env()
    worker_data = worker_q.get()
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.num)
    config = worker_data.config
    run = wandb.init( # thes are going in the inner group!!!!
        #group='test_cross_val', 
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=config,
    )

########## ADDED BY YAYA
    print("Train set length: ", len(train_set))
    print("Val set length: ", len(val_set))
    print("Test set length: ", len(test_set))
    print("")
    print("Val patients: ", val_set.data)
    print("")
    print("Test patients: ", test_set.data)

    optim_param = {
        'name': config['optim'],
        'lr': config['optim_lr'],
        'momentum': config['optim_momentum'],
    }

    model_param = {
        'physics': config['physics'],
        'name': config['model'],
    }

    gnn = GNN(
        config['path_model'] + config['model'],
        model_param,
        train_set,
        val_set,
        test_set,
        config['batch_size'],
        optim_param,
        config['weighted_loss'],
    )

    val_f1score = gnn.train(config['epochs'], config['early_stop'])

    run.log(dict(val_f1score=val_f1score))
    wandb.join()
    sweep_q.put(WorkerDoneData(val_f1score=val_f1score))


def main():
    num_folds = 5

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue() #what is this? gets result???
    workers = []

    test_set, split_list = split_data("../data/CoordToCnc", 
                                      3,
                                      cv=True,
                                      k_cross=num_folds)
    for num, (train_set, val_set) in enumerate(split_list):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=train, kwargs=dict(num=num, sweep_q=sweep_q, worker_q=q, train_set=train_set, val_set=val_set, test_set=test_set)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    sweep_run = wandb.init(group=SWEEP_NAME, 
                           job_type=JOB_NAME)
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    metrics = []


    for num, (train_set, val_set) in enumerate(split_list):
        worker = workers[num]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                num=num,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        metrics.append(result.val_f1score)

    print("METRIIIIIIIICS: ", metrics)
    sweep_run.log(dict(val_f1score=sum(metrics) / len(metrics))) # average all runs with these params
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)



main()
