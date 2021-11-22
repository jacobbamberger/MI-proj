import numpy as np
import os
import random
import torch
import argparse
import wandb
import yaml
from itertools import product

from util.train import GNN
from datasets import split_data

MODEL_TYPE = 'noAug'

print("cuda available: ", torch.cuda.is_available())


def run_cross_val(split_list, test_set, args):

    empty_run = wandb.init(reinit=True,
                          project='mi-prediction') # just to get a unique run name
    meta_name=empty_run.name
    #empty_run.log({'dummy': 8})
    empty_run.finish()


    dic_val_metrics = {
        'avg_val_acurracy': [],
        'avg_val_precision': [],
        'avg_val_recall': [],
        'avg_val_sensitivity': [],
        'avg_val_specificity': [],
        'avg_val_f1score': [],
        'avg_val_graph_loss': []
    }

    for i, (train_set, val_set) in enumerate(split_list):

        run = wandb.init(reinit=True,
                project='mi-prediction',
                group='indiv-CV-run_'+MODEL_TYPE,
                job_type=meta_name,
                name=meta_name+'-'+str(i),
                config=args)
        
        print('Cross val: ', i)
        print("Train set length: ", len(train_set))
        print("Val set length: ", len(val_set))
        print("Val patients: ", val_set.data)

        #args['path_data'] = os.path.join(args['path_data']) 

        optim_param = {
            'name': args['optim'],
            'lr': args['optim_lr'],
            'momentum': args['optim_momentum'],
        }

        model_param = {
            'physics': args['physics'],
            'name': args['model'],
        }

        gnn = GNN(
            args['path_model'] + args['model'],
            model_param,
            train_set,
            val_set,
            test_set,
            args['batch_size'],
            optim_param,
            args['weighted_loss'],
        )


        print('Runnning!!')
        acc, prec, rec, sensitivity, specificity, f1score, graph_loss = gnn.train(args['epochs'], args['early_stop'], args['allow_stop'], run)
        run.save()
        
        run.finish()

        dic_val_metrics['avg_val_acurracy'].append(acc)
        dic_val_metrics['avg_val_precision'].append(prec)
        dic_val_metrics['avg_val_recall'].append(rec)
        dic_val_metrics['avg_val_sensitivity'].append(sensitivity)
        dic_val_metrics['avg_val_specificity'].append(specificity)
        dic_val_metrics['avg_val_f1score'].append(f1score)
        dic_val_metrics['avg_val_graph_loss'].append(graph_loss)

    meta_run = wandb.init(reinit=True,
                            project='mi-prediction',
                            group='avg-CV-val-scores_'+ MODEL_TYPE,
                            name=meta_name,
                            config=args)
    print("arguments for the k runs: ", args)

    print("Test set length: ", len(test_set))
    print("Test patients: ", test_set.data)

    avg_dic = {}
    for key in dic_val_metrics.keys(): # could do all at same time......
        avg_dic[key] = sum(dic_val_metrics[key]) / len(dic_val_metrics[key])
    meta_run.log(avg_dic)

    wandb.finish()


print("starting .......")

test_set, split_list = split_data(path= "../data/CoordToCnc", 
                        num_node_feat=3,
                        cv=True,
                        k_cross=5)

yaml_file = open("hyper_params.yaml")
arg_dic = yaml.load(yaml_file, Loader=yaml. FullLoader)

keys = arg_dic.keys()
values = arg_dic.values()
for instance in product(*values):
    config = dict(zip(keys, instance))
    run_cross_val(split_list=split_list, test_set=test_set, args=config)

