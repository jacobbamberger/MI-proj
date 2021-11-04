import numpy as np
import os
import random
import torch
import argparse
import wandb

from util.train import GNN
from datasets import DataSet

print("parsiiiiing")
print(torch.cuda.is_available())
#print(os.listdir("./data/"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='NoPhysicsGnn') #TsviPlusCnc or NoPhysicsGnn
parser.add_argument("--num_node_features", type=int, default=3) # 3 for Coord, 60 for WSS it seems
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--early_stop", type=int, default=10)
parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'SGD'])
parser.add_argument("--optim_lr", type=float, default=0.0005)
parser.add_argument("--optim_momentum", type=float, default=0.0)
parser.add_argument("--physics", type=int, default=0)
parser.add_argument("--weighted_loss", type=float, default=0.5)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--save_name", type=str, default='unnamed_model')

# parser.add_argument("--cross_validation", action='store_true', default=False)
parser.add_argument("--path_data", type=str, default="./data/CoordToCnc")
parser.add_argument("--path_model", type=str, default="./experiments/util/")
args = parser.parse_args()

wandb.init(project='mi-prediction', config=args)

# if args.cross_validation:
#     seed = 0
#     cv = args.seed
# else: # not cross val
#     seed = args.seed
#     cv = -1

# random.seed(seed) # What does this do? Why is it important?
# np.random.seed(seed)
# torch.manual_seed(seed)

print("starting .......")

args.path_data = os.path.join(args.path_data) 
train_set = DataSet(args.path_data, 'train', args.num_node_features) # , cv)
valid_set = DataSet(args.path_data, 'val', args.num_node_features) #, cv)
test_set = DataSet(args.path_data, 'test', args.num_node_features) # , cv)

print("Train set length: ", len(train_set))
print("Val set length: ", len(valid_set))
print("Test set length: ", len(test_set))

optim_param = {
    'name': args.optim,
    'lr': args.optim_lr,
    'momentum': args.optim_momentum,
}

model_param = {
    'physics': args.physics,
    'name': args.model,
}

gnn = GNN(
    args.path_model + args.model,
    model_param,
    train_set,
    valid_set,
    test_set,
    args.batch_size,
    optim_param,
    args.weighted_loss,
)

# train_set = DataSet("./data/CoordToCnc", 'train', 3, 0)
# valid_set = DataSet("./data/CoordToCnc", 'val', 3, 0)
# test_set = DataSet("./data/CoordToCnc", 'test', 3, 0)


# gnn = GNN("./experiments/util/NoPhysicsGnn", {'physics': 0, 'name': 'NoPhysicsGnn'}, train_set,valid_set,test_set,5,{'name': 'Adam', 'lr': 0.0005, 'momentum': 0.0},0.5)

print('Runnning!!')
gnn.train(args.epochs, args.early_stop) #train the model. ERROR
gnn.evaluate(val_set=False)


print("Saving under: ", args.save_name)
torch.save(gnn, args.save_name)
# with open('model_CoordToCnc_rot(-45,45, 9).pt', 'rb') as f:
#     gnn_re = torch.load(f)

print("Done!!")
