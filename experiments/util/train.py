import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

from torch_geometric.loader import DataLoader

from .models import NoPhysicsGnn


class GNN:
    def __init__(
            self,
            model_path,
            model_param,
            train_set,
            valid_set,
            test_set,
            batch_size,
            optim_param,
            weight1
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)
        self.model_name = model_param['name']
        self.model_path = model_path
        self.ratio = 1.0

        if self.model_name == 'NoPhysicsGnn':
            self.physics = False
            self.automatic_update = False
            self.model = NoPhysicsGnn(train_set)
        else:
            print("Unrecognized model name")
        self.model.to(self.device)

        self.train_set = train_set
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True) # set pin_memory=True to go faster? https://pytorch.org/docs/stable/data.html
        self.val_loader = DataLoader(valid_set, batch_size, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size, shuffle=False)

        if optim_param['name'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optim_param['lr'])
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=optim_param['lr'], momentum=optim_param['momentum'])
        weights = torch.tensor([1 - weight1, weight1]) # is this due to class imbalance??

        self.criterion = torch.nn.CrossEntropyLoss(weight=weights).to(self.device)
        # self.criterion_node = torch.nn.MSELoss().to(self.device)

    @staticmethod
    def calculate_metrics(y_pred, y_true):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1score = f1_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        return accuracy, precision, recall, sensitivity, specificity, f1score

    def get_losses(self, data):
        if self.physics:
            print("Wrong argument: self.physics set to true")
        else:
            cnc = data.y
            cnc_pred = self.model(data.x, data.edge_index, data.batch, data.segment)
            loss = loss_cnc = self.criterion(cnc_pred, data.y)
        return loss, loss_cnc, cnc_pred, cnc

    def train(self, epochs, early_stop):
        self.model.eval()
        self.evaluate(val_set=True)
        self.model.train()
        n_epochs_stop = early_stop
        epochs_no_improve = 0
        min_val_loss = 1e8
        for epoch_idx in range(epochs):
            if epoch_idx %10==0:
                print('epoch nb:', epoch_idx)
            running_loss_cnc = 0.0 # what is this??
            y_pred = np.array([])
            y_true = np.array([])
            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                loss, loss_cnc, cnc_pred, cnc = self.get_losses(data)
                loss.backward()
                self.optimizer.step()
                pred = cnc_pred.argmax(dim=1)
                y_pred = np.append(y_pred, pred.cpu().detach().numpy())
                y_true = np.append(y_true, cnc.cpu().detach().numpy())
                running_loss_cnc += loss_cnc.item()

            train_loss_cnc = running_loss_cnc / len(self.train_loader.dataset)
            acc, prec, rec, sens, spec, f1score = self.calculate_metrics(y_pred, y_true)
            wandb.log({
                'ratio': self.ratio,
                'train_accuracy': acc,
                'train_precision': prec,
                'train_recall': rec,
                'train_loss_graph': train_loss_cnc
            })
            val_loss = self.evaluate(val_set=True)
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                return



    def train_aug_rot_periodic(self, epochs, early_stop, eps=0.05, aug_num=10):
        '''trains model by augmenting data by a rotation matrix.
        at each epoch, it is augmented by a different matrix'''

        self.model.eval()
        self.evaluate(val_set=True)

        # First augment:
        # sample strategy from SO(3)
        a = torch.rand((3, aug_num), device=self.device) #axis of rotation
        a = a.div(a.norm(dim=0)) #normalize
        b = torch.rand((3, aug_num), device=self.device)
        b = b.div(b.norm(dim=0)) #normalize
        dot = torch.mul(a, b).sum(dim=0) #dot product of the list of vectors
        b = b - dot*a # perpendicular
        b = b.div(b.norm(dim=0)) #normalize
        c = torch.cross(a, b) # third member of orthonormal basis

        M = torch.cat((a.transpose(0, 1),
                       b.transpose(0,1),
                       c.transpose(0, 1)), dim=1).reshape(aug_num, 3, 3).transpose(1,2) #each dim0 entry is a 3x3 tensor with columns a, b, c
        #M = torch.cat([a, b, c]).reshape(3,3) # has a, b, and c as rows
        M[0] = torch.eye(3)
        Minv = M.transpose(1, 2)
        # theta = eps * (torch.rand(aug_num, device=self.device)- 0.5) #sample random angle, does this run on GPU?
        cos = 1 - eps * (torch.rand(aug_num, device=self.device)) # sample random cos values
        sin = 1 - torch.sqrt(1-cos*cos)
        first_row = torch.tensor([1., 0., 0.], device=self.device).repeat(aug_num, 1)
        second_row = torch.vstack((torch.zeros(aug_num, device=self.device), cos, -sin)).transpose(0,1)
        third_row = torch.vstack((torch.zeros(aug_num, device=self.device), sin, cos)).transpose(0,1)
        base_rot = torch.cat((first_row, second_row, third_row), dim=1).reshape(aug_num, 3, 3).transpose(1,2)
        # base_rot = torch.tensor([[1., 0., 0.],
        #                             [0., torch.cos(theta), -torch.sin(theta)],
        #                             [0., torch.sin(theta), torch.cos(theta)]], device=self.device)
        # when smapling cos and sin directly
        # base_rot = torch.tensor([[1., 0., 0.],
        #                             [0., cos, - sin],
        #                             [0., sin, cos]], device=self.device)

        rot = Minv@base_rot@M

        self.model.train()
        n_epochs_stop = early_stop
        epochs_no_improve = 0
        min_val_loss = 1e8
        for epoch_idx in range(epochs):
            rot_mat = M[epoch_idx%10]
            if epoch_idx %50 ==0:
                print('epoch nb:', epoch_idx)
            running_loss_cnc = 0.0 # what is this??
            y_pred = np.array([])
            y_true = np.array([])
            for data in self.train_loader:#self.train_set: #removed loader for now...
                data = data.to(self.device)

                #print('egde_inde: ', data.edge_index.shape)
                #print('x shape: ', data.x.shape)

                x = torch.transpose(rot_mat@torch.transpose(data.x, 0, 1), 0,1) #rotates the data
                #edge_index = data.edge_index.repeat(1, aug_num)
                #data.batch = torch.arange(0, aug_num, device=self.device).unsqueeze(dim=1).repeat(1,data.x.shape[0]).reshape((-1,))
                #data.segment = torch.tensor(data.segment, device=self.device).repeat(aug_num)
                #data.y = torch.tensor(data.y, device=self.device).repeat(aug_num)
                data.x = x
                #data.edge_index = edge_index

                self.optimizer.zero_grad()
                loss, loss_cnc, cnc_pred, cnc = self.get_losses(data)
                loss.backward()
                self.optimizer.step()
                pred = cnc_pred.argmax(dim=1)
                y_pred = np.append(y_pred, pred.cpu().detach().numpy())
                y_true = np.append(y_true, cnc.cpu().detach().numpy())
                running_loss_cnc += loss_cnc.item()

            train_loss_cnc = running_loss_cnc / len(self.train_loader.dataset)
            acc, prec, rec, sens, spec, f1score = self.calculate_metrics(y_pred, y_true)
            wandb.log({
                'ratio': self.ratio,
                'train_accuracy': acc,
                'train_precision': prec,
                'train_recall': rec,
                'train_loss_graph': train_loss_cnc
            })
            val_loss = self.evaluate(val_set=True)
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                return

    def train_aug_rot_one_sample_per_batch(self, epochs, early_stop, eps=0.1, aug_num=10):
        '''trains model by first finding the angles and directions of rotations (randomly),
        and then augmenting each batch by these, so the batch size is multiplied by aug_num.
        ONly works when batch_size=1
        ToDo: make it work also when batch_size!=1!!! '''

        self.model.eval()
        self.evaluate(val_set=True)

        # First augment:
        # sample strategy from SO(3)
        a = torch.rand((3, aug_num), device=self.device) #axis of rotation
        a = a.div(a.norm(dim=0)) #normalize
        b = torch.rand((3, aug_num), device=self.device)
        b = b.div(b.norm(dim=0)) #normalize
        dot = torch.mul(a, b).sum(dim=0) #dot product of the list of vectors
        b = b - dot*a # perpendicular
        b = b.div(b.norm(dim=0)) #normalize
        c = torch.cross(a, b) # third member of orthonormal basis

        M = torch.cat((a.transpose(0, 1),
                       b.transpose(0,1),
                       c.transpose(0, 1)), dim=1).reshape(aug_num, 3, 3).transpose(1,2) #each dim0 entry is a 3x3 tensor with columns a, b, c
        #M = torch.cat([a, b, c]).reshape(3,3) # has a, b, and c as rows
        M[0] = torch.eye(3)
        Minv = M.transpose(1, 2)
        # theta = eps * (torch.rand(aug_num, device=self.device)- 0.5) #sample random angle, does this run on GPU?
        cos = 1 - eps * (torch.rand(aug_num, device=self.device)) # sample random cos values
        sin = 1 - torch.sqrt(1-cos*cos)
        first_row = torch.tensor([1., 0., 0.], device=self.device).repeat(aug_num, 1)
        second_row = torch.vstack((torch.zeros(aug_num, device=self.device), cos, -sin)).transpose(0,1)
        third_row = torch.vstack((torch.zeros(aug_num, device=self.device), sin, cos)).transpose(0,1)
        base_rot = torch.cat((first_row, second_row, third_row), dim=1).reshape(aug_num, 3, 3).transpose(1,2)
        # base_rot = torch.tensor([[1., 0., 0.],
        #                             [0., torch.cos(theta), -torch.sin(theta)],
        #                             [0., torch.sin(theta), torch.cos(theta)]], device=self.device)
        # when smapling cos and sin directly
        # base_rot = torch.tensor([[1., 0., 0.],
        #                             [0., cos, - sin],
        #                             [0., sin, cos]], device=self.device)

        rot = Minv@base_rot@M

        self.model.train()
        n_epochs_stop = early_stop
        epochs_no_improve = 0
        min_val_loss = 1e8
        for epoch_idx in range(epochs):
            if epoch_idx %50 ==0:
                print('epoch nb:', epoch_idx)
            running_loss_cnc = 0.0 # what is this??
            y_pred = np.array([])
            y_true = np.array([])
            for data in self.train_loader:#self.train_set: #removed loader for now...
                data = data.to(self.device)

                #print('egde_inde: ', data.edge_index.shape)
                #print('x shape: ', data.x.shape)

                x = torch.transpose(rot@torch.transpose(data.x, 0, 1), 1, 2).reshape(-1, 3) #augments the data
                edge_index = data.edge_index.repeat(1, aug_num)
                data.batch = torch.arange(0, aug_num, device=self.device).unsqueeze(dim=1).repeat(1,data.x.shape[0]).reshape((-1,))
                data.segment = torch.tensor(data.segment, device=self.device).repeat(aug_num)
                data.y = torch.tensor(data.y, device=self.device).repeat(aug_num)
                data.x = x
                data.edge_index = edge_index

                self.optimizer.zero_grad()
                loss, loss_cnc, cnc_pred, cnc = self.get_losses(data)
                loss.backward()
                self.optimizer.step()
                pred = cnc_pred.argmax(dim=1)
                y_pred = np.append(y_pred, pred.cpu().detach().numpy())
                y_true = np.append(y_true, cnc.cpu().detach().numpy())
                running_loss_cnc += loss_cnc.item()

            train_loss_cnc = running_loss_cnc / len(self.train_loader.dataset)
            acc, prec, rec, sens, spec, f1score = self.calculate_metrics(y_pred, y_true)
            wandb.log({
                'ratio': self.ratio,
                'train_accuracy': acc,
                'train_precision': prec,
                'train_recall': rec,
                'train_loss_graph': train_loss_cnc
            })
            val_loss = self.evaluate(val_set=True)
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                return


    def train_aug_rot_per_batch(self, epochs, early_stop, eps = 0.1):
        '''trains model augmenting with rotations centered around avg and var angles.
        this is done uniformly for now...'''
        #self.model.eval()
        #self.evaluate(val_set=True)
        self.model.train()
        n_epochs_stop = early_stop
        epochs_no_improve = 0
        min_val_loss = 1e8
        for epoch_idx in range(epochs):
            print('epoch nb:', epoch_idx)
            running_loss_cnc = 0.0 # what is this??
            y_pred = np.array([])
            y_true = np.array([])
            for data in self.train_loader:
                data = data.to(self.device)

                # sample strategy from SO(3)
                a = torch.rand(3, device=self.device) #axis of rotation
                a = a.div(a.norm()) #normalize
                b = torch.rand(3, device=self.device)
                b = b.div(b.norm()) #normalize
                b = b - torch.dot(a, b)*a # perpendicular
                b = b.div(b.norm()) #normalize
                c = torch.cross(a, b) # third member of orthonormal basis
                M = torch.cat([a, b, c]).reshape(3,3) # has a, b, and c as rows
                Minv = torch.transpose(M, 0, 1)
                theta = eps * (torch.rand(1, device=self.device)- 0.5) #sample random angle, does this run on GPU?
                base_rot = torch.tensor([[1., 0., 0.],
                                         [0., torch.cos(theta), -torch.sin(theta)],
                                         [0., torch.sin(theta), torch.cos(theta)]], device=self.device)

                rot = Minv@base_rot@M

                data.x = torch.transpose(rot@torch.transpose(data.x, 0, 1), 0, 1) #this rotates with one specific angle

                self.optimizer.zero_grad()
                loss, loss_cnc, cnc_pred, cnc = self.get_losses(data)
                loss.backward()
                self.optimizer.step()
                pred = cnc_pred.argmax(dim=1)
                y_pred = np.append(y_pred, pred.cpu().detach().numpy())
                y_true = np.append(y_true, cnc.cpu().detach().numpy())
                running_loss_cnc += loss_cnc.item()

            train_loss_cnc = running_loss_cnc / len(self.train_loader.dataset)
            acc, prec, rec, sens, spec, f1score = self.calculate_metrics(y_pred, y_true)
            wandb.log({
                'ratio': self.ratio,
                'train_accuracy': acc,
                'train_precision': prec,
                'train_recall': rec,
                'train_loss_graph': train_loss_cnc
            })
            val_loss = self.evaluate(val_set=True)
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                return

    def evaluate(self, val_set): # val set is boolean, saying whether we use calidation or test set
        if val_set:
            dataloader = self.val_loader
            prefix = 'val'
        else:
            dataloader = self.test_loader
            prefix = 'test'
        self.model.eval()
        running_loss_cnc, running_loss = 0.0, 0.0
        y_pred = np.array([])
        y_true = np.array([])
        with torch.no_grad():
            for data in self.train_loader: 
                data = data.to(self.device)
                loss, loss_cnc, cnc_pred, cnc = self.get_losses(data)
                pred = cnc_pred.argmax(dim=1)
                y_pred = np.append(y_pred, pred.cpu().detach().numpy())
                y_true = np.append(y_true, cnc.cpu().detach().numpy())
                running_loss_cnc += loss_cnc.item()
                running_loss += loss.item()
        val_loss_cnc = running_loss_cnc / len(self.val_loader.dataset)
        val_loss = running_loss / len(self.val_loader.dataset)
        acc, prec, rec, sensitivity, specificity, f1_score = self.calculate_metrics(y_pred, y_true)
        wandb.log({
            prefix + '_accuracy': acc,
            prefix + '_precision': prec,
            prefix + '_recall': rec,
            prefix + '_sensitivity': sensitivity,
            prefix + '_specificity': specificity,
            prefix + '_f1score': f1_score,
            prefix + '_loss_graph': val_loss_cnc
        })
        return val_loss
