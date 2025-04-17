import os
import glob
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import normalize
import base64
from torchmetrics.classification import BinaryConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
from sklearn.manifold import TSNE
import seaborn as sns
np.set_printoptions(threshold=np.inf)
from target_shadow_nn_models import *
from opacus import PrivacyEngine
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
# import EarlyStopping
from early_stopping_pytorch import EarlyStopping

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    # if tempBool:
        # print("Elapsed time: %f seconds." % tempTimeInterval)
    return tempTimeInterval

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)  # The first call to toc() after this will measure from here


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class shadow_train_class():
    def __init__(self, trainloader, testloader, dataset_name, model, device, use_DP, noise, norm, delta):
        self.delta = delta
        self.use_DP = use_DP
        self.device = device
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        # set a codition on weight decay to be 5e-3 in case if dataset is purchase
        if dataset_name == 'purchase':
            self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)


        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-3)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)



    # Training
    def train(self):
        self.net.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            # outputs = self.model(inputs)
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if self.use_DP:
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
            # epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B5: %.3f \u03B4: 1e-5" % (epsilon))
                
        self.scheduler.step()

        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total


    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        # self.model.eval()
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total

     
class target_train_class():
    def __init__(self, trainloader, testloader, dataset_name, model, device, use_DP, noise, norm, delta, arch):
        self.use_DP = use_DP
        self.device = device
        self.delta = delta
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.arch = arch

        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        self.noise_multiplier, self.max_grad_norm = noise, norm

        
        if dataset_name == 'purchase':
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)

            if self.use_DP:
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                    module=model,
                    optimizer=self.optimizer,
                    data_loader=self.trainloader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )          
                print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))

            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)
        elif dataset_name == 'texas':
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9,  weight_decay=1e-4)

            if self.use_DP:
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                    module=model,
                    optimizer=self.optimizer,
                    data_loader=self.trainloader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )          
                print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))

            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif dataset_name == 'adult':
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9,  weight_decay=1e-4)

            if self.use_DP:
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                    module=model,
                    optimizer=self.optimizer,
                    data_loader=self.trainloader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )          
                print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))

            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

        else:

            if self.arch == 'vgg16':
                self.optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay = 0.005, momentum = 0.9)
            elif self.arch == 'wrn':
                # optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
                self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum = 0.9)
            else:
                self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2,  weight_decay=5e-4, momentum=0.9)

            if self.use_DP:
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                    module=model,
                    optimizer=self.optimizer,
                    data_loader=self.trainloader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )          
                print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))

            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)

        
        
        
        


        # self.noise_multiplier, self.max_grad_norm = noise, norm
        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)


        

      
    # Training
    def train(self):
        self.net.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if isinstance(targets, list):
                targets = targets[0]

            if str(self.criterion) != "CrossEntropyLoss()":
                targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()
             
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            # print(f"inputs size: {inputs.size()}, targets size: {targets.size()}")
            # exit()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            # self.scheduler.step()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if str(self.criterion) != "CrossEntropyLoss()":
                _, targets= targets.max(1)

            correct += predicted.eq(targets).sum().item()

        if self.use_DP:
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
            # epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B5: %.3f \u03B4: 1e-5" % (epsilon))
                
        self.scheduler.step()

        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total


    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)
        

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                if isinstance(targets, list):
                    targets = targets[0]
                if str(self.criterion) != "CrossEntropyLoss()":
                    targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if str(self.criterion) != "CrossEntropyLoss()":
                    _, targets= targets.max(1)

                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total

            
class attack_for_blackbox_com_Prev():
    def __init__(self,TARGET_PATH, Perturb_MODELS_PATH, ATTACK_SETS,ATTACK_SETS_PV_CSV, attack_train_loader, attack_test_loader, target_model, attack_model, perturb_model, device, dataset_name):
        self.device = device

        self.TARGET_PATH = TARGET_PATH
        # self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS
        self.ATTACK_SETS_PV_CSV = ATTACK_SETS_PV_CSV
        
        self.target_model = target_model.to(self.device)
        # self.shadow_model = shadow_model.to(self.device)
        self.Perturb_MODELS_PATH = Perturb_MODELS_PATH
        
        print( 'self.TARGET_PATH: %s' % self.TARGET_PATH)
     
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH, weights_only=True))
       
        self.target_model.eval()
        
        
        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        self.perturb_model = perturb_model.to(self.device)

        torch.manual_seed(0)
        self.attack_model.apply(weights_init)
        # self.perturb_model.apply(weights_init)
        # pred_component_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_attack = optim.Adam(self.attack_model.parameters(), lr=1e-3) # try lr=1e-5
       
      
        

   
    def _get_data(self, model, inputs, targets):
        
        result = model(inputs)
        # output, _ = torch.sort(result, descending=True) 
        # results = F.softmax(results[:,:5], dim=1)
        output = F.softmax(result, dim=1)
        _, predicts = result.max(1)

        prediction = predicts.eq(targets).float()
        
        # prediction = []
        # for predict in predicts:
        #     prediction.append([1,] if predict else [0,])

        # prediction = torch.Tensor(prediction)

        # final_inputs = torch.cat((results, prediction), 1)
        # print(final_inputs.shape)

        return output, prediction.unsqueeze(-1)

    def prepare_dataset_analyse(self):
        print("Preparing  and analysing the dataset")
        
        
        # Save train dataset to CSV
        with open(self.ATTACK_SETS_PV_CSV, "w", newline='') as f:
            writer = csv.writer(f)
            # Write the header row (optional)
            # Write the header row (optional, adjust depending on the number of output dimensions)
            num_output_classes = 10  # Assuming output size is [batch_size, num_classes]
            header = ["Output_" + str(i) for i in range(num_output_classes)] + ["Prediction", "Members", "Targets"]
            writer.writerow(header)
            
            # writer.writerow(["Output", "Prediction", "Members", "Targets"])
            
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if 1:
                    output, prediction = self._get_data(self.shadow_model, inputs, targets)

                    # Write each batch as rows in the CSV file
                    for i in range(output.shape[0]):
                        # Unpack the output (PV) to individual elements
                        row = output[i].cpu().tolist() + [  # Unpacking the prediction vector (each element becomes a column)
                            prediction[i].item(),           # Correct or wrong (0/1)
                            members[i].item(),              # Membership (0/1)
                            targets[i].item()               # Target label
                        ]
                        writer.writerow(row)

        print("Finished Saving Train Dataset")


    

        # with open(self.ATTACK_SETS + "test.p", "wb") as f:
        #     for inputs, targets, members in self.attack_test_loader:
        #         inputs, targets = inputs.to(self.device), targets.to(self.device)
        #         if inputs.size()[0] == 64:
                    
        #             output, prediction = self._get_data(self.target_model, inputs, targets)
        #             # output = output.cpu().detach().numpy()
                
        #             pickle.dump((output, prediction, members, targets), f)
        #         else:
        #              print("test data skipping: ",inputs.size()[0])
    
    
    def prepare_dataset(self):
        print("Preparing dataset")
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.size()[0] == 64:
                    
                    # change the self.shadow_model to self.target_model to get the PVs assuming shadow model performance is the same as the target model
                    output, prediction = self._get_data(self.target_model, inputs, targets)
                    # output = output.cpu().detach().numpy()
                    # output size: torch.Size([64, 10]), prediction size: torch.Size([64, 1]), members: torch.Size([64]), batch of 64
                    # prediction: a specific sample in the batch is predicted correct (1) or predicted wrong (0)
                    # output: Not PVs but raw 10 logits (based on the number of classes)
                    # print(f"output size: {output.shape}, prediction size: {prediction.shape}, members: {members.shape}")
                    # print(output)
                    # print(prediction)
                    # print(members)
                    # print(targets)
                    # exit()
                    pickle.dump((output, prediction, members, targets), f)
                else:
                    print("skipping: ",inputs.size()[0])


        # # Load the data from train.p and check members
        # with open(self.ATTACK_SETS + "train.p", "rb") as f:
        #     all_members = []
        #     while True:
        #         try:
        #             output, prediction, members, targets = pickle.load(f)
        #             all_members.extend(members)
        #         except EOFError:
        #             break

        # # Convert all_members to a tensor
        # all_members = torch.tensor(all_members)

        # # Count the number of 0s and 1s
        # num_zeros = torch.sum(all_members == 0).item()
        # num_ones = torch.sum(all_members == 1).item()

        # print(f"Number of 0s in members: {num_zeros}")
        # print(f"Number of 1s in members: {num_ones}")
        
        # exit()

        print("Finished Saving Train Dataset")
    

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.size()[0] == 64:
                    
                    output, prediction = self._get_data(self.target_model, inputs, targets)
                    # output = output.cpu().detach().numpy()
                
                    pickle.dump((output, prediction, members, targets), f)
                else:
                     print("test data skipping: ",inputs.size()[0])
        
        # Load the data from train.p and check members
        # with open(self.ATTACK_SETS + "test.p", "rb") as f:
        #     all_members = []
        #     while True:
        #         try:
        #             output, prediction, members, targets = pickle.load(f)
        #             all_members.extend(members)
        #         except EOFError:
        #             break

        # # Convert all_members to a tensor
        # all_members = torch.tensor(all_members)

        # # Count the number of 0s and 1s
        # num_zeros = torch.sum(all_members == 0).item()
        # num_ones = torch.sum(all_members == 1).item()

        # print(f"Number of 0s in members: {num_zeros}")
        # print(f"Number of 1s in members: {num_ones}")

        # Create the dataset from the pickle file
        # pickle_path = self.ATTACK_SETS + "train.p"
        
        self.dataset = AttackDataset(self.ATTACK_SETS + "train.p")

        print("Finished Saving Test Dataset")
        return self.dataset
        # exit()
        
    def prepare_dataset_mul(self, num_classes):

        batch_size = 8
        # read the whole attack_train_loader batche by batch 
        # put the samples into corresponding buckets
        
        # transform each k-bucket into batches using dataloader
        # loade from dataloader, get the corresponing predictions
        # save into train_i.p file
        
    
        #! Traing Data class buckets 
        with torch.no_grad():
            counter=0
            print(f"classes: {num_classes}")
            for class_name in range(num_classes):
                # class_name = 29
                output_coll =  torch.empty((0, num_classes))
                predictions_coll = torch.empty((0, 1))
                members_coll = torch.empty((0, 1))
                targets_Coll = torch.empty((0, 1))
            
                file_path = self.ATTACK_SETS + f"_train_{class_name}.p"
                # print(f"new path: {file_path}")
                
                counter = 0
                with open(file_path, "wb") as f:
                    # This loop will iterate over all batches and put samples in their corresponding class file
                    for inputs, targets, members in self.attack_train_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        if 32 == 32:
                            counter+=1
                            # Assuming _get_data returns output and prediction based on the class
                            output, prediction = self._get_data(self.shadow_model, inputs, targets)
                            # print(f"output : {output.size()}")
                            # exit()
                            # print(f"inputs: {inputs[0]} \n\nand targets: {targets[0]}")
                            # print(f"CUDA sumary: {torch.cuda.memory_summary()}")
                            # print(f"allocated memory: {torch.cuda.memory_allocated()}")
                            output, prediction, targets  = output.cpu(), prediction.cpu(), targets.cpu()
                            # Find the indices where the value is class_name
                            # print(f"type of output: {type(output)}, device: {output.device}")
                            # print(f"type of prediction: {type(prediction)}, device: {prediction.device}")
                            # print(f"type of targets: {type(targets)}, device: {targets.device}")
                            # exit()
                            # print(f"targets: {targets}\n class_name: {class_name}")
                            indices = torch.where(targets == class_name)[0]
                            # print(f"indices in the batch: {indices}")
                            # exit()
                            # print(f"output_coll device: {output_coll.device}, output device: {output.device}")
                            output_coll = torch.vstack((output_coll, output[indices]))
                            # print(f"output_coll: {output_coll.size()}")
                            predictions_coll = torch.vstack((predictions_coll, prediction[indices]))
                            # print(f"targets_Coll size: {targets_Coll.size()}, targets size: {targets[indices].unsqueeze(1).size()}")
                            targets_Coll = torch.vstack((targets_Coll, targets[indices].unsqueeze(1)))
                            members_coll = torch.vstack((members_coll, members[indices].unsqueeze(1)))
                            
                            
                            
                        
                            
                        else:
                            print("skipping: ", inputs.size()[0])
                        
                        del inputs
                        del targets
                        torch.cuda.empty_cache()
                    
                    # print(f"Class {class_name} information")
                    # print(f"output_coll class {class_name}: {output_coll.size()} and size: {output_coll.size()[0]}")
                    # print(f"predictions: {predictions_coll.size()}")
                    # print(f"members: {members_coll.size()}")
                    # print(f"targets: {targets_Coll.size()}")
                    # exit()
                    # print(f"counter: {counter}")
            
                    # save in class_i file 
                    # attack_train = (output_coll, predictions_coll.squeeze(), members_coll.squeeze(), targets_Coll.squeeze())
                    attack_train = []
                    for i in range(output_coll.size()[0]):
                        attack_train.append((output_coll[i], predictions_coll[i].item(), members_coll[i].item(), targets_Coll[i].item()))
                        
                    # print(f"output_coll: {attack_train[0].shape}, predictions_coll : {attack_train[1].shape}, members_coll: {attack_train[2].shape}, targets_coll : {attack_train[3].shape}")
                    # print(f"attack_train size: {attack_train[0]}, len: {len(attack_train)}")
                    
                    # get track of the dimension of the dataset and append for later use
                    attack_trainloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
                    for output_coll, predictions_coll, members_coll, targets_Coll  in attack_trainloader:
                        # print(f"len of batch output_coll: {len(output_coll)}, and size: {output_coll.shape}")
                        if output_coll.size()[0] == batch_size:
                            pickle.dump((output_coll, predictions_coll, members_coll, targets_Coll), f)
                        # output, prediction, members = pickle.load(f)
                        else:
                            print(f"skipping the last {output_coll.size()[0]} samples")
                     
            #     exit()
            # exit()
        print(f"Finished Saving {num_classes} Train Dataset")   
            
            
        
        #! Test Data class buckets 
        with torch.no_grad():
            counter=0
            print(f"classes: {num_classes}")
            for class_name in range(num_classes):
                output_coll =  torch.empty((0, num_classes))
                predictions_coll = torch.empty((0, 1))
                members_coll = torch.empty((0, 1))
                targets_Coll = torch.empty((0, 1))
            
                file_path = self.ATTACK_SETS + f"_test_{class_name}.p"
                # print(f"new path: {file_path}")
                
                counter = 0
            
                with open(file_path, "wb") as f:
                    for inputs, targets, members in self.attack_test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        if 32 == 32:
                            counter+=1
                            output, prediction = self._get_data(self.target_model, inputs, targets)
                            # output = output.cpu().detach().numpy()
                            
                            
                            # print(f"output : {output.size()}")
                            # exit()
                            # print(f"inputs: {inputs[0]} \n\nand targets: {targets[0]}")
                            # print(f"CUDA sumary: {torch.cuda.memory_summary()}")
                            # print(f"allocated memory: {torch.cuda.memory_allocated()}")
                            output, prediction, targets  = output.cpu(), prediction.cpu(), targets.cpu()
                            # Find the indices where the value is class_name
                            # print(f"type of output: {type(output)}, device: {output.device}")
                            # print(f"type of prediction: {type(prediction)}, device: {prediction.device}")
                            # print(f"type of targets: {type(targets)}, device: {targets.device}")
                            # exit()
                            indices = torch.where(targets == class_name)[0]
                            # print(f"indices: {indices}")
                            # exit()
                            # print(f"output_coll device: {output_coll.device}, output device: {output.device}")
                            output_coll = torch.vstack((output_coll, output[indices]))
                            # print(f"output_coll: {output_coll.size()}")
                            predictions_coll = torch.vstack((predictions_coll, prediction[indices]))
                            # print(f"targets_Coll size: {targets_Coll.size()}, targets size: {targets[indices].unsqueeze(1).size()}")
                            targets_Coll = torch.vstack((targets_Coll, targets[indices].unsqueeze(1)))
                            members_coll = torch.vstack((members_coll, members[indices].unsqueeze(1)))
                            
                            # pickle.dump((output, prediction, members), f)
                        else:
                            print("test data skipping: ",inputs.size()[0])
                        
                        del inputs
                        del targets
                        torch.cuda.empty_cache()
                    
                    # print(f"Class {class_name} information Training")
                    # print(f"output_coll class {class_name}: {output_coll.size()} and size: {output_coll.size()[0]}")
                    # print(f"predictions: {predictions_coll.size()}")
                    # print(f"members: {members_coll.size()}")
                    # print(f"targets: {targets_Coll.size()}")
                    
            
                    # save in class_i file 
                    # attack_train = (output_coll, predictions_coll.squeeze(), members_coll.squeeze(), targets_Coll.squeeze())
                    attack_train = []
                    for i in range(output_coll.size()[0]):
                        attack_train.append((output_coll[i], predictions_coll[i].item(), members_coll[i].item(), targets_Coll[i].item()))
                        
                    # print(f"output_coll: {attack_train[0].shape}, predictions_coll : {attack_train[1].shape}, members_coll: {attack_train[2].shape}, targets_coll : {attack_train[3].shape}")
                    # print(f"attack_test size: {attack_train[0]}, len: {len(attack_train)}")
                    
                    # get track of the dimension of the dataset and append for later use
                    attack_testloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
                    for output_coll, predictions_coll, members_coll, targets_Coll  in attack_testloader:
                        # print(f"len of batch output_coll: {len(output_coll)}, and size: {output_coll.shape}")
                        if output_coll.size()[0] == batch_size:
                            pickle.dump((output_coll, predictions_coll, members_coll, targets_Coll), f)
                        # output, prediction, members = pickle.load(f)
                        else:
                            print(f"skipping the last {output_coll.size()[0]} samples")
                        
                
         
        print(f"Finished Saving {num_classes} Test Dataset")   
        # exit()

   
    def train(self, epoch, result_path, result_path_csv):
        self.attack_model.train()
        # self.perturb_model.train()

        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        # for output, prediction, members, targets in dataloader:
        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break

                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)
                # targets can remain on CPU or be moved if needed

                

                # ----- Step 5: Forward Pass through the Attack Model -----
                results = self.attack_model(output, prediction, targets)
                results = F.softmax(results, dim=1)
                # ----- Step 6: Compute Loss -----
                attack_loss = self.criterion(results, members) # need to confirm if need to use results = F.softmax(results, dim=1) here
               
               
                # ----- Step 7: Backpropagation -----
                self.optimizer_attack.zero_grad()
                attack_loss.backward()
                self.optimizer_attack.step()
        
               
                # ----- Step 8: Metrics Calculation -----
                train_loss += attack_loss.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                # Collect predictions for metrics.
                if epoch:
                    final_train_gndtrth.append(members)
                    final_train_predict.append(predicted)
                    final_train_probabe.append(results[:, 1])

                batch_idx += 1

        # Post Epoch Evaluation
        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.extend([
                100. * correct / total,
                (prec / batch_idx).item(),
                (recall / batch_idx).item(),
                train_f1_score,
                train_roc_auc_score,
            ])

            # Save Results
            with open(result_path, "wb") as f_out:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f_out)

            with open(result_path_csv, "w") as f_out:
                pickled_data = pickle.dumps((final_train_gndtrth, final_train_predict, final_train_probabe))
                encoded_data = base64.b64encode(pickled_data)
                f_out.write(encoded_data.decode("utf-8"))

            print("Saved Attack Train Ground Truth and Predict Sets")
            print(f"Train F1: {train_f1_score:.6f}\nAUC: {train_roc_auc_score:.6f}")

        train_loss = train_loss/batch_idx
        # print('Train Acc: %.3f%% (%d/%d) | Loss: %.3f precision: %.3f recall: %.3f' %
            # (100. * correct / total, correct, total, train_loss / batch_idx, 100 * prec / batch_idx, 100 * recall / batch_idx))
        print(f"Train Acc: {100. * correct / total:.3f}% ({correct}/{total}) | "
          f"Loss: {train_loss:.3f} | "
          f"Precision: {100. * train_loss / batch_idx:.3f} | "
          f"Recall: {100. * train_loss / batch_idx:.3f}")
        
    
    def test(self, epoch, result_path):
        self.attack_model.eval()
       

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    # Move tensors to device.
                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)
                    # targets can remain on CPU or be moved if needed.

                    
                    # ----- Step 6: Forward Pass through the Attack Model -----
                    results = self.attack_model(output, prediction, targets)
                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        # ----- Post Evaluation -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx
      
        final_result.extend([
        correct / total,
        (prec / batch_idx).item(),
        (recall / batch_idx).item(),
        test_f1_score,
        test_roc_auc_score,
        avg_test_loss  # Append average test loss
        ])

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

       
        print(f"Test Acc: {100.*correct/(1.0*total):.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, precision: {100.*prec/(1.0*batch_idx):.3f}, recall: {100.*recall/batch_idx:.3f}")

        
        return final_result
   
    def compute_roc_curve(self, models_apth, plot=True, save_path=None):
       
        checkpoint = torch.load(models_apth, map_location=self.device, weights_only=True)
        # Load state dictionaries for the models.
        self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        # self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        # Restore the learned threshold parameters.
        # self.cosine_threshold.data = checkpoint['cosine_threshold'].to(self.device)
        # self.Entropy_quantile_threshold.data = checkpoint['Entropy_quantile_threshold'].to(self.device)
        
        # Set models to evaluation mode.
        self.attack_model.eval()
        # self.perturb_model.eval()
        
        
        
        final_ground_truth = []
        final_probabilities = []
        
        with torch.no_grad():
            test_file = self.ATTACK_SETS + "test.p"
            with open(test_file, "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    # Move tensors to device.
                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)
                    # (targets can remain on CPU if not used for perturbation)

                    

                    # Forward pass through the attack model.
                    results = self.attack_model(output, prediction, targets)
                    results = F.softmax(results, dim=1)
                    # results = self.attack_model(perturbed_pvs, prediction, targets)
                    # results = F.softmax(results, dim=1)

                    # _, predicted = results.max(dim=1)
                    # We assume that column 1 gives the probability for membership.
                    probabilities = results[:, 1]
                    # print(f"First few probabilities: {probabilities[:5]}")
                    # print(f"First few members: {members[:5]}")
                    # print(f"predicted: {predicted[:5]}")
                    # exit()

                    final_ground_truth.append(members.cpu())
                    final_probabilities.append(probabilities.cpu())
                    # exit()
            # Concatenate collected results.
            final_ground_truth = torch.cat(final_ground_truth, dim=0).numpy()
            final_probabilities = torch.cat(final_probabilities, dim=0).numpy()
        
        # Compute the ROC curve and ROC AUC.
        fpr, tpr, thresholds = roc_curve(final_ground_truth, final_probabilities)
        roc_auc = auc(fpr, tpr)

        # if plot:
        #     plt.figure(figsize=(8, 6))
        #     plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc, lw=2)
        #     plt.plot([0, 1], [0, 1], 'k--', lw=2)
        #     plt.xlim([0.0, 1.0])
        #     plt.ylim([0.0, 1.05])
        #     plt.xlabel("False Positive Rate")
        #     plt.ylabel("True Positive Rate")
        #     plt.title("Receiver Operating Characteristic")
        #     plt.legend(loc="lower right")
        #     if save_path is not None:
        #         plt.savefig(save_path)
        #     plt.show()

        return fpr, tpr, thresholds, roc_auc

  

            
    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)
    
    def save_pertub_Model(self, path):
        torch.save(self.perturb_model.state_dict(), self.Perturb_MODELS_PATH)
    # chch
    def save_att_per_thresholds_models(self, path):
        models_threshold_params = {
            'attack_model_state_dict': self.attack_model.state_dict(),
            # 'perturb_model_state_dict': self.perturb_model.state_dict(),
            # 'cosine_threshold': self.cosine_threshold.detach().cpu(),
            # 'Entropy_quantile_threshold': self.Entropy_quantile_threshold.detach().cpu()
        }
        torch.save(models_threshold_params, path)
        
    def load_perturb_model(self):

        # gan_path = self.Perturb_MODELS_PATH
        # generator = Generator(input_dim).to(device)
        # self.perturb_model = perturb_model.to(self.device)

        self.perturb_model.load_state_dict(torch.load(self.Perturb_MODELS_PATH, weights_only=True))
        self.perturb_model.eval()  # Set the generator to evaluation mode
        return self.perturb_model

   
def get_ent_lr(acc_gap, max_lr=0.005, k=10, mid=0.5):
    return max_lr * (1 - 1 / (1 + np.exp(-k * (acc_gap - mid))))

def get_cs_lr(acc_gap, max_lr=0.01, k=10, mid=0.5):
    return max_lr * (1 - 1 / (1 + np.exp(-k * (acc_gap - mid))))

def sigmoid_adaptive_lr(gap, mid_gap=0.475, gap_range=0.55, max_lr=0.1, min_lr=0.001):
    """
    Adaptive sigmoid-based learning rate scheduler.

    Parameters:
    - gap (float): Accuracy gap (0 to 1 scale)
    - mid_gap (float): Center point of the sigmoid (e.g., 0.475 for 47.5%)
    - gap_range (float): Controls steepness (difference between upper and lower bounds; smaller = steeper)
    - max_lr (float): Maximum learning rate
    - min_lr (float): Minimum learning rate

    Returns:
    - lr (float): Adapted learning rate
    """
    k = 10 / gap_range  # Steepness from gap_range (e.g., 0.55 gives k ≈ 18.18)
    sigmoid = 1 / (1 + np.exp(-k * (gap - mid_gap)))
    return min_lr + (max_lr - min_lr) * sigmoid

def sigmoid_adaptive_lr(gap, mid_gap=0.45, gap_range=0.65, max_lr=0.1, min_lr=0.003):
    k = 10 / gap_range  # Controls steepness
    sigmoid = 1 / (1 + np.exp(-k * (gap - mid_gap)))
    return min_lr + (max_lr - min_lr) * sigmoid

# attack_for_blackbox_com(TARGET_PATH, ATTACK_SETS,ATTACK_SETS_PV_CSV, attack_trainloader, attack_testloader, target_model, attack_model,perturb_model,  device)
class attack_for_blackbox_com_NEW():
    def __init__(self,TARGET_PATH, Perturb_MODELS_PATH, ATTACK_SETS,ATTACK_SETS_PV_CSV, attack_train_loader, attack_test_loader, target_model, attack_model, perturb_model, device, dataset_name, attack_name, num_classes, acc_gap):
        self.device = device

        self.TARGET_PATH = TARGET_PATH
        # self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS
        self.ATTACK_SETS_PV_CSV = ATTACK_SETS_PV_CSV
        
        self.target_model = target_model.to(self.device)
        # self.shadow_model = shadow_model.to(self.device)
        self.Perturb_MODELS_PATH = Perturb_MODELS_PATH
        self.attack_name = attack_name
        print( 'self.TARGET_PATH: %s' % self.TARGET_PATH)
        # exit()
        # self.pred_component_model = pred_component_model
        self.num_classes = num_classes
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH, weights_only=True))
        # self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH, weights_only=True))

        self.target_model.eval()
        # self.shadow_model.eval()
        self.member_mean = 0.0
        self.member_std = 0.0
        self.non_member_mean = 0.0
        self.non_member_std  = 0.0

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        self.perturb_model = perturb_model.to(self.device)
        self.patience = 20
        self.early_stopping = EarlyStopping(self.patience, verbose=True)
        # torch.manual_seed(0)
        self.attack_model.apply(weights_init)
        self.perturb_model.apply(weights_init)
  
        self.criterion = nn.CrossEntropyLoss()
        # fff
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-3)
        self.optimizer_perturb = optim.Adam(self.perturb_model.parameters(), lr=1e-3) # need to change the learning rate to see the effect latter
        
        

        self.dataset_name = dataset_name
    
      
       
        dataset_k_values = {
            "stl10": 1000.0,
            "cifar100": 1000.0,
            # "location": 1000.0,
            "fmnist": 1000000.0,
            "purchase": 1000000.0,
            "cifar10": 10000.0,
            "utkface": 1000000,
            "texas": 10000,
            "adult": 1000000,
        }

        self.k = dataset_k_values.get(dataset_name)  # Default to 1000000 if dataset_name is not found
    
        if dataset_name == "fmnist" or dataset_name == "cifar10" or dataset_name == "utkface":
            self.k1 = 1000000.0
            # self.k1 = 10.0

        else:
            self.k1 = 10.0

        # NON-image datasets
        if dataset_name == "location":
            self.k = 1000.0
            self.k1 = 10000.0   
        if dataset_name == "texas":
            self.k = 10000.0
            self.k1 = 10.0
        if dataset_name == "adult":
            self.k = 1000000.0
            self.k1 = 100.0
        if dataset_name == "purchase":
            self.k = 1000000.0
            self.k1 = 100.0 # was 10.0
        

        # Image datasets
        if dataset_name == "stl10":
            self.k = 1000.0
            self.k1 = 10.0
        elif dataset_name == "cifar100":
            self.k = 1000.0
            self.k1 = 10.0
        elif dataset_name == "cifar10":
            self.k = 10000.0
            self.k1 = 1000000.0 
        elif dataset_name == "utkface":
            self.k = 1000000.0
            self.k1 = 1000000.0
        elif dataset_name == "fmnist":
            self.k = 1000000.0
            self.k1 = 1000000.0
        
    
    
        # Note: The following has been used as fixed optimized rates to generate results in the paper
        if dataset_name == "cifar100":
            cs_lr = 0.01
            ent_lr = 0.1
        elif dataset_name == "cifar10":
            cs_lr = 0.01
            ent_lr = 0.001
        elif dataset_name == "fmnist":
            cs_lr = 0.01
            ent_lr = 0.001
        elif dataset_name == "utkface":
            cs_lr = 0.01
            ent_lr = 0.001
        elif dataset_name == "purchase":
            cs_lr = 0.01 # was 0.001 both
            ent_lr = 0.01
        elif dataset_name == "location":
            cs_lr = 0.01
            ent_lr = 0.01
        elif dataset_name == "adult":
            cs_lr = 0.01
            ent_lr = 0.01 # was 0.1
        elif dataset_name == "texas":
            cs_lr = 0.001
            ent_lr = 0.001
        else: # stl10
            cs_lr = 0.01
            ent_lr = 0.001



        # Note: the following were used to test fixed rates for VGG16-CIFAR10-5K
        # cs_lr = 0.01
        # ent_lr = 0.001

        # Note: the following were used (in the paper) to test sgimoid-target acc gap based rates for VGG16, Without constrastive Loss
        
        # cs_lr = get_cs_lr(acc_gap)
        # ent_lr = get_ent_lr(acc_gap)
        

        self.cosine_threshold = nn.Parameter(torch.tensor(0.5, device=self.device))
        self.Entropy_quantile_threshold = nn.Parameter(torch.tensor(0.5, device=self.device))


        self.optimizer_cosine = optim.Adam([self.cosine_threshold], cs_lr)
        self.optimizer_quantile_threshold = optim.Adam([self.Entropy_quantile_threshold], lr=ent_lr)

        self.kl_threshold = torch.nn.Parameter(torch.tensor(0.5))
        kl_lr = 0.01
        self.optimizer_kl = torch.optim.Adam([self.kl_threshold], kl_lr)

     

    
    def _get_data(self, model, inputs, targets):
        
        result = model(inputs)
        # output, _ = torch.sort(result, descending=True) 
        # results = F.softmax(results[:,:5], dim=1)
        output = F.softmax(result, dim=1)
        _, predicts = result.max(1)

        prediction = predicts.eq(targets).float()
        
        # prediction = []
        # for predict in predicts:
        #     prediction.append([1,] if predict else [0,])

        # prediction = torch.Tensor(prediction)

        # final_inputs = torch.cat((results, prediction), 1)
        # print(final_inputs.shape)

        return output, prediction.unsqueeze(-1)

    def prepare_dataset_analyse(self):
        print("Preparing  and analysing the dataset")
        
        
        # Save train dataset to CSV
        with open(self.ATTACK_SETS_PV_CSV, "w", newline='') as f:
            writer = csv.writer(f)
            # Write the header row (optional)
            # Write the header row (optional, adjust depending on the number of output dimensions)
            num_output_classes = 10  # Assuming output size is [batch_size, num_classes]
            header = ["Output_" + str(i) for i in range(num_output_classes)] + ["Prediction", "Members", "Targets"]
            writer.writerow(header)
            
            # writer.writerow(["Output", "Prediction", "Members", "Targets"])
            
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if 1:
                    output, prediction = self._get_data(self.shadow_model, inputs, targets)

                    # Write each batch as rows in the CSV file
                    for i in range(output.shape[0]):
                        # Unpack the output (PV) to individual elements
                        row = output[i].cpu().tolist() + [  # Unpacking the prediction vector (each element becomes a column)
                            prediction[i].item(),           # Correct or wrong (0/1)
                            members[i].item(),              # Membership (0/1)
                            targets[i].item()               # Target label
                        ]
                        writer.writerow(row)

        print("Finished Saving Train Dataset")


    

        # with open(self.ATTACK_SETS + "test.p", "wb") as f:
        #     for inputs, targets, members in self.attack_test_loader:
        #         inputs, targets = inputs.to(self.device), targets.to(self.device)
        #         if inputs.size()[0] == 64:
                    
        #             output, prediction = self._get_data(self.target_model, inputs, targets)
        #             # output = output.cpu().detach().numpy()
                
        #             pickle.dump((output, prediction, members, targets), f)
        #         else:
        #              print("test data skipping: ",inputs.size()[0])
    
    
    def prepare_dataset(self):
        print("Preparing dataset")
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.size()[0] == 64:
                    
                    # change the self.shadow_model to self.target_model to get the PVs assuming shadow model performance is the same as the target model
                    output, prediction = self._get_data(self.target_model, inputs, targets)
                    # output = output.cpu().detach().numpy()
                    # output size: torch.Size([64, 10]), prediction size: torch.Size([64, 1]), members: torch.Size([64]), batch of 64
                    # prediction: a specific sample in the batch is predicted correct (1) or predicted wrong (0)
                    # output: Not PVs but raw 10 logits (based on the number of classes)
                    # print(f"output size: {output.shape}, prediction size: {prediction.shape}, members: {members.shape}")
                    # print(output)
                    # print(prediction)
                    # print(members)
                    # print(targets)
                    # exit()
                    pickle.dump((output, prediction, members, targets), f)
                else:
                    print("skipping: ",inputs.size()[0])


        # # Load the data from train.p and check members
        # with open(self.ATTACK_SETS + "train.p", "rb") as f:
        #     all_members = []
        #     while True:
        #         try:
        #             output, prediction, members, targets = pickle.load(f)
        #             all_members.extend(members)
        #         except EOFError:
        #             break

        # # Convert all_members to a tensor
        # all_members = torch.tensor(all_members)

        # # Count the number of 0s and 1s
        # num_zeros = torch.sum(all_members == 0).item()
        # num_ones = torch.sum(all_members == 1).item()

        # print(f"Number of 0s in members: {num_zeros}")
        # print(f"Number of 1s in members: {num_ones}")
        
        # exit()

        print("Finished Saving Train Dataset")
    

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.size()[0] == 64:
                    
                    output, prediction = self._get_data(self.target_model, inputs, targets)
                    # output = output.cpu().detach().numpy()
                
                    pickle.dump((output, prediction, members, targets), f)
                else:
                     print("test data skipping: ",inputs.size()[0])
        
        # Load the data from train.p and check members
        # with open(self.ATTACK_SETS + "test.p", "rb") as f:
        #     all_members = []
        #     while True:
        #         try:
        #             output, prediction, members, targets = pickle.load(f)
        #             all_members.extend(members)
        #         except EOFError:
        #             break

        # # Convert all_members to a tensor
        # all_members = torch.tensor(all_members)

        # # Count the number of 0s and 1s
        # num_zeros = torch.sum(all_members == 0).item()
        # num_ones = torch.sum(all_members == 1).item()

        # print(f"Number of 0s in members: {num_zeros}")
        # print(f"Number of 1s in members: {num_ones}")

        # Create the dataset from the pickle file
        # pickle_path = self.ATTACK_SETS + "train.p"
        
        self.dataset = AttackDataset(self.ATTACK_SETS + "train.p")

        # self.member_mean, self.member_std, self.non_member_mean, self.non_member_std = self.approximate_perturbation_distribution()


        print("Finished Saving Test Dataset")
        return self.dataset
        # exit()
        
    def prepare_dataset_mul(self, num_classes):

        batch_size = 8
        # read the whole attack_train_loader batche by batch 
        # put the samples into corresponding buckets
        
        # transform each k-bucket into batches using dataloader
        # loade from dataloader, get the corresponing predictions
        # save into train_i.p file
        
    
        #! Traing Data class buckets 
        with torch.no_grad():
            counter=0
            print(f"classes: {num_classes}")
            for class_name in range(num_classes):
                # class_name = 29
                output_coll =  torch.empty((0, num_classes))
                predictions_coll = torch.empty((0, 1))
                members_coll = torch.empty((0, 1))
                targets_Coll = torch.empty((0, 1))
            
                file_path = self.ATTACK_SETS + f"_train_{class_name}.p"
                # print(f"new path: {file_path}")
                
                counter = 0
                with open(file_path, "wb") as f:
                    # This loop will iterate over all batches and put samples in their corresponding class file
                    for inputs, targets, members in self.attack_train_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        if 32 == 32:
                            counter+=1
                            # Assuming _get_data returns output and prediction based on the class
                            output, prediction = self._get_data(self.shadow_model, inputs, targets)
                            # print(f"output : {output.size()}")
                            # exit()
                            # print(f"inputs: {inputs[0]} \n\nand targets: {targets[0]}")
                            # print(f"CUDA sumary: {torch.cuda.memory_summary()}")
                            # print(f"allocated memory: {torch.cuda.memory_allocated()}")
                            output, prediction, targets  = output.cpu(), prediction.cpu(), targets.cpu()
                            # Find the indices where the value is class_name
                            # print(f"type of output: {type(output)}, device: {output.device}")
                            # print(f"type of prediction: {type(prediction)}, device: {prediction.device}")
                            # print(f"type of targets: {type(targets)}, device: {targets.device}")
                            # exit()
                            # print(f"targets: {targets}\n class_name: {class_name}")
                            indices = torch.where(targets == class_name)[0]
                            # print(f"indices in the batch: {indices}")
                            # exit()
                            # print(f"output_coll device: {output_coll.device}, output device: {output.device}")
                            output_coll = torch.vstack((output_coll, output[indices]))
                            # print(f"output_coll: {output_coll.size()}")
                            predictions_coll = torch.vstack((predictions_coll, prediction[indices]))
                            # print(f"targets_Coll size: {targets_Coll.size()}, targets size: {targets[indices].unsqueeze(1).size()}")
                            targets_Coll = torch.vstack((targets_Coll, targets[indices].unsqueeze(1)))
                            members_coll = torch.vstack((members_coll, members[indices].unsqueeze(1)))
                            
                            
                            
                        
                            
                        else:
                            print("skipping: ", inputs.size()[0])
                        
                        del inputs
                        del targets
                        torch.cuda.empty_cache()
                    
                    # print(f"Class {class_name} information")
                    # print(f"output_coll class {class_name}: {output_coll.size()} and size: {output_coll.size()[0]}")
                    # print(f"predictions: {predictions_coll.size()}")
                    # print(f"members: {members_coll.size()}")
                    # print(f"targets: {targets_Coll.size()}")
                    # exit()
                    # print(f"counter: {counter}")
            
                    # save in class_i file 
                    # attack_train = (output_coll, predictions_coll.squeeze(), members_coll.squeeze(), targets_Coll.squeeze())
                    attack_train = []
                    for i in range(output_coll.size()[0]):
                        attack_train.append((output_coll[i], predictions_coll[i].item(), members_coll[i].item(), targets_Coll[i].item()))
                        
                    # print(f"output_coll: {attack_train[0].shape}, predictions_coll : {attack_train[1].shape}, members_coll: {attack_train[2].shape}, targets_coll : {attack_train[3].shape}")
                    # print(f"attack_train size: {attack_train[0]}, len: {len(attack_train)}")
                    
                    # get track of the dimension of the dataset and append for later use
                    attack_trainloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
                    for output_coll, predictions_coll, members_coll, targets_Coll  in attack_trainloader:
                        # print(f"len of batch output_coll: {len(output_coll)}, and size: {output_coll.shape}")
                        if output_coll.size()[0] == batch_size:
                            pickle.dump((output_coll, predictions_coll, members_coll, targets_Coll), f)
                        # output, prediction, members = pickle.load(f)
                        else:
                            print(f"skipping the last {output_coll.size()[0]} samples")
                     
            #     exit()
            # exit()
        print(f"Finished Saving {num_classes} Train Dataset")   
            
            
        
        #! Test Data class buckets 
        with torch.no_grad():
            counter=0
            print(f"classes: {num_classes}")
            for class_name in range(num_classes):
                output_coll =  torch.empty((0, num_classes))
                predictions_coll = torch.empty((0, 1))
                members_coll = torch.empty((0, 1))
                targets_Coll = torch.empty((0, 1))
            
                file_path = self.ATTACK_SETS + f"_test_{class_name}.p"
                # print(f"new path: {file_path}")
                
                counter = 0
            
                with open(file_path, "wb") as f:
                    for inputs, targets, members in self.attack_test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        if 32 == 32:
                            counter+=1
                            output, prediction = self._get_data(self.target_model, inputs, targets)
                            # output = output.cpu().detach().numpy()
                            
                            
                            # print(f"output : {output.size()}")
                            # exit()
                            # print(f"inputs: {inputs[0]} \n\nand targets: {targets[0]}")
                            # print(f"CUDA sumary: {torch.cuda.memory_summary()}")
                            # print(f"allocated memory: {torch.cuda.memory_allocated()}")
                            output, prediction, targets  = output.cpu(), prediction.cpu(), targets.cpu()
                            # Find the indices where the value is class_name
                            # print(f"type of output: {type(output)}, device: {output.device}")
                            # print(f"type of prediction: {type(prediction)}, device: {prediction.device}")
                            # print(f"type of targets: {type(targets)}, device: {targets.device}")
                            # exit()
                            indices = torch.where(targets == class_name)[0]
                            # print(f"indices: {indices}")
                            # exit()
                            # print(f"output_coll device: {output_coll.device}, output device: {output.device}")
                            output_coll = torch.vstack((output_coll, output[indices]))
                            # print(f"output_coll: {output_coll.size()}")
                            predictions_coll = torch.vstack((predictions_coll, prediction[indices]))
                            # print(f"targets_Coll size: {targets_Coll.size()}, targets size: {targets[indices].unsqueeze(1).size()}")
                            targets_Coll = torch.vstack((targets_Coll, targets[indices].unsqueeze(1)))
                            members_coll = torch.vstack((members_coll, members[indices].unsqueeze(1)))
                            
                            # pickle.dump((output, prediction, members), f)
                        else:
                            print("test data skipping: ",inputs.size()[0])
                        
                        del inputs
                        del targets
                        torch.cuda.empty_cache()
                    
                    # print(f"Class {class_name} information Training")
                    # print(f"output_coll class {class_name}: {output_coll.size()} and size: {output_coll.size()[0]}")
                    # print(f"predictions: {predictions_coll.size()}")
                    # print(f"members: {members_coll.size()}")
                    # print(f"targets: {targets_Coll.size()}")
                    
            
                    # save in class_i file 
                    # attack_train = (output_coll, predictions_coll.squeeze(), members_coll.squeeze(), targets_Coll.squeeze())
                    attack_train = []
                    for i in range(output_coll.size()[0]):
                        attack_train.append((output_coll[i], predictions_coll[i].item(), members_coll[i].item(), targets_Coll[i].item()))
                        
                    # print(f"output_coll: {attack_train[0].shape}, predictions_coll : {attack_train[1].shape}, members_coll: {attack_train[2].shape}, targets_coll : {attack_train[3].shape}")
                    # print(f"attack_test size: {attack_train[0]}, len: {len(attack_train)}")
                    
                    # get track of the dimension of the dataset and append for later use
                    attack_testloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
                    for output_coll, predictions_coll, members_coll, targets_Coll  in attack_testloader:
                        # print(f"len of batch output_coll: {len(output_coll)}, and size: {output_coll.shape}")
                        if output_coll.size()[0] == batch_size:
                            pickle.dump((output_coll, predictions_coll, members_coll, targets_Coll), f)
                        # output, prediction, members = pickle.load(f)
                        else:
                            print(f"skipping the last {output_coll.size()[0]} samples")
                        
                
         
        print(f"Finished Saving {num_classes} Test Dataset")   
        # exit()

    # def train(self, epoch, result_path, result_path_csv):
    #     self.attack_model.train()
    #     self.perturb_model.train()

    #     batch_idx = 1
    #     train_loss = 0
    #     correct = 0
    #     prec = 0
    #     recall = 0
    #     total = 0
    #     bcm = BinaryConfusionMatrix().to(self.device)
    #     final_train_gndtrth = []
    #     final_train_predict = []
    #     final_train_probabe = []

    #     final_result = []

    #     with open(self.ATTACK_SETS + "train.p", "rb") as f:
    #         while(True):
    #             try:
    #                 output, prediction, members, targets = pickle.load(f)
    #                 output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)
    #                 # print(f"1-output: {output.size(), prediction.size(), members.size()}")
    #                 # print(f"1-output output: {type(output),output.dtype, type(prediction), prediction.dtype, members.dtype}")
                    
    #                 # learned_values = self.perturb_model(output, prediction, targets)
    #                 # perturbed_pvs = output + learned_values

    #                 # Mask for non-members
    #                 non_member_mask = (members == 0)
    #                 # print(f"non_member_mask: {non_member_mask}")
    #                 # print(f"output: {output.size()}, prediction: {prediction.size()}, members: {members.size()}, targets: {targets.size()}")
    #                 # print(f"non_member_mask: {non_member_mask.size()}")
    #                 # print(f"non_member_mask: {non_member_mask.sum()}")
    #                 # print(f"members: {members}")
    #                 # # print the non-members samples
    #                 # print(f"output: {output[non_member_mask]}")
    #                 # exit()

    #                 # Apply Perturbation only to Non-Members
    #                 perturbed_pvs = output.clone()
    #                 if non_member_mask.sum() > 0:  # Check if there are non-members in the batch
    #                     learned_values = self.perturb_model(output[non_member_mask], 
    #                                                         prediction[non_member_mask], 
    #                                                         targets[non_member_mask])
    #                     perturbed_pvs[non_member_mask] = output[non_member_mask] + learned_values
                        

    #                 # Normalize and Clip the Perturbed PVs
    #                 perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
    #                 perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)
                    
    #                 results = self.attack_model(perturbed_pvs, prediction, targets)
    #                 # exit()
    #                 # results = F.softmax(results, dim=1)
                    
                    
    #                 self.optimizer.zero_grad()
    #                 self.optimizer_perturb.zero_grad()

    #                 losses = self.criterion(results, members)
                    

    #                 # lambda_div = 0.2
    #                 # lambda_grad = 0.5
    #                 #  # Step 5: Compute Divergence Penalty (KL Divergence)
    #                 # kl_div = F.kl_div(torch.log(perturbed_pvs), output, reduction="batchmean")
    #                 # # Step 6: Total Loss = Attack Loss + KL Divergence Penalty
    #                 # total_loss = losses + lambda_div * kl_div
    #                 # # total_loss = losses
                    
    #                 # # Use a Dynamic Scaling Factor
    #                 # kl_div = F.kl_div(torch.log(perturbed_pvs), output, reduction="batchmean")
    #                 # dynamic_lambda_div = losses.item() / (kl_div.item() + 1e-6)  # Prevent division by zero

    #                 # total_loss = losses + lambda_div * log_kl_div
                    

    #                 lambda_div = 0.2
                   

    #                 kl_div = F.kl_div(torch.log(perturbed_pvs), output, reduction="batchmean")
    #                 log_kl_div = torch.log(1 + kl_div)

    #                 total_loss = losses 
                                        


    #                 # l2_reg = torch.sum(learned_values**2)

    #                 # Step 7: Total Loss = Attack Loss + Regularization Terms
    #                 # total_loss = losses + lambda_div * kl_div + lambda_l2 * l2_reg
                    
        
    #                 total_loss.backward(retain_graph=True)

    #                 self.optimizer.step()
    #                 self.optimizer_perturb.step()
                    
    #                 train_loss += losses.item()
    #                 _, predicted = results.max(1)
    #                 total += members.size(0)
    #                 correct += predicted.eq(members).sum().item()
                    
    #                 # print(f"correctly predicted member and non-members: {predicted.eq(members).sum().item()} out of :{members.size(0)}")
                   
    #                 # print(f"members type: {type(members)}, device: {members.get_device()}, predicted: {type(predicted)}, device: {predicted.get_device()}")
    #                 conf_mat = bcm(predicted, members)
                    
    #                 prec += conf_mat[1,1]/torch.sum(conf_mat[:,-1])    
    #                 recall+=conf_mat[1,1]/torch.sum(conf_mat[-1,:])
    #                 # print(conf_mat)
    #                 # print(f"correct: {torch.sum(torch.diagonal(conf_mat, 0))}")
    #                 # print(f"last col sum: {torch.sum(conf_mat[:,-1])}")
                    
    #                 if epoch:
    #                     final_train_gndtrth.append(members)
    #                     final_train_predict.append(predicted)
    #                     final_train_probabe.append(results[:, 1])

    #                 batch_idx += 1
    #             except EOFError:
    #                 break

    #     if epoch:
    #         final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
    #         final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
    #         final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

    #         train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
    #         train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

    #         # final_result.append(train_f1_score)
    #         # final_result.append(train_roc_auc_score)
            
    #         train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
    #         train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)
            
    #         final_result.append(1.*correct/total)
    #         final_result.append((prec/batch_idx).item())
            
    #         final_result.append((recall/batch_idx).item())
            
    #         final_result.append(train_f1_score)
    #         final_result.append(train_roc_auc_score)

    #         with open(result_path, "wb") as f:
    #             pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
                
    #         with open(result_path_csv, "w") as f:
    #             # Encode the pickled data using Base64
    #             pickled_data = pickle.dumps((final_train_gndtrth, final_train_predict, final_train_probabe))
    #             encoded_data = base64.b64encode(pickled_data)

    #             # Write the encoded data to the CSV file
    #             f.write(encoded_data.decode('utf-8'))
    #             # pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
    #         print("Saved Attack Train Ground Truth and Predict Sets")
    #         print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

    #     # final_result.append(1.*correct/total)
    #     print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f precision: %.3f recall: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx,100*prec/batch_idx,100*recall/batch_idx))
        
    #     # exit()

    #     return final_result

    # # def train(self, epoch, result_path, result_path_csv):
    #     self.attack_model.train()
    #     self.perturb_model.train()

    #     batch_idx = 1
    #     train_loss = 0
    #     correct = 0
    #     prec = 0
    #     recall = 0
    #     total = 0

    #     bcm = BinaryConfusionMatrix().to(self.device)
    #     final_train_gndtrth = []
    #     final_train_predict = []
    #     final_train_probabe = []

    #     final_result = []

    #     with open(self.ATTACK_SETS + "train.p", "rb") as f:
    #         while True:
    #             try:
    #                 output, prediction, members, targets = pickle.load(f)
    #                 output, prediction, members = (
    #                     output.to(self.device),
    #                     prediction.to(self.device),
    #                     members.to(self.device),
    #                 )

    #                 # Mask for non-members
    #                 non_member_mask = (members == 0)

    #                 # Apply Perturbation only to Non-Members
    #                 perturbed_pvs = output.clone()
    #                 if non_member_mask.sum() > 0:  # If non-members exist in the batch
    #                     learned_values = self.perturb_model(
    #                         output[non_member_mask],
    #                         prediction[non_member_mask],
    #                         targets[non_member_mask],
    #                     )
    #                     perturbed_pvs[non_member_mask] = output[non_member_mask] + learned_values

    #                 # Normalize and Clip Perturbed PVs
    #                 perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
    #                 perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

    #                 # Forward Pass through Attack Model
    #                 results = self.attack_model(perturbed_pvs, prediction, targets)

    #                 # Compute Losses
    #                 attack_loss = self.criterion(results, members)

    #                 # KL Divergence Regularization
    #                 kl_div = F.kl_div(torch.log(perturbed_pvs), output, reduction="batchmean")

    #                 # L2 Regularization
    #                 if non_member_mask.sum() > 0:
    #                     l2_reg = torch.sum(learned_values**2)
    #                 else:
    #                     l2_reg = torch.tensor(0.0, device=self.device)

    #                 # Dynamic Weighting for KL Divergence
    #                 lambda_div = 0.2
    #                 lambda_l2 = 0.01
    #                 total_loss = attack_loss + lambda_div * kl_div + lambda_l2 * l2_reg

    #                 # Backpropagation
    #                 self.optimizer.zero_grad()
    #                 self.optimizer_perturb.zero_grad()
    #                 total_loss.backward(retain_graph=True)
    #                 self.optimizer.step()
    #                 self.optimizer_perturb.step()

    #                 # Metrics Calculation
    #                 train_loss += attack_loss.item()
    #                 _, predicted = results.max(1)
    #                 total += members.size(0)
    #                 correct += predicted.eq(members).sum().item()

    #                 conf_mat = bcm(predicted, members)
    #                 prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
    #                 recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

    #                 # Collect predictions for metrics
    #                 if epoch:
    #                     final_train_gndtrth.append(members)
    #                     final_train_predict.append(predicted)
    #                     final_train_probabe.append(results[:, 1])

    #                 batch_idx += 1

    #             except EOFError:
    #                 break

    #     # Post Epoch Evaluation
    #     if epoch:
    #         final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
    #         final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
    #         final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

    #         train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
    #         train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

    #         final_result.extend(
    #             [
    #                 1.0 * correct / total,
    #                 (prec / batch_idx).item(),
    #                 (recall / batch_idx).item(),
    #                 train_f1_score,
    #                 train_roc_auc_score,
    #             ]
    #         )

    #         # Save Results
    #         with open(result_path, "wb") as f:
    #             pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)

    #         with open(result_path_csv, "w") as f:
    #             pickled_data = pickle.dumps((final_train_gndtrth, final_train_predict, final_train_probabe))
    #             encoded_data = base64.b64encode(pickled_data)
    #             f.write(encoded_data.decode("utf-8"))

    #         print("Saved Attack Train Ground Truth and Predict Sets")
    #         print(f"Train F1: {train_f1_score:.6f}\nAUC: {train_roc_auc_score:.6f}")

    #     print(
    #         f"Train Acc: {100. * correct / total:.3f}% ({correct}/{total}) | "
    #         f"Loss: {train_loss / batch_idx:.3f} | Precision: {100. * prec / batch_idx:.3f} | "
    #         f"Recall: {100. * recall / batch_idx:.3f}"
    #     )

    #     return final_result
    # # def train(self, epoch, result_path, result_path_csv):
    #     self.attack_model.train()
    #     self.perturb_model.train()

    #     batch_idx = 1
    #     train_loss = 0
    #     correct = 0
    #     prec = 0
    #     recall = 0
    #     total = 0

    #     bcm = BinaryConfusionMatrix().to(self.device)
    #     final_train_gndtrth = []
    #     final_train_predict = []
    #     final_train_probabe = []

    #     final_result = []

    #     with open(self.ATTACK_SETS + "train.p", "rb") as f:
    #         while True:
    #             try:
    #                 # Load batch data
    #                 output, prediction, members, targets = pickle.load(f)
    #                 output, prediction, members = (
    #                     output.to(self.device),
    #                     prediction.to(self.device),
    #                     members.to(self.device),
    #                 )

    #                 # Identify Boundary Points
    #                 if epoch < 5:  # Early epochs: perturb all non-members
    #                     boundary_mask = (members == 0)
    #                 else:  # Later epochs: use attack model predictions
    #                     with torch.no_grad():
    #                         attack_probs = F.softmax(self.attack_model(output, prediction, targets), dim=1)
    #                     boundary_mask = (attack_probs[:, 1] > 0.4) & (attack_probs[:, 1] < 0.6)

    #                 # Apply Perturbation only to Boundary Points
    #                 perturbed_pvs = output.clone()
    #                 if boundary_mask.sum() > 0:  # If boundary points exist in the batch
    #                     learned_values = self.perturb_model(
    #                         output[boundary_mask],
    #                         prediction[boundary_mask],
    #                         targets[boundary_mask],
    #                     )
    #                     # Scale perturbations to prevent large separations
    #                     alpha = 0.1  # Scaling factor for perturbations
    #                     perturbed_pvs[boundary_mask] = output[boundary_mask] + alpha * learned_values

    #                 # Normalize and Clip Perturbed PVs
    #                 perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
    #                 perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

    #                 # Forward Pass through Attack Model
    #                 results = self.attack_model(perturbed_pvs, prediction, targets)

    #                 # Compute Losses
    #                 attack_loss = self.criterion(results, members)

    #                 # KL Divergence Regularization
    #                 kl_div = F.kl_div(torch.log(perturbed_pvs[boundary_mask]), output[boundary_mask], reduction="batchmean")

    #                 # L2 Regularization
    #                 if boundary_mask.sum() > 0:
    #                     l2_reg = torch.sum(learned_values**2)
    #                 else:
    #                     l2_reg = torch.tensor(0.0, device=self.device)

    #                 # Dynamic Lambda for KL Divergence
    #                 lambda_div = 0.2
    #                 lambda_l2 = 0.01
    #                 total_loss = attack_loss + lambda_div * kl_div + lambda_l2 * l2_reg

    #                 # Backpropagation
    #                 self.optimizer.zero_grad()
    #                 self.optimizer_perturb.zero_grad()
    #                 total_loss.backward(retain_graph=True)
    #                 self.optimizer.step()
    #                 self.optimizer_perturb.step()

    #                 # Metrics Calculation
    #                 train_loss += attack_loss.item()
    #                 _, predicted = results.max(1)
    #                 total += members.size(0)
    #                 correct += predicted.eq(members).sum().item()

    #                 conf_mat = bcm(predicted, members)
    #                 prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
    #                 recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

    #                 # Collect predictions for metrics
    #                 if epoch:
    #                     final_train_gndtrth.append(members)
    #                     final_train_predict.append(predicted)
    #                     final_train_probabe.append(results[:, 1])

    #                 batch_idx += 1

    #             except EOFError:
    #                 break

    #     # Post Epoch Evaluation
    #     if epoch:
    #         final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
    #         final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
    #         final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

    #         train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
    #         train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

    #         final_result.extend(
    #             [
    #                 1.0 * correct / total,
    #                 (prec / batch_idx).item(),
    #                 (recall / batch_idx).item(),
    #                 train_f1_score,
    #                 train_roc_auc_score,
    #             ]
    #         )

    #         # Save Results
    #         with open(result_path, "wb") as f:
    #             pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)

    #         with open(result_path_csv, "w") as f:
    #             pickled_data = pickle.dumps((final_train_gndtrth, final_train_predict, final_train_probabe))
    #             encoded_data = base64.b64encode(pickled_data)
    #             f.write(encoded_data.decode("utf-8"))

    #         print("Saved Attack Train Ground Truth and Predict Sets")
    #         print(f"Train F1: {train_f1_score:.6f}\nAUC: {train_roc_auc_score:.6f}")

    #     print(
    #         f"Train Acc: {100. * correct / total:.3f}% ({correct}/{total}) | "
    #         f"Loss: {train_loss / batch_idx:.3f} | Precision: {100. * prec / batch_idx:.3f} | "
    #         f"Recall: {100. * recall / batch_idx:.3f}"
    #     )

    #     return final_result
    # # def train(self, epoch, result_path, result_path_csv):
    #     self.attack_model.train()
    #     self.perturb_model.train()

    #     batch_idx = 1
    #     train_loss = 0
    #     correct = 0
    #     prec = 0
    #     recall = 0
    #     total = 0

    #     bcm = BinaryConfusionMatrix().to(self.device)
    #     final_train_gndtrth = []
    #     final_train_predict = []
    #     final_train_probabe = []

    #     final_result = []

    #     # for output, prediction, members, targets in dataloader:
    #     with open(self.ATTACK_SETS + "train.p", "rb") as f:
    #         while True:
    #             try:
    #                 output, prediction, members, targets = pickle.load(f)
    #             except EOFError:
    #                 break

    #             output = output.to(self.device)
    #             prediction = prediction.to(self.device)
    #             members = members.to(self.device)
    #             # targets can remain on CPU or be moved if needed

    #             # Create masks for members and non-members.
    #             member_mask = (members == 1)
    #             non_member_mask = (members == 0)

    #             if member_mask.sum() > 0 and non_member_mask.sum() > 0:
    #                 # Get indices (in the original batch) for members and non-members
    #                 member_indices = member_mask.nonzero(as_tuple=True)[0]  # shape: (n_members,)
    #                 non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]  # shape: (n_non_members,)

    #                 # Extract their corresponding representations (PVs) from output.
    #                 member_pvs = output[member_indices]       # shape: (n_members, C)
    #                 non_member_pvs = output[non_member_indices] # shape: (n_non_members, C)

    #                 # ----- Step 2: Overlap Detection via Cosine Similarity -----
    #                 cos_sim = F.cosine_similarity(
    #                     non_member_pvs.unsqueeze(1),  # (n_non_members, 1, C)
    #                     member_pvs.unsqueeze(0),      # (1, n_members, C)
    #                     dim=2
    #                 )
    #                 # For each non-member, take the maximum cosine similarity with any member.
    #                 max_cos_sim, _ = cos_sim.max(dim=1)  # shape: (n_non_members,)

    #                 # cosine_threshold = 
    #                 alpha = 1
    #                 # Create a boolean mask over non-members for those that exceed the threshold.
    #                 overlap_mask = max_cos_sim > self.cosine_threshold  # shape: (n_non_members,)

    #                 # ----- Step 3: Among Overlapping Non-members, Select the One with Highest Entropy -----
    #                 if overlap_mask.sum() > 0:
    #                     # Get the indices (relative to the original batch) for overlapping non-members.
    #                     overlapping_non_member_indices = non_member_indices[overlap_mask]

    #                     # Compute the entropy for each sample.
    #                     overlapping_outputs = output[overlapping_non_member_indices]  # shape: (n_overlap, C)
    #                     entropy = -(overlapping_outputs * torch.log(overlapping_outputs + 1e-10)).sum(dim=1)  # shape: (n_overlap,)

    #                     median_entropy = torch.quantile(entropy, self.Entropy_quantile_threshold)

    #                     # Create a mask selecting only the overlapping non-members with entropy above the median.
    #                     entropy_mask = entropy > median_entropy  # Boolean mask of shape (n_overlap,)

    #                     # Get the final indices (in the original batch) to be perturbed.
    #                     selected_index = overlapping_non_member_indices[entropy_mask]

    #                     # ----- Step 4: Perturb Only the Selected Sample -----
    #                     learned_values = self.perturb_model(
    #                         output[selected_index],
    #                         targets[selected_index]
    #                     )
    #                     perturbed_pvs = output.clone()
    #                     perturbed_pvs[selected_index] = output[selected_index] + alpha * learned_values
    #                 else:
    #                     # No overlapping non-members detected above the cosine threshold.
    #                     perturbed_pvs = output.clone()
    #             else:
    #                 perturbed_pvs = output.clone()

    #             # Normalize and Clip Perturbed PVs
    #             perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
    #             perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

    #             # Step 4: Forward Pass through Attack Model
    #             results = self.attack_model(perturbed_pvs, prediction, targets)

    #             # Step 5: Compute Losses
    #             attack_loss = self.criterion(results, members)

    #             # Dynamic Lambda for KL Divergence (if used)
    #             lambda_div = 0.9
    #             lambda_l2 = 0.5
    #             # total_loss = attack_loss + lambda_div * kl_div + lambda_l2 * l2_reg
    #             total_loss = attack_loss

    #             # Step 6: Backpropagation
    #             self.optimizer.zero_grad()
    #             self.optimizer_perturb.zero_grad()
    #             self.optimizer_cosine.zero_grad()
    #             self.optimizer_quantile_threshold.zero_grad()





                 
                
    #             total_loss.backward(retain_graph=True)
    #             self.optimizer.step()
    #             self.optimizer_perturb.step()
                
    #             self.optimizer_quantile_threshold.step()

    #             # Check and store the gradient of cosine_threshold
    #             if self.cosine_threshold.grad is None:
    #                 print("No gradient computed for cosine_threshold!")
    #                 cosine_grad = None
    #             else:
    #                 cosine_grad = self.cosine_threshold.grad.clone().detach()
    #                 print("Cosine Threshold Gradient:", cosine_grad)
               
    #             print(f"before Cosine Threshold: {self.cosine_threshold.item()}")
    #             self.optimizer_cosine.step()
    #             print(f"after Cosine Threshold: {self.cosine_threshold.item()}")
    #             exit()
    #             # Step 7: Metrics Calculation
    #             train_loss += attack_loss.item()
    #             _, predicted = results.max(1)
    #             total += members.size(0)
    #             correct += predicted.eq(members).sum().item()

    #             conf_mat = bcm(predicted, members)
    #             prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
    #             recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

    #             # Collect predictions for metrics
    #             if epoch:
    #                 final_train_gndtrth.append(members)
    #                 final_train_predict.append(predicted)
    #                 final_train_probabe.append(results[:, 1])

    #             batch_idx += 1

    #     # Post Epoch Evaluation
    #     if epoch:
    #         final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
    #         final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
    #         final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

    #         train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
    #         train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

    #         final_result.extend([
    #             1.0 * correct / total,
    #             (prec / batch_idx).item(),
    #             (recall / batch_idx).item(),
    #             train_f1_score,
    #             train_roc_auc_score,
    #         ])

    #         # Save Results
    #         with open(result_path, "wb") as f_out:
    #             pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f_out)

    #         with open(result_path_csv, "w") as f_out:
    #             pickled_data = pickle.dumps((final_train_gndtrth, final_train_predict, final_train_probabe))
    #             encoded_data = base64.b64encode(pickled_data)
    #             f_out.write(encoded_data.decode("utf-8"))

    #         print("Saved Attack Train Ground Truth and Predict Sets")
    #         print(f"Train F1: {train_f1_score:.6f}\nAUC: {train_roc_auc_score:.6f}")

    #     print('Train Acc: %.3f%% (%d/%d) | Loss: %.3f precision: %.3f recall: %.3f' %
    #         (100. * correct / total, correct, total, 1. * train_loss / batch_idx, 100 * prec / batch_idx, 100 * recall / batch_idx))
    #     print(f"Cosine Threshold: {self.cosine_threshold.item():.4f}, quantile Threshold: {self.Entropy_quantile_threshold.item():.4f}")
    
    
    
    # def train(self, epoch, result_path, result_path_csv, mode):
    #         self.attack_model.train()
    #         self.perturb_model.train()

    #         batch_idx = 1
    #         train_loss = 0
    #         correct = 0
    #         prec = 0
    #         recall = 0
    #         total = 0

    #         bcm = BinaryConfusionMatrix().to(self.device)
    #         final_train_gndtrth = []
    #         final_train_predict = []
    #         final_train_probabe = []

    #         final_result = []

    #         # for output, prediction, members, targets in dataloader:
    #         with open(self.ATTACK_SETS + "train.p", "rb") as f:
    #             while True:
    #                 try:
    #                     output, prediction, members, targets = pickle.load(f)
    #                 except EOFError:
    #                     break

    #                 output = output.to(self.device)
    #                 prediction = prediction.to(self.device)
    #                 members = members.to(self.device)
    #                 # targets can remain on CPU or be moved if needed
    #                 if self.attack_name == "apcmia": #new
    #                     # Create masks for members and non-members.
    #                     member_mask = (members == 1)
    #                     non_member_mask = (members == 0)

    #                     if member_mask.sum() > 0 and non_member_mask.sum() > 0:
    #                         # Get indices for members and non-members.
    #                         member_indices = member_mask.nonzero(as_tuple=True)[0]  # (n_members,)
    #                         non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]  # (n_non_members,)

    #                         # Extract PVs for members and non-members.
    #                         member_pvs = output[member_indices]       # (n_members, C)
    #                         non_member_pvs = output[non_member_indices] # (n_non_members, C)

    #                         # ----- Step 2: Overlap Detection via Cosine Similarity -----
    #                         # Compute cosine similarity between each non-member and each member.
    #                         cos_sim = F.cosine_similarity(
    #                             non_member_pvs.unsqueeze(1),  # (n_non_members, 1, C)
    #                             member_pvs.unsqueeze(0),      # (1, n_members, C)
    #                             dim=2
    #                         )
    #                         # For each non-member, take the maximum cosine similarity with any member.
    #                         max_cos_sim, _ = cos_sim.max(dim=1)  # (n_non_members,)
    #                         # self.k1 = temperature = 1000.0  # scaling factor for the logits
    #                         # Instead of hard thresholding, compute a differentiable soft weight.
    #                         # Use a sigmoid to convert the difference (max_cos_sim - threshold) into a weight in [0, 1].
    #                         # temperature = 10000.0  # Controls sharpness of the sigmoid.
    #                         # soft_overlap = torch.sigmoid((max_cos_sim - self.cosine_threshold) * temperature)
    #                         # Now soft_overlap is near 0 if max_cos_sim is well below self.cosine_threshold,
    #                         # and near 1 if max_cos_sim is above it.
    #                         # here
    #                         tau = 0.5            # Gumbel–Softmax temperature (lower tau makes the distribution sharper)
    #                         cosine_threshold = torch.sigmoid(self.cosine_threshold)
    #                         logits = (max_cos_sim - cosine_threshold) * self.k1  # shape: (n_non_members,)
    #                         binary_logits = torch.stack([-logits, logits], dim=1)  # shape: (n_non_members, 2
    #                         gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True) 
    #                         binary_selection = gumbel_selection[:, 1]
                            
    #                         # print(binary_selection.unsqueeze(1))
    #                         # exit()
    #                         alpha = 1

    #                         # ----- Step 3: Incorporate Entropy (Optional) -----
    #                         # Compute the entropy for each non-member.
    #                         # (Higher entropy can indicate less confidence; you might want to perturb those more.)
    #                         # entropy = -(non_member_pvs * torch.log(non_member_pvs + 1e-10)).sum(dim=1)  # (n_non_members,)
    #                         # To “softly” favor samples with higher entropy, apply softmax over entropy.
    #                         # (If you prefer to rely solely on cosine similarity, you can omit this step.)
    #                         # entropy_weight = F.softmax(entropy, dim=0)  # (n_non_members,)
    #                         # Combine the two signals. Here we multiply them, but you could also consider a weighted sum.
    #                         # selection_weights = soft_overlap # (n_non_members,)
    #                         # Optionally, you can normalize these weights so they lie in [0,1]:
    #                         # selection_weights = selection_weights / (selection_weights.max() + 1e-10)

    #                         # ----- Step 4: Perturbation with Soft–Selection -----
    #                         # Compute the learned perturbations for all non-member PVs.
    #                         # Note: self.perturb_model should be differentiable.
    #                         learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
    #                         # print(f"Learned Values: {learned_values}")
    #                         # print(binary_selection.unsqueeze(1) * learned_values)
    #                         # exit()
    #                         # Instead of selecting a subset, perturb all non-members weighted by selection_weights.
    #                         # alpha = 0.5
    #                         # perturbed_non_member_pvs = non_member_pvs + alpha * selection_weights.unsqueeze(1) * learned_values
    #                         # perturbed_non_member_pvs = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values
                            
    #                         # First, compute a tentative perturbed output using the binary selection.
    #                         tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

    #                         # Compute the entropy of each tentative perturbed non-member PV.
    #                         # (Entropy is computed over the probability distribution; add a small epsilon for numerical stability.)
    #                         epsilon = 1e-10
    #                         entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1).to(self.device)  # (n_non_members,)

    #                         # Select top samples based on entropy.
    #                         # For example, use a quantile threshold (self.Entropy_quantile_threshold should be a float in (0,1)).
    #                         # quantile_val = torch.quantile(entropy, 0.25)
    #                         quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
    #                         quantile_val = torch.quantile(entropy, quantile_threshold)


    #                         # self.k = 100000.0  # A scaling factor to control the steepness of the transition.
    #                         entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)

    #                         # entropy_mask = (entropy >= quantile_val).float()  # 1 if entropy is high, 0 otherwise

    #                         final_selection = binary_selection * entropy_mask
    #                         perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            
                            
    #                         # Replace the non-member PVs in the overall output with their perturbed versions.
    #                         perturbed_pvs = output.clone()
    #                         perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
    #                     else:
    #                         perturbed_pvs = output.clone()

    #                     # Normalize and clip the perturbed PVs.
    #                     perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
    #                     perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

    #                     # ----- Step 5: Forward Pass through the Attack Model -----
    #                     results = self.attack_model(perturbed_pvs, prediction, targets)
    #                 else:
    #                     results = self.attack_model(output, prediction, targets)
    #                 # ----- Step 6: Compute Loss -----
    #                 attack_loss = self.criterion(results, members)
    #                 lambda_div = 0.9
    #                 lambda_l2 = 0.5
    #                 # total_loss = attack_loss  # + lambda_div * kl_div + lambda_l2 * l2_reg (if used)
                    
    #                 # # Regularize the output of the model:
    #                 # # Here we add an L2 penalty to encourage the perturbed output to stay close to the original output.
    #                 # lambda_output_reg = 0.8  # adjust this value as needed

    #                 # # Create a target distribution (uniform) for each output.
    #                 # # Assume there are C classes.
    #                 # C = perturbed_pvs.size(1)
    #                 # target_dist = torch.full_like(perturbed_pvs, 1.0 / C)

    #                 # # Compute KL divergence for the outputs.
    #                 # # Note: We take the log of perturbed_pvs for KL; make sure perturbed_pvs is > 0.
    #                 # kl_div = F.kl_div(perturbed_pvs.log(), target_dist, reduction='batchmean')

    #                 total_loss = attack_loss
                        

    #                 # ----- Step 7: Backpropagation -----
    #                 self.optimizer.zero_grad()
    #                 self.optimizer_perturb.zero_grad()
    #                 self.optimizer_cosine.zero_grad()
    #                 self.optimizer_quantile_threshold.zero_grad()

    #                 total_loss.backward(retain_graph=True)
    #                 self.optimizer.step()
    #                 self.optimizer_perturb.step()
    #                 self.optimizer_quantile_threshold.step()
                    
                    
                

    #                 # print(f"before Cosine Threshold: {self.cosine_threshold.item()}")
    #                 self.optimizer_cosine.step()
    #                 # print(f"after Cosine Threshold: {self.cosine_threshold.item()}")
    #                 # exit()   # Remove this when not debugging.

    #                 # ----- Step 8: Metrics Calculation -----
    #                 train_loss += attack_loss.item()
    #                 _, predicted = results.max(1)
    #                 total += members.size(0)
    #                 correct += predicted.eq(members).sum().item()

    #                 conf_mat = bcm(predicted, members)
    #                 prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
    #                 recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

    #                 # Collect predictions for metrics.
    #                 if epoch:
    #                     final_train_gndtrth.append(members)
    #                     final_train_predict.append(predicted)
    #                     final_train_probabe.append(results[:, 1])

    #                 batch_idx += 1

    #         # Post Epoch Evaluation
    #         if epoch:
    #             final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
    #             final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
    #             final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

    #             train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
    #             train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

    #             final_result.extend([
    #                 100. * correct / total,
    #                 (prec / batch_idx).item(),
    #                 (recall / batch_idx).item(),
    #                 train_f1_score,
    #                 train_roc_auc_score,
    #             ])

    #             # Save Results
    #             with open(result_path, "wb") as f_out:
    #                 pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f_out)

    #             with open(result_path_csv, "w") as f_out:
    #                 pickled_data = pickle.dumps((final_train_gndtrth, final_train_predict, final_train_probabe))
    #                 encoded_data = base64.b64encode(pickled_data)
    #                 f_out.write(encoded_data.decode("utf-8"))

    #             print("Saved Attack Train Ground Truth and Predict Sets")
    #             print(f"Train F1: {train_f1_score:.6f}\nAUC: {train_roc_auc_score:.6f}")

    #         train_loss = train_loss/batch_idx
    #         # print('Train Acc: %.3f%% (%d/%d) | Loss: %.3f precision: %.3f recall: %.3f' %
    #             # (100. * correct / total, correct, total, train_loss / batch_idx, 100 * prec / batch_idx, 100 * recall / batch_idx))
    #         print(f"Train Acc: {100.*correct/(1.0*total):.3f}% ({correct}/{total}) | Loss: {train_loss:.3f} precision: {100.*prec/(1.0*batch_idx):.3f} recall: {100.*recall/batch_idx:.3f}")
    #         if self.attack_name != "apcmia":
    #             cosine_threshold = 0
    #             quantile_threshold = 0
    #         print(f"Cosine Threshold: {cosine_threshold:.4f}, quantile Threshold: {quantile_threshold:.4f}")
    #         return cosine_threshold, quantile_threshold

    def contrastive_loss(self, embeddings, labels, margin):
        """
        A simple contrastive loss function.
        For a pair of embeddings, if they belong to the same class (both member or both non-member),
        the target is 1 (and we penalize a large distance);
        if they belong to different classes, the target is -1 (and we penalize similarity if too high).
        
        This implementation computes the loss over all pairs in the batch.
        
        Args:
            embeddings: Tensor of shape (N, D) where N is batch size and D is embedding dimension.
            labels: Binary labels of shape (N,) indicating membership (1 for member, 0 for non-member).
            margin: Margin for dissimilar pairs.
        
        Returns:
            A scalar contrastive loss value.
        """
        # Normalize embeddings to unit vectors
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise cosine similarity matrix
        cosine_sim = torch.matmul(embeddings, embeddings.t())  # shape: (N, N)
        
        # Create label matrix: 1 if same class, -1 if different
        labels = labels.unsqueeze(1)  # shape (N, 1)
        label_matrix = (labels == labels.t()).float()
        # Convert same/different to targets of 1 and -1
        target = label_matrix * 2 - 1  # 1 for same, -1 for different
        
        # For same pairs, we want cosine similarity to be close to 1,
        # for different pairs, we want similarity to be at most a margin.
        # One simple loss is mean squared error from target:
        loss_same = F.mse_loss(cosine_sim * (target==1).float(), torch.ones_like(cosine_sim) * (target==1).float())
        # For different pairs, we want the cosine similarity to be less than some margin.
        # If similarity > margin, we incur a loss.
        diff_sim = cosine_sim * (target==-1).float()
        loss_diff = F.relu(diff_sim - margin).pow(2).mean()
        
        return loss_same + loss_diff

    # -------------------------
    # Modified Training Function
    # -------------------------
    def train(self, epoch, result_path, result_path_csv, mode):
        self.attack_model.train()
        self.perturb_model.train()

        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []
        final_result = []

        lambda_contrast = 0.5  # Weight for the contrastive loss term; adjust as necessary

        # Open your training data file
        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break

                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)
                # targets can remain on CPU or be moved if needed

                if self.attack_name == "apcmia":
                    # -- Your code for perturbation using cosine similarity, entropy etc. --
                    # This block remains as before.
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]

                        # Overlap detection via cosine similarity:
                        cos_sim = F.cosine_similarity(
                            non_member_pvs.unsqueeze(1),
                            member_pvs.unsqueeze(0),
                            dim=2
                        )
                        max_cos_sim, _ = cos_sim.max(dim=1)
                        tau = 0.5
                        cosine_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_cos_sim - cosine_threshold) * self.k1
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]

                        alpha = 1

                        # Entropy step:
                        epsilon = 1e-10
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * \
                                                self.perturb_model(non_member_pvs, targets[non_member_indices])
                        entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * \
                                                    self.perturb_model(non_member_pvs, targets[non_member_indices])
                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    # Forward pass through the attack model:
                    results = self.attack_model(perturbed_pvs, prediction, targets) #with perturbation
                    # results = self.attack_model(output, prediction, targets) # without perturbation

                    # # NOTE: the constastive loss part is commented out and also the from the total loss., 
                    # # # --- Extract embeddings for contrastive loss ---
                    # # # Here, we assume your attack model has a method get_embeddings() that returns
                    # # # a feature embedding for each sample.
                    margin=0.5
                    # # embeddings = self.attack_model.get_embeddings(output, prediction, targets) # without perturbation
                    embeddings = self.attack_model.get_embeddings(perturbed_pvs, prediction, targets) # with perturbation

                    # # # Contrastive loss using the embeddings.
                    contrast_loss = self.contrastive_loss(embeddings, members, margin)

                 
                else:
                    results = self.attack_model(output, prediction, targets)
                    contrast_loss = 0  # no contrastive loss for non-apcmia attacks

                # Compute primary loss (attack loss)
                attack_loss = self.criterion(results, members)
                # total_loss = attack_loss #  without Constrastive loss 
                total_loss = attack_loss + lambda_contrast * contrast_loss #  with Contrastive loss


                # total_loss = attack_loss

                # Backpropagation
                self.optimizer.zero_grad()
                self.optimizer_perturb.zero_grad()
                self.optimizer_cosine.zero_grad()
                self.optimizer_quantile_threshold.zero_grad()

                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer_perturb.step()
                self.optimizer_quantile_threshold.step()
                self.optimizer_cosine.step()

                train_loss += attack_loss.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_train_gndtrth.append(members)
                final_train_predict.append(predicted)
                final_train_probabe.append(results[:, 1])

                batch_idx += 1

        # Post Epoch Evaluation
        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

        train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
        train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

        final_result = [
            100. * correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            train_f1_score,
            train_roc_auc_score,
        ]

        # (Saving results code omitted for brevity)
        print(f"Train Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Loss: {train_loss/batch_idx:.3f} "
            f"Precision: {100.*prec/batch_idx:.3f} Recall: {100.*recall/batch_idx:.3f}")
        
        ent_thre = torch.sigmoid(self.Entropy_quantile_threshold)
        cos_thre = torch.sigmoid(self.cosine_threshold)

        if self.attack_name == "apcmia":
            print(f"CosineT: {cos_thre:.4f}, Quantile Threshold: {ent_thre:.4f}")
     
        return cos_thre, ent_thre

    def test(self, epoch_flag, result_path, mode):
        self.attack_model.eval()
        self.perturb_model.eval()

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    # Move tensors to device.
                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)
                    # targets can remain on CPU or be moved if needed.
                    if self.attack_name == "apcmia": #new

                        # Create masks for members and non-members.
                        member_mask = (members == 1)
                        non_member_mask = (members == 0)

                        if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                            # Get indices for members and non-members.
                            member_indices = member_mask.nonzero(as_tuple=True)[0]
                            non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                            # Extract PVs.
                            member_pvs = output[member_indices]       # (n_members, C)
                            non_member_pvs = output[non_member_indices] # (n_non_members, C)

                            # ----- Step 2: Overlap Detection via Cosine Similarity -----
                            cos_sim = F.cosine_similarity(
                                non_member_pvs.unsqueeze(1),  # (n_non_members, 1, C)
                                member_pvs.unsqueeze(0),      # (1, n_members, C)
                                dim=2
                            )
                            max_cos_sim, _ = cos_sim.max(dim=1)  # (n_non_members,)

                            # ----- Step 3: Differentiable Binary Selection using Gumbel–Softmax -----
                            temperature = self.k1  # scaling factor for logits
                            tau = 0.5           # Gumbel–Softmax temperature
                            # Reparameterize the cosine threshold to (0,1)
                            cosine_threshold = torch.sigmoid(self.cosine_threshold)
                            logits = (max_cos_sim - cosine_threshold) * temperature  # (n_non_members,)
                            binary_logits = torch.stack([-logits, logits], dim=1)  # (n_non_members, 2)
                            gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                            binary_selection = gumbel_selection[:, 1]  # (n_non_members,)

                            alpha = 1.0

                            # ----- Step 4: Perturbation with Entropy Filtering -----
                            learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                            # Compute tentative perturbed outputs.
                            tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                            # Compute entropy for each tentative perturbed PV.
                            epsilon = 1e-10
                            entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1).to(self.device)  # (n_non_members,)

                            # Select top samples based on entropy.
                            # For example, use a quantile threshold (self.Entropy_quantile_threshold should be a float in (0,1)).
                            # quantile_val = torch.quantile(entropy, 0.25)
                            quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                            quantile_val = torch.quantile(entropy, quantile_threshold)
                            # Use a sigmoid to create a soft entropy mask.
                            # self.k controls the steepness (if not defined, default to 50.0).
                            # k = self.k if hasattr(self, "k") else 50.0
                            entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)

                            # Combine the binary selection with the entropy mask.
                            final_selection = binary_selection * entropy_mask
                            perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            # Replace non-member PVs in the overall output.
                            perturbed_pvs = output.clone()
                            perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                        else:
                            perturbed_pvs = output.clone()

                        # ----- Step 5: Normalize and Clip Perturbed PVs -----
                        perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                        perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                        # ----- Step 6: Forward Pass through the Attack Model -----
                        results = self.attack_model(perturbed_pvs, prediction, targets)
                    else:
                        results = self.attack_model(output, prediction, targets)

                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        # ----- Post Evaluation -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        if epoch_flag:
                fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)
        else:
            fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)

        
        
        # roc_auc = auc(fpr, tpr)
        avg_test_loss = total_test_loss / batch_idx
        
        # final_result.append(100. * correct / total)
        # final_result.append((prec / batch_idx).item())
        # final_result.append((recall / batch_idx).item())
        # final_result.append(test_f1_score)
        # final_result.append(test_roc_auc_score)
        # final_result.append(avg_test_loss)


        final_result.extend([
        correct / total,
        (prec / batch_idx).item(),
        (recall / batch_idx).item(),
        test_f1_score,
        test_roc_auc_score,
        avg_test_loss  # Append average test loss
        ])

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        # print("Saved Attack Test Ground Truth and Predict Sets")
        # print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))
        # print('Test Acc: %.3f%% (%d/%d), Precision: %.3f, Recall: %.3f' %
        #     (100. * correct / total, correct, total, 100. * prec / batch_idx, 100 * recall / batch_idx))
        # print( 'Test Acc: %.3f%% (%d/%d), Loss: %.3f%% , precision: %.3f, recall: %.3f' % (100.*correct/(1.0*total), correct, total, 100.*prec/(1.0*batch_idx),100*recall/batch_idx))
        print(f"Test Acc: {100.*correct/(1.0*total):.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, precision: {100.*prec/(1.0*batch_idx):.3f}, recall: {100.*recall/batch_idx:.3f}")


        # self.early_stopping(avg_test_loss, self.attack_model)
        
        # if self.early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        
        # # herererer
        # # load the last checkpoint with the best model
        # model.load_state_dict(torch.load('checkpoint.pt', weights_only=True))
        
        # here unload the full test.p and 
        return final_result, fpr, tpr
    
    def train_KL(self, epoch, result_path, result_path_csv, mode):
        self.attack_model.train()
        self.perturb_model.train()

        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []
        final_result = []

        lambda_contrast = 0.5  # Weight for the contrastive loss term

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break

                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)

                if self.attack_name == "apcmia":
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]

                        # KL divergence similarity
                        epsilon = 1e-8
                        p = non_member_pvs.unsqueeze(1).clamp(min=epsilon, max=1)
                        q = member_pvs.unsqueeze(0).clamp(min=epsilon, max=1)

                        kl_div = (p * (p / q).log()).sum(dim=2)
                        min_kl_div, _ = kl_div.min(dim=1)

                        tau = 0.5
                        kl_threshold = torch.sigmoid(self.kl_threshold)
                        logits = (kl_threshold - min_kl_div) * self.k1
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]

                        alpha = 1
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * \
                                            self.perturb_model(non_member_pvs, targets[non_member_indices])
                        entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        final_selection = binary_selection * entropy_mask

                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * \
                                                self.perturb_model(non_member_pvs, targets[non_member_indices])

                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    results = self.attack_model(perturbed_pvs, prediction, targets)

                    margin = 0.5
                    embeddings = self.attack_model.get_embeddings(perturbed_pvs, prediction, targets)
                    contrast_loss = self.contrastive_loss(embeddings, members, margin)

                else:
                    results = self.attack_model(output, prediction, targets)
                    contrast_loss = 0

                attack_loss = self.criterion(results, members)
                total_loss = attack_loss + lambda_contrast * contrast_loss

                self.optimizer.zero_grad()
                self.optimizer_perturb.zero_grad()
                self.optimizer_kl.zero_grad()
                self.optimizer_quantile_threshold.zero_grad()

                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer_perturb.step()
                self.optimizer_quantile_threshold.step()
                self.optimizer_kl.step()

                train_loss += attack_loss.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_train_gndtrth.append(members)
                final_train_predict.append(predicted)
                final_train_probabe.append(results[:, 1])

                batch_idx += 1

        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

        train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
        train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

        final_result = [
            100. * correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            train_f1_score,
            train_roc_auc_score,
        ]

        print(f"Train Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Loss: {train_loss/batch_idx:.3f} "
            f"Precision: {100.*prec/batch_idx:.3f} Recall: {100.*recall/batch_idx:.3f}")
        print(f"KL Threshold: {self.kl_threshold.item():.4f}, Quantile Threshold: {self.Entropy_quantile_threshold.item():.4f}")

        return self.kl_threshold.item(), self.Entropy_quantile_threshold.item()
    

    def train_ecld(self, epoch, result_path, result_path_csv, mode): 
        
        self.attack_model.train()
        self.perturb_model.train()

        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []
        final_result = []

        lambda_contrast = 0.5  # Weight for the contrastive loss term; adjust as necessary

        # Open your training data file
        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break

                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)
                # targets can remain on CPU or be moved if needed

                if self.attack_name == "apcmia":
                    # --- Perturbation using negative Euclidean distance instead of cosine similarity ---
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]       # (n_members, C)
                        non_member_pvs = output[non_member_indices] # (n_non_members, C)

                        # Compute pairwise Euclidean distances: shape (n_non_members, n_members)
                        distances = torch.cdist(non_member_pvs, member_pvs, p=2)
                        # Convert distances to similarity scores (smaller distance -> higher similarity)
                        euclid_sim = -distances
                        # For each non-member, select maximum similarity (i.e. the most similar member)
                        max_sim, _ = euclid_sim.max(dim=1)  # (n_non_members,)

                        tau = 0.5
                        # Use a learnable threshold (same parameter reused; can be renamed)
                        similarity_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_sim - similarity_threshold) * self.k1
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]
                        
                        alpha = 1

                        # --- Entropy-Based Weighting ---
                        epsilon = 1e-10
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * \
                                                self.perturb_model(non_member_pvs, targets[non_member_indices])
                        entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * \
                                                    self.perturb_model(non_member_pvs, targets[non_member_indices])
                        
                        # Replace non-member PVs in output with perturbed values.
                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    # Normalize the perturbed probability vectors:
                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    # Forward pass through the attack model:
                    results = self.attack_model(perturbed_pvs, prediction, targets) # Using original output (or you can try perturbed_pvs)

                    # --- Contrastive Loss ---
                    margin = 0.5
                    embeddings = self.attack_model.get_embeddings(perturbed_pvs, prediction, targets)
                    contrast_loss = self.contrastive_loss(embeddings, members, margin)
                else:
                    results = self.attack_model(output, prediction, targets)
                    contrast_loss = 0  # no contrastive loss for non-apcmia attacks

                # Compute primary loss (attack loss)
                attack_loss = self.criterion(results, members)
                total_loss = attack_loss + lambda_contrast * contrast_loss 

                # Backpropagation
                self.optimizer.zero_grad()
                self.optimizer_perturb.zero_grad()
                self.optimizer_cosine.zero_grad()
                self.optimizer_quantile_threshold.zero_grad()

                total_loss.backward(retain_graph=True)

                # print(f"Grad of cosine_threshold: {self.cosine_threshold.grad}")

                self.optimizer.step()
                self.optimizer_perturb.step()
                self.optimizer_quantile_threshold.step()
                self.optimizer_cosine.step()

                # print(f"Grad of cosine_threshold: {self.cosine_threshold.grad}")

                train_loss += attack_loss.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_train_gndtrth.append(members)
                final_train_predict.append(predicted)
                final_train_probabe.append(results[:, 1])

                batch_idx += 1

        # Post Epoch Evaluation
        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

        train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
        train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

        final_result = [
            100. * correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            train_f1_score,
            train_roc_auc_score,
        ]

        print(f"Train Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Loss: {train_loss/batch_idx:.3f} "
            f"Precision: {100.*prec/batch_idx:.3f} Recall: {100.*recall/batch_idx:.3f}")
        print(f"Cosine Threshold: {torch.sigmoid(self.cosine_threshold).item():.4f}, Quantile Threshold: {torch.sigmoid(self.Entropy_quantile_threshold).item():.4f}")
        # print(f"Cosine Threshold (sigmoid): {torch.sigmoid(self.cosine_threshold).item():.4f}")
        # print(f"Quantile Threshold (sigmoid): {torch.sigmoid(self.Entropy_quantile_threshold).item():.4f}")

        return self.cosine_threshold.item(), self.Entropy_quantile_threshold.item()


  
    
    
    def test_KL(self, epoch_flag, result_path, mode):
        self.attack_model.eval()
        self.perturb_model.eval()

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)

                    if self.attack_name == "apcmia":
                        member_mask = (members == 1)
                        non_member_mask = (members == 0)

                        if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                            member_indices = member_mask.nonzero(as_tuple=True)[0]
                            non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                            member_pvs = output[member_indices]
                            non_member_pvs = output[non_member_indices]

                            # ----- KL Divergence Similarity -----
                            epsilon = 1e-8
                            p = non_member_pvs.unsqueeze(1).clamp(min=epsilon, max=1)
                            q = member_pvs.unsqueeze(0).clamp(min=epsilon, max=1)

                            kl_div = (p * (p / q).log()).sum(dim=2)  # shape: [n_non_members, n_members]
                            min_kl_div, _ = kl_div.min(dim=1)

                            # ----- Gumbel-Softmax Selection -----
                            tau = 0.5
                            kl_threshold = torch.sigmoid(self.kl_threshold)
                            logits = (kl_threshold - min_kl_div) * self.k1
                            binary_logits = torch.stack([-logits, logits], dim=1)
                            gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                            binary_selection = gumbel_selection[:, 1]

                            # ----- Entropy Filtering & Perturbation -----
                            alpha = 1.0
                            learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                            tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                            entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1)
                            quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                            quantile_val = torch.quantile(entropy, quantile_threshold)
                            entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                            final_selection = binary_selection * entropy_mask

                            perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            perturbed_pvs = output.clone()
                            perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                        else:
                            perturbed_pvs = output.clone()

                        # ----- Normalize & Clip -----
                        perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                        perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                        results = self.attack_model(perturbed_pvs, prediction, targets)
                    else:
                        results = self.attack_model(output, prediction, targets)

                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        # ----- Post-Evaluation -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx
        final_result.extend([
            correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss
        ])

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        print(f"Test Acc: {100.*correct/(1.0*total):.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, "
            f"precision: {100.*prec/(1.0*batch_idx):.3f}, recall: {100.*recall/batch_idx:.3f}")

        return final_result, fpr, tpr

    def test_ecld(self, epoch_flag, result_path, mode):
        self.attack_model.eval()
        self.perturb_model.eval()

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)

                    if self.attack_name == "apcmia":
                        member_mask = (members == 1)
                        non_member_mask = (members == 0)

                        if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                            member_indices = member_mask.nonzero(as_tuple=True)[0]
                            non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                            member_pvs = output[member_indices]
                            non_member_pvs = output[non_member_indices]

                            # ----- Euclidean Similarity -----
                            distances = torch.cdist(non_member_pvs, member_pvs, p=2)
                            euclid_sim = -distances
                            max_sim, _ = euclid_sim.max(dim=1)

                            tau = 0.5
                            similarity_threshold = torch.sigmoid(self.cosine_threshold)
                            logits = (max_sim - similarity_threshold) * self.k1
                            binary_logits = torch.stack([-logits, logits], dim=1)
                            gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                            binary_selection = gumbel_selection[:, 1]

                            alpha = 1.0
                            learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                            tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                            epsilon = 1e-10
                            entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1)
                            quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                            quantile_val = torch.quantile(entropy, quantile_threshold)
                            entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                            final_selection = binary_selection * entropy_mask

                            perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            perturbed_pvs = output.clone()
                            perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                        else:
                            perturbed_pvs = output.clone()

                        perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                        perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                        results = self.attack_model(perturbed_pvs, prediction, targets)
                    else:
                        results = self.attack_model(output, prediction, targets)

                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        # ----- Post-Evaluation -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx

        final_result.extend([
            correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss
        ])

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        print(f"Test Acc: {100.*correct/(1.0*total):.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, "
            f"precision: {100.*prec/(1.0*batch_idx):.3f}, recall: {100.*recall/batch_idx:.3f}")

        return final_result, fpr, tpr


    def train_pearson(self, epoch, result_path, result_path_csv, mode): 
        self.attack_model.train()
        self.perturb_model.train()

        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []
        final_result = []

        lambda_contrast = 0.5  # Weight for contrastive loss

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break

                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)

                if self.attack_name == "apcmia":
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)

                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]         # (m, C)
                        non_member_pvs = output[non_member_indices] # (n, C)

                        # Normalize PVs: subtract mean
                        member_mean = member_pvs.mean(dim=1, keepdim=True)
                        member_std = member_pvs.std(dim=1, unbiased=False, keepdim=True) + 1e-8
                        member_norm = (member_pvs - member_mean) / member_std  # (m, C)

                        non_member_mean = non_member_pvs.mean(dim=1, keepdim=True)
                        non_member_std = non_member_pvs.std(dim=1, unbiased=False, keepdim=True) + 1e-8
                        non_member_norm = (non_member_pvs - non_member_mean) / non_member_std  # (n, C)

                        # Compute Pearson correlation (dot product after normalization)
                        pearson_corr = torch.matmul(non_member_norm, member_norm.T) / non_member_norm.size(1)  # (n, m)
                        max_corr, _ = pearson_corr.max(dim=1)  # Most similar member for each non-member

                        # Use threshold and gumbel-softmax to determine which non-members to perturb
                        tau = 0.5
                        pearson_threshold = torch.sigmoid(self.cosine_threshold)  # reuse threshold param
                        logits = (max_corr - pearson_threshold) * self.k1
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]  # (n,)

                        # Apply perturbation
                        alpha = 1
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                        # Entropy-based filtering
                        epsilon = 1e-10
                        entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        final_selection = binary_selection * entropy_mask

                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    # Normalize & forward pass
                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    results = self.attack_model(perturbed_pvs, prediction, targets)

                    # Contrastive Loss
                    margin = 0.5
                    embeddings = self.attack_model.get_embeddings(perturbed_pvs, prediction, targets)
                    contrast_loss = self.contrastive_loss(embeddings, members, margin)

                else:
                    results = self.attack_model(output, prediction, targets)
                    contrast_loss = 0

                attack_loss = self.criterion(results, members)
                total_loss = attack_loss + lambda_contrast * contrast_loss

                self.optimizer.zero_grad()
                self.optimizer_perturb.zero_grad()
                self.optimizer_cosine.zero_grad()
                self.optimizer_quantile_threshold.zero_grad()

                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer_perturb.step()
                self.optimizer_quantile_threshold.step()
                self.optimizer_cosine.step()

                train_loss += attack_loss.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_train_gndtrth.append(members)
                final_train_predict.append(predicted)
                final_train_probabe.append(results[:, 1])

                batch_idx += 1

        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

        train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
        train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

        final_result = [
            100. * correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            train_f1_score,
            train_roc_auc_score,
        ]

        print(f"Train Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Loss: {train_loss/batch_idx:.3f} "
            f"Precision: {100.*prec/batch_idx:.3f} Recall: {100.*recall/batch_idx:.3f}")
        print(f"Pearson Threshold (used): {torch.sigmoid(self.cosine_threshold).item():.4f}, "
            f"Quantile Threshold: {torch.sigmoid(self.Entropy_quantile_threshold).item():.4f}")

        return self.cosine_threshold.item(), self.Entropy_quantile_threshold.item()
    
    def test_pearson(self, epoch_flag, result_path, mode):
        self.attack_model.eval()
        self.perturb_model.eval()

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)

                    if self.attack_name == "apcmia":
                        member_mask = (members == 1)
                        non_member_mask = (members == 0)

                        if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                            member_indices = member_mask.nonzero(as_tuple=True)[0]
                            non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                            member_pvs = output[member_indices]
                            non_member_pvs = output[non_member_indices]

                            # Pearson similarity calculation
                            member_mean = member_pvs.mean(dim=1, keepdim=True)
                            member_std = member_pvs.std(dim=1, unbiased=False, keepdim=True) + 1e-8
                            member_norm = (member_pvs - member_mean) / member_std

                            non_member_mean = non_member_pvs.mean(dim=1, keepdim=True)
                            non_member_std = non_member_pvs.std(dim=1, unbiased=False, keepdim=True) + 1e-8
                            non_member_norm = (non_member_pvs - non_member_mean) / non_member_std

                            pearson_corr = torch.matmul(non_member_norm, member_norm.T) / non_member_norm.size(1)
                            max_corr, _ = pearson_corr.max(dim=1)

                            # Gumbel-softmax selection
                            tau = 0.5
                            pearson_threshold = torch.sigmoid(self.cosine_threshold)
                            logits = (max_corr - pearson_threshold) * self.k1
                            binary_logits = torch.stack([-logits, logits], dim=1)
                            gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                            binary_selection = gumbel_selection[:, 1]

                            # Entropy-filtered perturbation
                            alpha = 1.0
                            learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                            tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                            epsilon = 1e-10
                            entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1)
                            quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                            quantile_val = torch.quantile(entropy, quantile_threshold)
                            entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                            final_selection = binary_selection * entropy_mask

                            perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            perturbed_pvs = output.clone()
                            perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                        else:
                            perturbed_pvs = output.clone()

                        perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                        perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                        results = self.attack_model(perturbed_pvs, prediction, targets)
                    else:
                        results = self.attack_model(output, prediction, targets)

                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx

        final_result.extend([
            correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss
        ])

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        print(f"Test Acc: {100.*correct/total:.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, "
            f"Precision: {100.*prec/batch_idx:.3f}, Recall: {100.*recall/batch_idx:.3f}")

        return final_result, fpr, tpr


    def train_mahalanobis(self, epoch, result_path, result_path_csv, mode): 
        self.attack_model.train()
        self.perturb_model.train()

        batch_idx = 1
        train_loss = 0
        correct = 0
        prec = 0
        recall = 0
        total = 0

        bcm = BinaryConfusionMatrix().to(self.device)
        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []
        final_result = []

        lambda_contrast = 0.5  # Weight for contrastive loss

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break

                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)

                if self.attack_name == "apcmia":
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)

                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]         # (m, C)
                        non_member_pvs = output[non_member_indices] # (n, C)

                        # --- Compute Covariance Matrix for Member PVs ---
                        member_mean = member_pvs.mean(dim=0, keepdim=True)  # (1, C)
                        centered_member = member_pvs - member_mean
                        cov = centered_member.T @ centered_member / (member_pvs.size(0) - 1)  # (C, C)

                        # Invert covariance matrix (or use pseudo-inverse if singular)
                        try:
                            cov_inv = torch.inverse(cov)
                        except RuntimeError:
                            cov_inv = torch.pinverse(cov)

                        # --- Mahalanobis Distance: Each non-member to each member ---
                        delta = non_member_pvs.unsqueeze(1) - member_pvs.unsqueeze(0)  # (n, m, C)
                        dists = torch.einsum("nmc,cd,nmd->nm", delta, cov_inv, delta)  # (n, m)
                        mahala_sim = -dists  # Negative = similarity

                        max_sim, _ = mahala_sim.max(dim=1)  # (n,)

                        # --- Gumbel Selection ---
                        tau = 0.5
                        similarity_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_sim - similarity_threshold) * self.k1
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]

                        # --- Perturbation + Entropy Mask ---
                        alpha = 1.0
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                        epsilon = 1e-10
                        entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        final_selection = binary_selection * entropy_mask

                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values
                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    # Normalize perturbed PVs
                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    results = self.attack_model(perturbed_pvs, prediction, targets)

                    # Contrastive Loss
                    margin = 0.5
                    embeddings = self.attack_model.get_embeddings(perturbed_pvs, prediction, targets)
                    contrast_loss = self.contrastive_loss(embeddings, members, margin)

                else:
                    results = self.attack_model(output, prediction, targets)
                    contrast_loss = 0

                attack_loss = self.criterion(results, members)
                total_loss = attack_loss + lambda_contrast * contrast_loss

                self.optimizer.zero_grad()
                self.optimizer_perturb.zero_grad()
                self.optimizer_cosine.zero_grad()
                self.optimizer_quantile_threshold.zero_grad()

                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer_perturb.step()
                self.optimizer_quantile_threshold.step()
                self.optimizer_cosine.step()

                train_loss += attack_loss.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                conf_mat = bcm(predicted, members)
                prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                final_train_gndtrth.append(members)
                final_train_predict.append(predicted)
                final_train_probabe.append(results[:, 1])

                batch_idx += 1

        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

        train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
        train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

        final_result = [
            100. * correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            train_f1_score,
            train_roc_auc_score,
        ]

        print(f"Train Acc: {100.*correct/total:.3f}% ({correct}/{total}) | Loss: {train_loss/batch_idx:.3f} "
            f"Precision: {100.*prec/batch_idx:.3f} Recall: {100.*recall/batch_idx:.3f}")
        print(f"Mahalanobis Threshold (used): {torch.sigmoid(self.cosine_threshold).item():.4f}, "
            f"Quantile Threshold: {torch.sigmoid(self.Entropy_quantile_threshold).item():.4f}")

        return self.cosine_threshold.item(), self.Entropy_quantile_threshold.item()
    

    def test_mahalanobis(self, epoch_flag, result_path, mode):
        self.attack_model.eval()
        self.perturb_model.eval()

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)

                    if self.attack_name == "apcmia":
                        member_mask = (members == 1)
                        non_member_mask = (members == 0)

                        if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                            member_indices = member_mask.nonzero(as_tuple=True)[0]
                            non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                            member_pvs = output[member_indices]
                            non_member_pvs = output[non_member_indices]

                            # Compute covariance matrix from member PVs
                            member_mean = member_pvs.mean(dim=0, keepdim=True)
                            centered_member = member_pvs - member_mean
                            cov = centered_member.T @ centered_member / (member_pvs.size(0) - 1)

                            try:
                                cov_inv = torch.inverse(cov)
                            except RuntimeError:
                                cov_inv = torch.pinverse(cov)

                            # Compute Mahalanobis distance
                            delta = non_member_pvs.unsqueeze(1) - member_pvs.unsqueeze(0)  # (n, m, C)
                            dists = torch.einsum("nmc,cd,nmd->nm", delta, cov_inv, delta)  # (n, m)
                            mahala_sim = -dists  # Use as similarity

                            max_sim, _ = mahala_sim.max(dim=1)

                            # Gumbel-softmax based binary selection
                            tau = 0.5
                            similarity_threshold = torch.sigmoid(self.cosine_threshold)
                            logits = (max_sim - similarity_threshold) * self.k1
                            binary_logits = torch.stack([-logits, logits], dim=1)
                            gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                            binary_selection = gumbel_selection[:, 1]

                            alpha = 1.0
                            learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                            tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                            epsilon = 1e-10
                            entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1)
                            quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                            quantile_val = torch.quantile(entropy, quantile_threshold)
                            entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                            final_selection = binary_selection * entropy_mask

                            perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            perturbed_pvs = output.clone()
                            perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                        else:
                            perturbed_pvs = output.clone()

                        perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                        perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                        results = self.attack_model(perturbed_pvs, prediction, targets)
                    else:
                        results = self.attack_model(output, prediction, targets)

                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)
        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx

        final_result.extend([
            correct / total,
            (prec / batch_idx).item(),
            (recall / batch_idx).item(),
            test_f1_score,
            test_roc_auc_score,
            avg_test_loss
        ])

        with open(result_path, "wb") as f_out:
            pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        print(f"Test Acc: {100.*correct/total:.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, "
            f"Precision: {100.*prec/batch_idx:.3f}, Recall: {100.*recall/batch_idx:.3f}")

        return final_result, fpr, tpr
    def test_saved_model_apcmia(self, atk_model, prt_model, consin_thr, entrp_thr):
        
        # checkpoint = torch.load(models_path, map_location=self.device, weights_only=True)
        # self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        # self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        # self.cosine_threshold.data = checkpoint['cosine_threshold'].to(self.device)
        # self.Entropy_quantile_threshold.data = checkpoint['Entropy_quantile_threshold'].to(self.device)
        
        # torch.tensor(checkpoint['Entropy_quantile_threshold'], device=self.device)
        self.attack_model = atk_model
        self.perturb_model = prt_model
        self.cosine_threshold = torch.tensor(consin_thr, device=self.device)
        self.Entropy_quantile_threshold = torch.tensor(entrp_thr, device=self.device)

        # print(f"Cosine Threshold: {self.cosine_threshold.item():.4f}, Entropy Quantile Threshold: {self.Entropy_quantile_threshold.item():.4f}")
        # exit()
        # Set models to evaluation mode.
        self.attack_model.eval()
        self.perturb_model.eval()

        

        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    # Move tensors to device.
                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)
                    # targets can remain on CPU or be moved if needed.
                    if self.attack_name == "apcmia": #new

                        # Create masks for members and non-members.
                        member_mask = (members == 1)
                        non_member_mask = (members == 0)

                        if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                            # Get indices for members and non-members.
                            member_indices = member_mask.nonzero(as_tuple=True)[0]
                            non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                            # Extract PVs.
                            member_pvs = output[member_indices]       # (n_members, C)
                            non_member_pvs = output[non_member_indices] # (n_non_members, C)

                            # ----- Step 2: Overlap Detection via Cosine Similarity -----
                            cos_sim = F.cosine_similarity(
                                non_member_pvs.unsqueeze(1),  # (n_non_members, 1, C)
                                member_pvs.unsqueeze(0),      # (1, n_members, C)
                                dim=2
                            )
                            max_cos_sim, _ = cos_sim.max(dim=1)  # (n_non_members,)

                            # ----- Step 3: Differentiable Binary Selection using Gumbel–Softmax -----
                            temperature = self.k1  # scaling factor for logits
                            tau = 0.5           # Gumbel–Softmax temperature
                            # Reparameterize the cosine threshold to (0,1)
                            cosine_threshold = torch.sigmoid(self.cosine_threshold)
                            logits = (max_cos_sim - cosine_threshold) * temperature  # (n_non_members,)
                            binary_logits = torch.stack([-logits, logits], dim=1)  # (n_non_members, 2)
                            gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                            binary_selection = gumbel_selection[:, 1]  # (n_non_members,)

                            alpha = 1.0

                            # ----- Step 4: Perturbation with Entropy Filtering -----
                            learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                            # Compute tentative perturbed outputs.
                            tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                            # Compute entropy for each tentative perturbed PV.
                            epsilon = 1e-10
                            entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1).to(self.device)  # (n_non_members,)

                            # Select top samples based on entropy.
                            # For example, use a quantile threshold (self.Entropy_quantile_threshold should be a float in (0,1)).
                            # quantile_val = torch.quantile(entropy, 0.25)
                            quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                            quantile_val = torch.quantile(entropy, quantile_threshold)
                            # Use a sigmoid to create a soft entropy mask.
                            # self.k controls the steepness (if not defined, default to 50.0).
                            # k = self.k if hasattr(self, "k") else 50.0
                            entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)

                            # Combine the binary selection with the entropy mask.
                            final_selection = binary_selection * entropy_mask
                            perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                            # Replace non-member PVs in the overall output.
                            perturbed_pvs = output.clone()
                            perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                        else:
                            perturbed_pvs = output.clone()

                        # ----- Step 5: Normalize and Clip Perturbed PVs -----
                        perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                        perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                        # ----- Step 6: Forward Pass through the Attack Model -----
                        results = self.attack_model(perturbed_pvs, prediction, targets)
                    else:
                        results = self.attack_model(output, prediction, targets)

                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        # ----- Post Evaluation -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx
        
        


        final_result.extend([
        correct / total,
        (prec / batch_idx).item(),
        (recall / batch_idx).item(),
        test_f1_score,
        test_roc_auc_score,
        avg_test_loss  # Append average test loss
        ])

        # with open(models_apth, "wb") as f_out:
        #     pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f_out)

        
        print(f"Final: Test Acc: {100.*correct/(1.0*total):.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, precision: {100.*prec/(1.0*batch_idx):.3f}, recall: {100.*recall/batch_idx:.3f}")


       
        return final_result
    
    def test_saved_model_rest(self, model):
        
        # checkpoint = torch.load(models_apth, map_location=self.device, weights_only=True)
        # # Load state dictionaries for the models.
        # self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        self.attack_model = model
        # Set models to evaluation mode.
        self.attack_model.eval()
        
        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    # Move tensors to device.
                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)
                    
                    results = self.attack_model(output, prediction, targets)
                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        # ----- Post Evaluation -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx
        
       


        final_result.extend([
        correct / total,
        (prec / batch_idx).item(),
        (recall / batch_idx).item(),
        test_f1_score,
        test_roc_auc_score,
        avg_test_loss  # Append average test loss
        ])

        print(f"Final: Test Acc: {100.*correct/(1.0*total):.3f}% ({correct}/{total}), Loss: {avg_test_loss:.3f}, precision: {100.*prec/(1.0*batch_idx):.3f}, recall: {100.*recall/batch_idx:.3f}")

        return final_result
    def compute_roc_curve_rest(self, model):
        
        # checkpoint = torch.load(models_apth, map_location=self.device, weights_only=True)
        # # Load state dictionaries for the models.
        # self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        self.attack_model = model
        # Set models to evaluation mode.
        self.attack_model.eval()
        
        batch_idx = 1
        correct = 0
        total = 0
        prec = 0
        recall = 0
        total_test_loss = 0.0
        bcm = BinaryConfusionMatrix().to(self.device)

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []
        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    # Move tensors to device.
                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)
                    
                    results = self.attack_model(output, prediction, targets)
                    results = F.softmax(results, dim=1)
                    _, predicted = results.max(dim=1)

                    loss = self.criterion(results, members)
                    total_test_loss += loss.item()

                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    conf_mat = bcm(predicted, members)
                    prec += conf_mat[1, 1] / torch.sum(conf_mat[:, -1])
                    recall += conf_mat[1, 1] / torch.sum(conf_mat[-1, :])

                    final_test_gndtrth.append(members)
                    final_test_predict.append(predicted)
                    final_test_probabe.append(results[:, 1])

                    batch_idx += 1

        # ----- Post Evaluation -----
        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
        test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

        avg_test_loss = total_test_loss / batch_idx
        
       


        final_result.extend([
        correct / total,
        (prec / batch_idx).item(),
        (recall / batch_idx).item(),
        test_f1_score,
        test_roc_auc_score,
        avg_test_loss  # Append average test loss
        ])
        

        fpr, tpr, thresholds = roc_curve(final_test_gndtrth, final_test_probabe)
        roc_auc = auc(fpr, tpr)

        # if plot:
        #     plt.figure(figsize=(8, 6))
        #     plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc, lw=2)
        #     plt.plot([0, 1], [0, 1], 'k--', lw=2)
        #     plt.xlim([0.0, 1.0])
        #     plt.ylim([0.0, 1.05])
        #     plt.xlabel("False Positive Rate")
        #     plt.ylabel("True Positive Rate")
        #     plt.title("Receiver Operating Characteristic")
        #     plt.legend(loc="lower right")
        #     if save_path is not None:
        #         plt.savefig(save_path)
        #     plt.show()

        return fpr, tpr, thresholds, roc_auc

    def compute_roc_curve_apcmia(self, atk_model, prt_model, consin_thr, entrp_thr):
        """
        Computes the ROC curve (FPR, TPR, thresholds) and ROC AUC for the attack model,
        using the test set stored in self.ATTACK_SETS + "test.p". It follows the same
        perturbation/selection procedure as in self.test() to get final attack predictions.
        
        Args:
            plot (bool): Whether to plot the ROC curve.
            save_path (str): If provided, save the ROC plot to this file.
        
        Returns:
            fpr (np.ndarray): False positive rates.
            tpr (np.ndarray): True positive rates.
            thresholds (np.ndarray): Thresholds used.
            roc_auc (float): Area under the ROC curve.
        """

        # checkpoint = torch.load(models_apth, map_location=self.device, weights_only=True)
        # # Load state dictionaries for the models.
        # self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        # self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        # # Restore the learned threshold parameters.
        # self.cosine_threshold.data = checkpoint['cosine_threshold'].to(self.device)
        # self.Entropy_quantile_threshold.data = checkpoint['Entropy_quantile_threshold'].to(self.device)
        
        self.attack_model = atk_model
        self.perturb_model = prt_model
        self.cosine_threshold = torch.tensor(consin_thr, device=self.device)
        self.Entropy_quantile_threshold = torch.tensor(entrp_thr, device=self.device)
        # print(f"Cosine Threshold: {self.cosine_threshold.item():.4f}, Entropy Quantile Threshold: {self.Entropy_quantile_threshold.item():.4f}")
        # exit()

        # Set models to evaluation mode.
        self.attack_model.eval()
        self.perturb_model.eval()
        
        # # Set models to evaluation mode.
        # self.attack_model.eval()
        # self.perturb_model.eval()
        
        final_ground_truth = []
        final_probabilities = []
        
        with torch.no_grad():
            test_file = self.ATTACK_SETS + "test.p"
            with open(test_file, "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    # Move tensors to device.
                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)
                    # (targets can remain on CPU if not used for perturbation)

                    # Create masks for members and non-members.
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)

                    # Compute perturbed outputs using the same procedure as in test().
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]
                        
                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]
                        
                        # Overlap detection via cosine similarity.
                        cos_sim = F.cosine_similarity(
                            non_member_pvs.unsqueeze(1),  # shape: (n_non_members, 1, C)
                            member_pvs.unsqueeze(0),      # shape: (1, n_members, C)
                            dim=2
                        )
                        max_cos_sim, _ = cos_sim.max(dim=1)  # (n_non_members,)
                        
                        # Use Gumbel–Softmax for binary selection.
                        temperature = self.k1
                        tau = 0.5
                        cosine_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_cos_sim - cosine_threshold) * temperature
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]  # (n_non_members,)
                        
                        alpha = 1.0
                        # Compute the learned perturbations.
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                        # Compute the entropy of each tentative perturbed PV.
                        epsilon = 1e-10
                        entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1)
                        
                        # Compute a quantile value from entropy.
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy, quantile_threshold)
                        # Use a differentiable (soft) mask for entropy; self.k is a steepness factor.
                        entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                        
                        # Combine binary selection with the entropy mask.
                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                        # Replace the non-member PVs in the overall output.
                        perturbed_pvs = output.clone()
                        perturbed_pvs[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_pvs = output.clone()

                    # Normalize and clip the perturbed PVs.
                    perturbed_pvs = torch.clamp(perturbed_pvs, min=1e-6, max=1)
                    perturbed_pvs = perturbed_pvs / perturbed_pvs.sum(dim=1, keepdim=True)

                    # Forward pass through the attack model.
                    results = self.attack_model(perturbed_pvs, prediction, targets)
                    results = F.softmax(results, dim=1)
                    # results = self.attack_model(perturbed_pvs, prediction, targets)
                    # results = F.softmax(results, dim=1)

                    # _, predicted = results.max(dim=1)
                    # We assume that column 1 gives the probability for membership.
                    probabilities = results[:, 1]
                    # print(f"First few probabilities: {probabilities[:5]}")
                    # print(f"First few members: {members[:5]}")
                    # print(f"predicted: {predicted[:5]}")
                    # exit()

                    final_ground_truth.append(members.cpu())
                    final_probabilities.append(probabilities.cpu())
                    # exit()
            # Concatenate collected results.
            final_ground_truth = torch.cat(final_ground_truth, dim=0).numpy()
            final_probabilities = torch.cat(final_probabilities, dim=0).numpy()
        
        # Compute the ROC curve and ROC AUC.
        fpr, tpr, thresholds = roc_curve(final_ground_truth, final_probabilities)
        roc_auc = auc(fpr, tpr)

        # if plot:
        #     plt.figure(figsize=(8, 6))
        #     plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc, lw=2)
        #     plt.plot([0, 1], [0, 1], 'k--', lw=2)
        #     plt.xlim([0.0, 1.0])
        #     plt.ylim([0.0, 1.05])
        #     plt.xlabel("False Positive Rate")
        #     plt.ylabel("True Positive Rate")
        #     plt.title("Receiver Operating Characteristic")
        #     plt.legend(loc="lower right")
        #     if save_path is not None:
        #         plt.savefig(save_path)
        #     plt.show()

        return fpr, tpr, thresholds, roc_auc

    def compute_entropy_distribution_2(self, models_apth, dataset="test", plot=True, save_path=None):
        """
        Loads the saved models and thresholds from 'models_apth', then computes and (optionally)
        plots the entropy distributions of probability vectors (PVs) before and after perturbation.
        The resulting plot has two subplots:
        - Top: members vs. non-members (original entropies, before perturbation)
        - Bottom: members vs. non-members (perturbed entropies, after perturbation)
        """

        import os
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        # 1. Load checkpoint and restore attack/perturb models + thresholds.
        checkpoint = torch.load(models_apth, map_location=self.device, weights_only=True)
        self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        self.cosine_threshold.data = checkpoint['cosine_threshold'].to(self.device)
        self.Entropy_quantile_threshold.data = checkpoint['Entropy_quantile_threshold'].to(self.device)

        self.attack_model.eval()
        self.perturb_model.eval()

        # 2. Select the file based on dataset (train/test).
        if dataset.lower() == "test":
            file_path = self.ATTACK_SETS + "test.p"
        elif dataset.lower() == "train":
            file_path = self.ATTACK_SETS + "train.p"
        else:
            raise ValueError("Dataset must be 'test' or 'train'.")

        # We'll store four lists of entropies:
        # (a) Members' original entropies
        # (b) Non-members' original entropies
        # (c) Members' perturbed entropies
        # (d) Non-members' perturbed entropies
        members_orig_list = []
        nonmembers_orig_list = []
        members_pert_list = []
        nonmembers_pert_list = []

        epsilon = 1e-10
        k = self.k if hasattr(self, "k") else 50.0  # steepness factor for entropy filtering

        with torch.no_grad():
            with open(file_path, "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    # Move tensors to device
                    output = output.to(self.device)
                    members = members.to(self.device)
                    targets = targets.to(self.device)

                    # If output is not already a probability distribution, you might do:
                    # output = F.softmax(output, dim=1)

                    # (A) Original entropies
                    orig_entropy = -torch.sum(output * torch.log(output + epsilon), dim=1)

                    # Separate members vs. non-members for original entropies
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)
                    # Extend to CPU lists
                    members_orig_list.extend(orig_entropy[member_mask].cpu().numpy())
                    nonmembers_orig_list.extend(orig_entropy[non_member_mask].cpu().numpy())

                    # (B) Compute perturbed output using the same logic as in your test function
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]

                        # Overlap detection
                        cos_sim = F.cosine_similarity(
                            non_member_pvs.unsqueeze(1),
                            member_pvs.unsqueeze(0),
                            dim=2
                        )
                        max_cos_sim, _ = cos_sim.max(dim=1)

                        temperature = 10.0
                        tau = 0.5
                        cos_thresh = torch.sigmoid(self.cosine_threshold)
                        logits = (max_cos_sim - cos_thresh) * temperature
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]

                        alpha = 1.0
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                        # Entropy filtering
                        entropy_vals = -torch.sum(tentative_perturbed * torch.log(tentative_perturbed + epsilon), dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy_vals, quantile_threshold)
                        entropy_mask = torch.sigmoid((entropy_vals - quantile_val) * k)

                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                        # Build the final perturbed output
                        perturbed_output = output.clone()
                        perturbed_output[non_member_indices] = perturbed_non_member_pvs
                    else:
                        # If no members or no non-members, just keep the output as-is
                        perturbed_output = output.clone()

                    # Normalize & clip
                    perturbed_output = torch.clamp(perturbed_output, min=1e-6, max=1)
                    perturbed_output = perturbed_output / perturbed_output.sum(dim=1, keepdim=True)

                    # (C) Perturbed entropies
                    pert_entropy = -torch.sum(perturbed_output * torch.log(perturbed_output + epsilon), dim=1)

                    # Separate members vs. non-members for perturbed entropies
                    members_pert_list.extend(pert_entropy[member_mask].cpu().numpy())
                    nonmembers_pert_list.extend(pert_entropy[non_member_mask].cpu().numpy())

        # If plot=True, we create two subplots: top for "before", bottom for "after"
        if plot:
            # Choose a style
            if "seaborn-darkgrid" in plt.style.available:
                plt.style.use("seaborn-darkgrid")
            elif "seaborn" in plt.style.available:
                plt.style.use("seaborn")
            else:
                plt.style.use("default")

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)
            fig.suptitle("Entropy Distribution Before and After Perturbation", fontsize=16, fontweight="bold")

            # Colors
            color_members = "#E74C3C"     # Professional red
            color_nonmembers = "#2E86C1"  # Steel blue

            # Top subplot: original entropies
            axes[0].hist(members_orig_list, bins=50, alpha=0.5, label="Members (before)",
                        color=color_members, edgecolor="black")
            axes[0].hist(nonmembers_orig_list, bins=50, alpha=0.5, label="Non-members (before)",
                        color=color_nonmembers, edgecolor="black")
            axes[0].set_ylabel("Frequency", fontsize=13)
            axes[0].legend(fontsize=12)
            axes[0].grid(True, linestyle="--", alpha=0.5)
            axes[0].set_title("Before Perturbation", fontsize=14, fontweight="bold")

            # Bottom subplot: perturbed entropies
            axes[1].hist(members_pert_list, bins=50, alpha=0.7, label="Members (after)",
                        color=color_members, edgecolor="black")
            axes[1].hist(nonmembers_pert_list, bins=50, alpha=0.7, label="Non-members (after)",
                        color=color_nonmembers, edgecolor="black")
            axes[1].set_xlabel("Entropy", fontsize=13)
            axes[1].set_ylabel("Frequency", fontsize=13)
            axes[1].legend(fontsize=12)
            axes[1].grid(True, linestyle="--", alpha=0.5)
            axes[1].set_title("After Perturbation", fontsize=14, fontweight="bold")

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leaves space for the suptitle
            if save_path is not None:
                plt.savefig(save_path, dpi=300)
            plt.show()

        # Return the arrays if you need them
        return (np.array(members_orig_list),
                np.array(nonmembers_orig_list),
                np.array(members_pert_list),
                np.array(nonmembers_pert_list))


    # def approximate_perturbation_distribution(self, loader):
    def approximate_perturbation_distribution(self):
        """
        Load all samples from self.ATTACK_SETS + "train.p" and separate them by membership.
        Then compute the per-group mean and standard deviation of the output vectors.
        Returns:
            member_mean, member_std, non_member_mean, non_member_std
        """
        member_samples = []
        non_member_samples = []
        
        file_path = self.ATTACK_SETS + "train.p"
        with open(file_path, "rb") as f:
            while True:
                try:
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break
                # Ensure output and members are on CPU.
                output = output.cpu()
                members = members.cpu()
                for i in range(output.size(0)):
                    if members[i].item() == 1:
                        member_samples.append(output[i])
                    else:
                        non_member_samples.append(output[i])
                        
        if member_samples:
            member_tensor = torch.stack(member_samples)
            member_mean = member_tensor.mean(dim=0)
            member_std = member_tensor.std(dim=0)
        else:
            member_mean, member_std = None, None

        if non_member_samples:
            non_member_tensor = torch.stack(non_member_samples)
            non_member_mean = non_member_tensor.mean(dim=0)
            non_member_std = non_member_tensor.std(dim=0)
        else:
            non_member_mean, non_member_std = None, None

        return member_mean, member_std, non_member_mean, non_member_std
    
    
    def compute_entropy_distribution(self, atk_model, prt_model, consin_thr, entrp_thr, entropy_dis_dr):
        """
        Computes and (optionally) plots the entropy distribution of the probability vectors (PVs)
        before and after perturbation. It uses the same perturbation procedure as in your test function.
        
        Args:
            dataset (str): Which dataset to use ("test" or "train"). Default is "test".
            plot (bool): Whether to plot the histogram of entropies.
            save_path (str): If provided, the histogram figure will be saved to this path.
        
        Returns:
            original_entropies (np.ndarray): Array of entropy values computed from original PVs.
            perturbed_entropies (np.ndarray): Array of entropy values computed from perturbed PVs.
        """
        # Load checkpoint using weights_only=True for security.
        # checkpoint = torch.load(models_apth, map_location=self.device, weights_only=True)
        # self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        # self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        # self.cosine_threshold.data = checkpoint['cosine_threshold'].to(self.device)
        # self.Entropy_quantile_threshold.data = checkpoint['Entropy_quantile_threshold'].to(self.device)

        # checkpoint = torch.load(last_checkpoint, weights_only=True)
        # self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        # self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        # self.cosine_threshold = checkpoint['cosine_threshold']
        # self.Entropy_quantile_threshold = checkpoint['entropy_threshold']
        
        # models_threshold_params = {
        #     'attack_model_state_dict': self.attack_model.state_dict(),
        #     'perturb_model_state_dict': self.perturb_model.state_dict(),
        #     'cosine_threshold': self.cosine_threshold,
        #     'Entropy_quantile_threshold': self.Entropy_quantile_threshold
        # }
        # torch.save(models_threshold_params, path)

        
        self.attack_model = atk_model
        self.perturb_model = prt_model
        self.cosine_threshold = torch.tensor(consin_thr, device=self.device)
        self.Entropy_quantile_threshold = torch.tensor(entrp_thr, device=self.device)

        # Set models to evaluation mode.
        self.attack_model.eval()
        self.perturb_model.eval()

        epsilon = 1e-10
        original_entropy_list = []
        perturbed_entropy_list = []

        # Select the file based on dataset flag.
        # if dataset.lower() == "test":
        file_path = self.ATTACK_SETS + "test.p"
        # elif dataset.lower() == "train":
        #     file_path = self.ATTACK_SETS + "train.p"
        # else:
        #     raise ValueError("Dataset must be 'test' or 'train'.")

        with torch.no_grad():
            with open(file_path, "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break

                    # Move data to device.
                    output = output.to(self.device)
                    # If output is not already probabilities, you might want to apply softmax:
                    # output = F.softmax(output, dim=1)
                    members = members.to(self.device)
                    targets = targets.to(self.device)

                    # Compute the original entropy for each sample.
                    # (Assumes output is a probability vector that sums to 1.)
                    orig_entropy = -torch.sum(output * torch.log(output + epsilon), dim=1)
                    original_entropy_list.append(orig_entropy.cpu())

                    # --- Compute the perturbed output using the same procedure as in test() ---
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)

                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]
                        
                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]
                        
                        # Overlap detection via cosine similarity.
                        cos_sim = F.cosine_similarity(
                            non_member_pvs.unsqueeze(1),
                            member_pvs.unsqueeze(0),
                            dim=2
                        )
                        max_cos_sim, _ = cos_sim.max(dim=1)
                        
                        # Use Gumbel–Softmax to obtain binary selection.
                        temperature = 10.0
                        tau = 0.5
                        cosine_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_cos_sim - cosine_threshold) * temperature
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]
                        
                        alpha = 1.0
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices].to(self.device))
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values
                        
                        # Entropy filtering:
                        entropy_vals = -torch.sum(tentative_perturbed * torch.log(tentative_perturbed + epsilon), dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy_vals, quantile_threshold)
                        # Use self.k (steepness factor) if defined; otherwise default to 50.
                        k = self.k
                        entropy_mask = torch.sigmoid((entropy_vals - quantile_val) * k)
                        
                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values
                        
                        # Replace non-member outputs with perturbed ones.
                        perturbed_output = output.clone()
                        perturbed_output[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_output = output.clone()

                    # Normalize and clip (if needed).
                    perturbed_output = torch.clamp(perturbed_output, min=1e-6, max=1)
                    perturbed_output = perturbed_output / perturbed_output.sum(dim=1, keepdim=True)
                    # Compute perturbed entropy.
                    pert_entropy = -torch.sum(perturbed_output * torch.log(perturbed_output + epsilon), dim=1)
                    perturbed_entropy_list.append(pert_entropy.cpu())

            # Concatenate all batches.
            original_entropies = torch.cat(original_entropy_list, dim=0).numpy()
            perturbed_entropies = torch.cat(perturbed_entropy_list, dim=0).numpy()
            
        import numpy as np
        max_entropy = np.log2(self.num_classes)  # ≈ 3.3219
        original_entropies = original_entropies / max_entropy
        perturbed_entropies = perturbed_entropies / max_entropy

        # Plot settings as provided:
        size = 20
        params = {
            'axes.labelsize': size,
            'font.size': size,
            'legend.fontsize': size,
            'xtick.labelsize': size,
            'ytick.labelsize': size,
            'figure.figsize': [10, 8],
            "font.family": "arial",
        }
        plt.rcParams.update(params)

        # Create the figure and axis
        fig, ax = plt.subplots()

        # Define colors for the bars
        color_orig = "#2421f7"  # Blue for original entropies
        color_pert = "#f10219"  # Red for perturbed entropies

        # Compute histogram counts (not normalized to density)
        orig_counts, bin_edges = np.histogram(original_entropies, bins=20, density=False)
        pert_counts, _ = np.histogram(perturbed_entropies, bins=20, density=False)

        # Calculate bar width: split the bin width between the two sets of bars
        bin_width = bin_edges[1] - bin_edges[0]
        width_factor = 0.95  # Increase for thicker bars
        width = bin_width * width_factor / 2.0  # half width per bar

        # Enable grid lines and set grid behind the plot elements
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)

        # Plot the histogram bars for original entropies
        ax.bar(
            bin_edges[:-1],
            orig_counts,
            width=width,
            color=color_orig,
            label="Before",
            align='edge'
        )

        # Plot the histogram bars for perturbed entropies, shifted to the right by 'width'
        ax.bar(
            bin_edges[:-1] + width,
            pert_counts,
            width=width,
            color=color_pert,
            label="After",
            align='edge'
        )

        # Set the x-axis label

        # Get handles and labels from the axis to create a legend
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles, 
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1),  # Place the legend above the plot
            ncol=2,
            fontsize=size
        )
        # legend.get_frame().set_facecolor("0.95")
        legend.get_frame().set_edgecolor("0.91")

        # Adjust the layout to prevent clipping
        # plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        ax.set_xlabel("Prediction Uncertainty", fontsize=size)
        ax.set_ylabel("Frequency", fontsize=size)
        # plt.xlabel("False Positive Rate", )
        # plt.ylabel("True Positive Rate", fontsize=size)

        # Ensure the output directory exists
        os.makedirs(entropy_dis_dr, exist_ok=True)
        
        # Build the output path and save the figure
        output_path = os.path.join(entropy_dis_dr, self.dataset_name + "_entropy_dist.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved entropy distribution plot to {output_path}")
                
        return original_entropies, perturbed_entropies
    

    def compute_entropy_distribution_new(self, atk_model, prt_model, consin_thr, entrp_thr, entropy_dis_dr):
        """
        Computes and (optionally) plots the entropy distribution of the probability vectors (PVs)
        before and after perturbation, separately for members and non-members.
        
        The function:
        - Loads data from a file (assumed to be self.ATTACK_SETS + "test.p"),
        - Computes the original entropy for each sample,
        - Applies the perturbation procedure to non-member samples,
        - Computes the perturbed entropy,
        - Separates entropies by membership status,
        - Normalizes entropies by the maximum possible entropy (log2(num_classes)),
        - Plots subplots for "Before Perturbation" and "After Perturbation" showing histograms
            for members and non-members, and saves the figure.
        
        Returns:
            original_members, original_nonmembers, pert_members, pert_nonmembers : np.ndarray
        """
        # Set up the models and thresholds
        self.attack_model = atk_model
        self.perturb_model = prt_model
        self.cosine_threshold = torch.tensor(consin_thr, device=self.device)
        self.Entropy_quantile_threshold = torch.tensor(entrp_thr, device=self.device)
        self.attack_model.eval()
        self.perturb_model.eval()
        
        epsilon = 1e-10
        # Lists to store entropies for members and non-members (before and after perturbation)
        orig_members_list = []
        orig_nonmembers_list = []
        pert_members_list = []
        pert_nonmembers_list = []

        file_path = self.ATTACK_SETS + "test.p"
        
        with torch.no_grad():
            with open(file_path, "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    output = output.to(self.device)
                    members = members.to(self.device)
                    targets = targets.to(self.device)

                    # Compute original entropy: H = -sum(p * log(p))
                    orig_entropy = -torch.sum(output * torch.log(output + epsilon), dim=1)
                    # Separate based on membership flag (members==1)
                    orig_members = orig_entropy[members == 1]
                    orig_nonmembers = orig_entropy[members == 0]
                    orig_members_list.append(orig_members.cpu())
                    orig_nonmembers_list.append(orig_nonmembers.cpu())

                    # --- Perturbation procedure for non-members ---
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]
                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]
                        
                        # Overlap detection via cosine similarity.
                        cos_sim = F.cosine_similarity(
                            non_member_pvs.unsqueeze(1),
                            member_pvs.unsqueeze(0),
                            dim=2
                        )
                        max_cos_sim, _ = cos_sim.max(dim=1)
                        
                        # Use Gumbel–Softmax for binary selection.
                        temperature = 10.0
                        tau = 0.5
                        cosine_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_cos_sim - cosine_threshold) * temperature
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]
                        
                        alpha = 1.0
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices].to(self.device))
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values
                        
                        # Entropy filtering:
                        entropy_vals = -torch.sum(tentative_perturbed * torch.log(tentative_perturbed + epsilon), dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy_vals, quantile_threshold)
                        k = self.k  # steepness factor
                        entropy_mask = torch.sigmoid((entropy_vals - quantile_val) * k)
                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values
                        
                        perturbed_output = output.clone()
                        perturbed_output[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_output = output.clone()
                    
                    perturbed_output = torch.clamp(perturbed_output, min=1e-6, max=1)
                    perturbed_output = perturbed_output / perturbed_output.sum(dim=1, keepdim=True)
                    pert_entropy = -torch.sum(perturbed_output * torch.log(perturbed_output + epsilon), dim=1)
                    # Separate perturbed entropies by membership.
                    pert_members = pert_entropy[members == 1]
                    pert_nonmembers = pert_entropy[members == 0]
                    pert_members_list.append(pert_members.cpu())
                    pert_nonmembers_list.append(pert_nonmembers.cpu())

        # Concatenate all batches.
        original_members = torch.cat(orig_members_list, dim=0).numpy()
        original_nonmembers = torch.cat(orig_nonmembers_list, dim=0).numpy()
        pert_members = torch.cat(pert_members_list, dim=0).numpy()
        pert_nonmembers = torch.cat(pert_nonmembers_list, dim=0).numpy()

        # Normalize entropies by the maximum possible entropy: log2(num_classes)
        max_entropy = np.log2(self.num_classes)
        original_members = original_members / max_entropy
        original_nonmembers = original_nonmembers / max_entropy
        pert_members = pert_members / max_entropy
        pert_nonmembers = pert_nonmembers / max_entropy

        # --- Plot settings ---
        size = 20
        params = {
            'axes.labelsize': size,
            'font.size': size,
            'legend.fontsize': size,
            'xtick.labelsize': size-5,
            'ytick.labelsize': size-5,
            'figure.figsize': [10, 8],
            "font.family": "arial",
        }
        plt.rcParams.update(params)

        # --- Create subplots: left for "Before", right for "After" ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.grid(linestyle='dotted')
        ax2.grid(linestyle='dotted')
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        bins = 20

        # Left subplot: Original Entropies
        
        counts_orig_members, bin_edges = np.histogram(original_members, bins=bins, density=False)
        counts_orig_nonmembers, _ = np.histogram(original_nonmembers, bins=bins, density=False)
        bin_width = bin_edges[1] - bin_edges[0]
        width = bin_width * 0.95 / 2.0
        ax1.bar(bin_edges[:-1] - width/2, counts_orig_members, width=width,
                color="#2421f7", label="Members")
        ax1.bar(bin_edges[:-1] + width/2, counts_orig_nonmembers, width=width,
                color="#f10219", label="Non-Members")
        ax1.set_xlabel("Prediction Uncertainty", fontsize=size)
        ax1.set_ylabel("Frequency", fontsize=size)
        ax1.set_title("Before ", fontsize=size, fontweight="bold")
        ax1.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1), fontsize=size)

        # Right subplot: Perturbed Entropies
        counts_pert_members, bin_edges = np.histogram(pert_members, bins=bins, density=False)
        counts_pert_nonmembers, _ = np.histogram(pert_nonmembers, bins=bins, density=False)
        ax2.bar(bin_edges[:-1] - width/2, counts_pert_members, width=width,
                color="#2421f7", label="Members")
        ax2.bar(bin_edges[:-1] + width/2, counts_pert_nonmembers, width=width,
                color="#f10219", label="Non-Members")
        ax2.set_xlabel("Prediction Uncertainty", fontsize=size)
        # ax2.set_ylabel("Frequency", fontsize=size)
        ax2.set_title("After", fontsize=size, fontweight="bold")
        # ax2.grid(linestyle='dotted')
        ax2.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1), fontsize=size)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(entropy_dis_dr, exist_ok=True)
        output_path = os.path.join(entropy_dis_dr, self.dataset_name + "_entropy_dist.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved entropy distribution plot to {output_path}")
        # return original_members, original_nonmembers, pert_members, pert_nonmembers

        # return original_entropies, perturbed_entropies
    def compute_entropy_distribution_new_norm(self, atk_model, prt_model, consin_thr, entrp_thr, entropy_dis_dr):
        """
        Computes and (optionally) plots the entropy distribution of the probability vectors (PVs)
        before and after perturbation, separately for members and non-members.
        
        The function:
        - Loads data from a file (assumed to be self.ATTACK_SETS + "test.p"),
        - Computes the original entropy for each sample,
        - Applies the perturbation procedure to non-member samples,
        - Computes the perturbed entropy,
        - Separates entropies by membership status,
        - Normalizes entropies by the maximum possible entropy (log2(num_classes)),
        - Plots subplots for "Before Perturbation" and "After Perturbation" showing normalized histograms
            for members and non-members, and saves the figure.
        
        Returns:
            original_members, original_nonmembers, pert_members, pert_nonmembers : np.ndarray
        """
        # Set up the models and thresholds
        self.attack_model = atk_model
        self.perturb_model = prt_model
        self.cosine_threshold = torch.tensor(consin_thr, device=self.device)
        self.Entropy_quantile_threshold = torch.tensor(entrp_thr, device=self.device)
        self.attack_model.eval()
        self.perturb_model.eval()
        
        epsilon = 1e-10
        # Lists to store entropies for members and non-members (before and after perturbation)
        orig_members_list = []
        orig_nonmembers_list = []
        pert_members_list = []
        pert_nonmembers_list = []

        file_path = self.ATTACK_SETS + "test.p"
        
        with torch.no_grad():
            with open(file_path, "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    output = output.to(self.device)
                    members = members.to(self.device)
                    targets = targets.to(self.device)

                    # Compute original entropy: H = -sum(p * log(p))
                    orig_entropy = -torch.sum(output * torch.log(output + epsilon), dim=1)
                    # Separate based on membership flag (members==1)
                    orig_members = orig_entropy[members == 1]
                    orig_nonmembers = orig_entropy[members == 0]
                    orig_members_list.append(orig_members.cpu())
                    orig_nonmembers_list.append(orig_nonmembers.cpu())

                    # --- Perturbation procedure for non-members ---
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]
                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]
                        
                        # Overlap detection via cosine similarity.
                        cos_sim = F.cosine_similarity(
                            non_member_pvs.unsqueeze(1),
                            member_pvs.unsqueeze(0),
                            dim=2
                        )
                        max_cos_sim, _ = cos_sim.max(dim=1)
                        
                        # Use Gumbel–Softmax for binary selection.
                        temperature = 10.0
                        tau = 0.5
                        cosine_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_cos_sim - cosine_threshold) * temperature
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]
                        
                        alpha = 1.0
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices].to(self.device))
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values
                        
                        # Entropy filtering:
                        entropy_vals = -torch.sum(tentative_perturbed * torch.log(tentative_perturbed + epsilon), dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy_vals, quantile_threshold)
                        k = self.k  # steepness factor
                        entropy_mask = torch.sigmoid((entropy_vals - quantile_val) * k)
                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values
                        
                        perturbed_output = output.clone()
                        perturbed_output[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed_output = output.clone()
                    
                    perturbed_output = torch.clamp(perturbed_output, min=1e-6, max=1)
                    perturbed_output = perturbed_output / perturbed_output.sum(dim=1, keepdim=True)
                    pert_entropy = -torch.sum(perturbed_output * torch.log(perturbed_output + epsilon), dim=1)
                    # Separate perturbed entropies by membership.
                    pert_members = pert_entropy[members == 1]
                    pert_nonmembers = pert_entropy[members == 0]
                    pert_members_list.append(pert_members.cpu())
                    pert_nonmembers_list.append(pert_nonmembers.cpu())

        # Concatenate all batches.
        original_members = torch.cat(orig_members_list, dim=0).numpy()
        original_nonmembers = torch.cat(orig_nonmembers_list, dim=0).numpy()
        pert_members = torch.cat(pert_members_list, dim=0).numpy()
        pert_nonmembers = torch.cat(pert_nonmembers_list, dim=0).numpy()

        # Normalize entropies by the maximum possible entropy: log2(num_classes)
        max_entropy = np.log2(self.num_classes)
        original_members = original_members / max_entropy
        original_nonmembers = original_nonmembers / max_entropy
        pert_members = pert_members / max_entropy
        pert_nonmembers = pert_nonmembers / max_entropy

        
        # size = 30
        # params = {
        #     'axes.labelsize': size,
        #     'font.size': size,
        #     'legend.fontsize': size,
        #     'xtick.labelsize': size,
        #     'ytick.labelsize': size,
        #     'figure.figsize': [10, 8],
        #     "font.family": "arial",
        # }

        # # --- Plot settings ---
        # # size = 20
        # # params = {
        # #     'axes.labelsize': size,
        # #     'font.size': size,
        # #     'legend.fontsize': size,
        # #     'xtick.labelsize': size-5,
        # #     'ytick.labelsize': size-5,
        # #     'figure.figsize': [10, 8],
        # #     "font.family": "arial",
        # # }
        # plt.rcParams.update(params)

        # # --- Create subplots: left for "Before", right for "After" ---
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        # ax1.grid(linestyle='dotted')
        # ax2.grid(linestyle='dotted')
        # ax1.set_axisbelow(True)
        # ax2.set_axisbelow(True)
        # bins = 20

        # # Left subplot: Original Entropies (normalized histogram)
        # # Set density=True to normalize on the y-axis.
        # counts_orig_members, bin_edges = np.histogram(original_members, bins=bins, density=True)
        # counts_orig_nonmembers, _ = np.histogram(original_nonmembers, bins=bins, density=True)
        # bin_width = bin_edges[1] - bin_edges[0]
        # width = bin_width * 0.95 / 2.0
        # ax1.bar(bin_edges[:-1] - width/2, counts_orig_members, width=width,
        #         color="#2421f7", label="Member")
        # ax1.bar(bin_edges[:-1] + width/2, counts_orig_nonmembers, width=width,
        #         color="#f10219", label="Non-Member")
        # ax1.set_xlabel("Prediction Uncertainty", fontsize=size)
        # ax1.set_ylabel("Normalized Frequency", fontsize=size)
        # ax1.set_title("Before", fontsize=size, fontweight="bold")
        # ax1.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1), fontsize=size-8)

        # # Right subplot: Perturbed Entropies (normalized histogram)
        # counts_pert_members, bin_edges = np.histogram(pert_members, bins=bins, density=True)
        # counts_pert_nonmembers, _ = np.histogram(pert_nonmembers, bins=bins, density=True)
        # ax2.bar(bin_edges[:-1] - width/2, counts_pert_members, width=width,
        #         color="#2421f7", label="Member")
        # ax2.bar(bin_edges[:-1] + width/2, counts_pert_nonmembers, width=width,
        #         color="#f10219", label="Non-Member")
        # ax2.set_xlabel("Prediction Uncertainty", fontsize=size)
        # ax2.set_title("After", fontsize=size, fontweight="bold")
        # ax2.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1), fontsize=size-8)

        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # os.makedirs(entropy_dis_dr, exist_ok=True)
        # output_path = os.path.join(entropy_dis_dr, self.dataset_name + "_entropy_dist.pdf")
        # plt.savefig(output_path, dpi=300, bbox_inches="tight")
        # plt.close()

        # print(f"Saved entropy distribution plot to {output_path}")

# sssssssssssssssssss
        size = 30
        params = {
            'axes.labelsize': size,
            'font.size': size,
            'legend.fontsize': size,
            'xtick.labelsize': size,
            'ytick.labelsize': size,
            'figure.figsize': [10, 8],
            "font.family": "arial",
        }
        plt.rcParams.update(params)

        # Create a single figure (no subplots)
        plt.figure()
        ax = plt.gca()

        # Enable grid and set it below plot elements.
        ax.grid(linestyle='dotted')
        ax.set_axisbelow(True)

        # Define histogram parameters.
        bins = 20
        # Compute normalized histograms for original_members and original_nonmembers.
        counts_orig_members, bin_edges = np.histogram(original_members, bins=bins, density=True)
        counts_orig_nonmembers, _ = np.histogram(original_nonmembers, bins=bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        width = bin_width * 0.95 / 2.0

        # Plot histogram bars.
        ax.bar(bin_edges[:-1] - width/2, counts_orig_members, width=width,
            color="#2421f7", label="Member")
        ax.bar(bin_edges[:-1] + width/2, counts_orig_nonmembers, width=width,
            color="#f10219", label="Non-Member")

        # Set labels and title.
        ax.set_xlabel("Prediction Uncertainty")
        ax.set_ylabel("Normalized Frequency")
        # ax.set_title("Before", fontweight="bold")

        # Set the legend in the lower right and customize its frame.
        legend = plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1), fontsize=size-8)
        # ax1.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1), fontsize=size)
        frame = legend.get_frame()
        frame.set_edgecolor("0.91")
        # Uncomment the next line if you wish to set a specific face color:
        # frame.set_facecolor('0.97')

        # Adjust subplot margin.
        plt.subplots_adjust(left=0.60)
        plt.tight_layout()

        # Define the output directory and file name.
        entropy_dis_dr = "./results"  # Update with your desired directory.
        os.makedirs(entropy_dis_dr, exist_ok=True)
        # Ensure that self.dataset_name exists; otherwise, replace it with a string.
        output_path = os.path.join(entropy_dis_dr, self.dataset_name + "_entropy_dist.pdf")

        # Save the figure as a PDF at 300 dpi.
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved entropy distribution plot to {output_path}")
       

    def compute_cosine_similarity_distribution(self, checkpoint_path, dataset="test", plot=True, save_path=None):
        """
        Loads the saved models and learned thresholds from checkpoint, then computes the cosine
        similarity between the original PVs and the perturbed PVs for each sample. It separates
        the similarity values for member and non-member samples.

        Args:
            checkpoint_path (str): Path to the checkpoint saved via save_att_per_thresholds_models.
            dataset (str): "test" or "train" to choose which dataset to use.
            plot (bool): Whether to plot the histogram of cosine similarities.
            save_path (str): If provided, the plot is saved to this file.

        Returns:
            member_similarities (np.ndarray): Cosine similarities for member samples.
            non_member_similarities (np.ndarray): Cosine similarities for non-member samples.
        """
        # Load checkpoint (using weights_only=True for security).
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        self.cosine_threshold.data = checkpoint['cosine_threshold'].to(self.device)
        self.Entropy_quantile_threshold.data = checkpoint['Entropy_quantile_threshold'].to(self.device)
        
        self.attack_model.eval()
        self.perturb_model.eval()

        member_sim_list = []
        non_member_sim_list = []
        
        # Choose dataset file.
        if dataset.lower() == "test":
            file_path = self.ATTACK_SETS + "test.p"
        elif dataset.lower() == "train":
            file_path = self.ATTACK_SETS + "train.p"
        else:
            raise ValueError("Dataset must be 'test' or 'train'")
        
        with torch.no_grad():
            with open(file_path, "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    
                    output = output.to(self.device)
                    prediction = prediction.to(self.device)
                    members = members.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Assume 'output' already represents a probability distribution.
                    original = output  # Original PVs

                    # Compute perturbed output using the same procedure as in test().
                    member_mask = (members == 1)
                    non_member_mask = (members == 0)
                    
                    if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                        member_indices = member_mask.nonzero(as_tuple=True)[0]
                        non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]
                        
                        member_pvs = output[member_indices]
                        non_member_pvs = output[non_member_indices]
                        
                        # Overlap detection via cosine similarity.
                        cos_sim = F.cosine_similarity(
                            non_member_pvs.unsqueeze(1),  # (n_non_members, 1, C)
                            member_pvs.unsqueeze(0),      # (1, n_members, C)
                            dim=2
                        )
                        max_cos_sim, _ = cos_sim.max(dim=1)  # (n_non_members,)
                        
                        # Gumbel–Softmax binary selection.
                        temperature = 10.0
                        tau = 0.5
                        cosine_threshold = torch.sigmoid(self.cosine_threshold)
                        logits = (max_cos_sim - cosine_threshold) * temperature
                        binary_logits = torch.stack([-logits, logits], dim=1)
                        gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                        binary_selection = gumbel_selection[:, 1]  # (n_non_members,)
                        
                        alpha = 1.0
                        learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                        tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values
                        
                        # Entropy filtering.
                        epsilon = 1e-10
                        entropy_vals = -torch.sum(tentative_perturbed * torch.log(tentative_perturbed + epsilon), dim=1)
                        quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                        quantile_val = torch.quantile(entropy_vals, quantile_threshold)
                        k = self.k if hasattr(self, "k") else 50.0
                        entropy_mask = torch.sigmoid((entropy_vals - quantile_val) * k)
                        
                        final_selection = binary_selection * entropy_mask
                        perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values
                        
                        # Construct perturbed output.
                        perturbed = output.clone()
                        perturbed[non_member_indices] = perturbed_non_member_pvs
                    else:
                        perturbed = output.clone()
                    
                    # Ensure perturbed output is normalized.
                    perturbed = torch.clamp(perturbed, min=1e-6, max=1)
                    perturbed = perturbed / perturbed.sum(dim=1, keepdim=True)
                    
                    # Now, compute cosine similarity for each sample between original and perturbed.
                    # This computes the cosine similarity for each row (sample).
                    sim = F.cosine_similarity(original, perturbed, dim=1)
                    
                    # Append the similarity values separately for members and non-members.
                    member_sim_list.append(sim[member_mask].cpu())
                    non_member_sim_list.append(sim[non_member_mask].cpu())
        

        
        if member_sim_list:
            member_similarities = torch.cat(member_sim_list, dim=0).numpy()
        else:
            member_similarities = np.array([])
        print(f"size: {member_similarities.size}")
        if non_member_sim_list:
            non_member_similarities = torch.cat(non_member_sim_list, dim=0).numpy()
        else:
            non_member_similarities = np.array([])

        print(f"First few member similarities: {member_similarities[:5]}")
        print(f"First few non-member similarities: {non_member_similarities[:5]}")
        exit()
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,6))
            if member_similarities.size > 0:
                plt.hist(member_similarities, bins=50, alpha=0.5, label="Members (Unperturbed)", color="blue")
            if non_member_similarities.size > 0:
                plt.hist(non_member_similarities, bins=50, alpha=0.5, label="Non-Members (Perturbed)", color="orange")
            plt.xlabel("Cosine Similarity (Original vs. Perturbed)")
            plt.ylabel("Frequency")
            plt.title("Cosine Similarity Distribution Before and After Perturbation")
            plt.legend()
            if save_path is not None:
                plt.savefig(save_path)
            plt.show()

        return member_similarities, non_member_similarities

            
    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)
    
    def save_pertub_Model(self, path):
        torch.save(self.perturb_model.state_dict(), self.Perturb_MODELS_PATH)
    # chch
    def save_att_per_thresholds_models(self, last_checkpoint, path):

        checkpoint = torch.load(last_checkpoint, weights_only=True)
        self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        self.cosine_threshold = checkpoint['cosine_threshold']
        self.Entropy_quantile_threshold = checkpoint['entropy_threshold']
        
        models_threshold_params = {
            'attack_model_state_dict': self.attack_model.state_dict(),
            'perturb_model_state_dict': self.perturb_model.state_dict(),
            'cosine_threshold': self.cosine_threshold,
            'Entropy_quantile_threshold': self.Entropy_quantile_threshold
        }
        torch.save(models_threshold_params, path)
        
    def load_perturb_model(self):

        # gan_path = self.Perturb_MODELS_PATH
        # generator = Generator(input_dim).to(device)
        # self.perturb_model = perturb_model.to(self.device)

        self.perturb_model.load_state_dict(torch.load(self.Perturb_MODELS_PATH, weights_only=True))
        self.perturb_model.eval()  # Set the generator to evaluation mode
        return self.perturb_model

  
    def visualize_transformed_pvs_classwise(self, target_class, atk_model, prt_model, consin_thr, entrp_thr, sub_folder):
        # Load the perturb model first from the perturb model path.
        # self.perturb_model = self.load_perturb_model()
        

      
            
        # exit()

          # Load the checkpoint and restore models and thresholds.
        # checkpoint = torch.load(models_path, map_location=self.device, weights_only=True)
        # self.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        # self.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        # self.cosine_threshold.data = checkpoint['cosine_threshold'].to(self.device)
        # self.Entropy_quantile_threshold.data = checkpoint['Entropy_quantile_threshold'].to(self.device)

        self.attack_model = atk_model
        self.perturb_model = prt_model
        self.cosine_threshold = torch.tensor(consin_thr, device=self.device)
        self.Entropy_quantile_threshold = torch.tensor(entrp_thr, device=self.device)

        self.perturb_model.eval()
        self.attack_model.eval()
        # Set professional style.
        sns.set_context("paper", font_scale=1.5)
        sns.set_style("whitegrid")

        # Storage for PVs, their target class labels, membership flags, and perturbation flags.
        original_pvs = []
        perturbed_pvs = []
        class_labels = []       # Target class for each sample.
        membership_flags = []   # 1 for member, 0 for non-member.
        perturb_flags = []      # For non-members: True if perturbed, False otherwise.

        with open(self.ATTACK_SETS + "test.p", "rb") as f:
            while True:
                try:
                    # Load batch data.
                    output, prediction, members, targets = pickle.load(f)
                except EOFError:
                    break  # End of file reached.

                # Move tensors to device.
                output = output.to(self.device)
                prediction = prediction.to(self.device)
                members = members.to(self.device)
                targets = targets.to(self.device)

                # Initialize a per-batch perturbation flag (False by default).
                batch_perturb_flags = torch.zeros_like(members, dtype=torch.bool)
                # Make a copy for the perturbed output.
                perturbed_output = output.clone()

                # Create masks for members and non-members.
                member_mask = (members == 1)
                non_member_mask = (members == 0)

                if member_mask.sum() > 0 and non_member_mask.sum() > 0:
                    # Get indices for members and non-members.
                    member_indices = member_mask.nonzero(as_tuple=True)[0]
                    non_member_indices = non_member_mask.nonzero(as_tuple=True)[0]

                    # Extract corresponding probability vectors (PVs).
                    member_pvs = output[member_indices]       # (n_members, C)
                    non_member_pvs = output[non_member_indices] # (n_non_members, C)

                    # ----- Step 2: Overlap Detection via Cosine Similarity -----
                    # Compute cosine similarity between each non-member and each member.
                    cos_sim = F.cosine_similarity(
                        non_member_pvs.unsqueeze(1),  # (n_non_members, 1, C)
                        member_pvs.unsqueeze(0),      # (1, n_members, C)
                        dim=2
                    )
                    # For each non-member, take the maximum cosine similarity with any member.
                    max_cos_sim, _ = cos_sim.max(dim=1)  # (n_non_members,)

                    # ----- Step 3: Differentiable Binary Selection using Gumbel–Softmax -----
                    temperature = 10.0  # scaling factor for logits
                    tau = 0.5           # Gumbel–Softmax temperature (lower -> sharper)
                    # Reparameterize the cosine threshold so it lies in (0,1).
                    

                    cosine_threshold = torch.sigmoid(self.cosine_threshold)
                    logits = (max_cos_sim - cosine_threshold) * temperature  # (n_non_members,)
                    binary_logits = torch.stack([-logits, logits], dim=1)      # (n_non_members, 2)
                    gumbel_selection = F.gumbel_softmax(binary_logits, tau=tau, hard=True)
                    binary_selection = gumbel_selection[:, 1]  # (n_non_members,)

                    alpha = 1.0

                    # ----- Step 4: Perturbation with Entropy Filtering -----
                    # Compute the learned perturbations for all non-member PVs.
                    # (For visualization we use no_grad(), so gradients are not tracked.)
                    learned_values = self.perturb_model(non_member_pvs, targets[non_member_indices])
                    # Compute a tentative perturbed output.
                    tentative_perturbed = non_member_pvs + alpha * binary_selection.unsqueeze(1) * learned_values

                    # Compute the entropy of each tentative perturbed non-member PV.
                    epsilon = 1e-10
                    entropy = - (tentative_perturbed * torch.log(tentative_perturbed + epsilon)).sum(dim=1).to(self.device)  # (n_non_members,)

                    # Select top samples based on entropy.
                    # For example, use a quantile threshold (self.Entropy_quantile_threshold should be a float in (0,1)).
                    # quantile_val = torch.quantile(entropy, 0.25)
                    quantile_threshold = torch.sigmoid(self.Entropy_quantile_threshold)
                    quantile_val = torch.quantile(entropy, quantile_threshold)

                    # Use a sigmoid to generate a soft entropy mask.
                    # self.k controls the steepness; if not defined, default to 50.0.
                    # k = self.k if hasattr(self, "k") else 50.0
                    entropy_mask = torch.sigmoid((entropy - quantile_val) * self.k)
                    # Combine the binary selection with the entropy mask.
                    final_selection = binary_selection * entropy_mask
                    # Compute final perturbed non-member PVs.
                    perturbed_non_member_pvs = non_member_pvs + alpha * final_selection.unsqueeze(1) * learned_values

                    # Replace the non-member PVs in the overall output with the final perturbed versions.
                    perturbed_output[non_member_indices] = perturbed_non_member_pvs
                    # Mark these non-members as perturbed if final_selection > 0.5.
                    batch_perturb_flags[non_member_indices] = (final_selection > 0.5)
                else:
                    perturbed_output = output.clone()

                # Append data from this batch.
                original_pvs.append(output.detach().cpu().numpy())
                perturbed_pvs.append(perturbed_output.detach().cpu().numpy())
                class_labels.append(targets.detach().cpu().numpy())
                membership_flags.append(members.detach().cpu().numpy())
                perturb_flags.append(batch_perturb_flags.detach().cpu().numpy())

        # Convert lists to arrays.
        if not (original_pvs and perturbed_pvs and class_labels and membership_flags and perturb_flags):
            print("No PVs selected for visualization.")
            return

        original_pvs = np.vstack(original_pvs)        # (total_samples, C)
        perturbed_pvs = np.vstack(perturbed_pvs)        # (total_samples, C)
        class_labels = np.hstack(class_labels)          # (total_samples,)
        membership_flags = np.hstack(membership_flags)  # (total_samples,)
        perturb_flags = np.hstack(perturb_flags)        # (total_samples,)

        # ----- Filter to Only Include the Target Class -----
        target_mask = (class_labels == target_class)
        if np.sum(target_mask) == 0:
            print(f"No samples found for target class {target_class}.")
            return
        original_pvs = original_pvs[target_mask]
        perturbed_pvs = perturbed_pvs[target_mask]
        class_labels = class_labels[target_mask]
        membership_flags = membership_flags[target_mask]
        perturb_flags = perturb_flags[target_mask]

        # ----- t-SNE Dimensionality Reduction -----
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        # Stack original and perturbed PVs.
        all_pvs = np.vstack([original_pvs, perturbed_pvs])
        reduced_pvs = reducer.fit_transform(all_pvs)

        num_samples = original_pvs.shape[0]
        original_pvs_2d = reduced_pvs[:num_samples]
        perturbed_pvs_2d = reduced_pvs[num_samples:]

        # Define colors and markers.
        palette = {
            "Members": "#1f77b4",      # Blue
            "Non-Members": "#ff7f0e",  # Orange
        }

        # ----- Plotting with Seaborn -----
       # --- Create subplots ---
       # --- Global Plot Settings with Arial Font ---
        size = 20
        params = {
            'axes.labelsize': size,
            'font.size': size,
            'legend.fontsize': size,
            'xtick.labelsize': size,
            'ytick.labelsize': size,
            'figure.figsize': [16, 8],
            "font.family": "arial",
        }
        plt.rcParams.update(params)

        # --- Create Two Subplots ---
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        # Left subplot: Before Perturbation
        sns.scatterplot(
            x=original_pvs_2d[:, 0], y=original_pvs_2d[:, 1],
            hue=np.where(membership_flags == 1, "Members", "Non-Members"),
            palette=palette, alpha=0.7, s=100, edgecolor='black',
            ax=ax[0]
        )
        ax[0].set_title("Before", fontsize=20, fontweight='bold')
        ax[0].set_xlabel("t-SNE Component 1", fontsize=20)
        ax[0].set_ylabel("t-SNE Component 2", fontsize=20)
        ax[0].grid(True, linestyle="--", alpha=0.5)
        # Remove local legend
        if ax[0].get_legend():
            ax[0].get_legend().remove()

        # Right subplot: After Perturbation
        sns.scatterplot(
            x=perturbed_pvs_2d[:, 0], y=perturbed_pvs_2d[:, 1],
            hue=np.where(membership_flags == 1, "Members", "Non-Members"),
            palette=palette, alpha=0.7, s=100, edgecolor='black',
            ax=ax[1]
        )
        ax[1].set_title("After", fontsize=20, fontweight='bold')
        ax[1].set_xlabel("t-SNE Component 1", fontsize=20)
        # ax[1].set_ylabel("t-SNE Component 2", fontsize=20)
        ax[1].grid(True, linestyle="--", alpha=0.5)

        # Overlay a highlight for perturbed non-members on the "After" subplot.
        non_member_mask = (membership_flags != 1) & (perturb_flags == 1)
        ax[1].scatter(
            perturbed_pvs_2d[non_member_mask, 0],
            perturbed_pvs_2d[non_member_mask, 1],
            facecolors='#ff7f0e',    # no fill
            edgecolors='red',     # red outline
            s=100,                # larger markers
            marker='o',
            label='Perturbed Non-Members'
        )
        # Remove local legend if exists
        if ax[1].get_legend():
            ax[1].get_legend().remove()

        # --- Combine Legend Entries from Both Subplots ---
        handles_left, labels_left = ax[0].get_legend_handles_labels()
        handles_right, labels_right = ax[1].get_legend_handles_labels()
        combined_handles = []
        combined_labels = []
        for h, l in zip(handles_left + handles_right, labels_left + labels_right):
            if l not in combined_labels:
                combined_handles.append(h)
                combined_labels.append(l)

        # --- Create a Global Legend Below the X-Axis ---
        fig.legend(
            combined_handles, combined_labels,
            loc='lower center',
            ncol=len(combined_labels),
            fontsize=20,
            bbox_to_anchor=(0.5, 0.02)  # adjust the y-value as needed
        )

        # Adjust layout to make room for the legend
        # plt.tight_layout(rect=[0, 0.1, 1, 1])

        # Adjust layout to make room for the global legend.
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        # plt.tight_layout()

        # plt.suptitle(f"t-SNE Visualization of PVs for Target Class {target_class}\nMembers and Non-Members", 
                    # fontsize=16, fontweight='bold')
        # plt.show()

        # 9. Save the plot in sub_folder with a name referencing the class label.
        plot_filename = f"class_{target_class}.pdf"
        save_path_full = os.path.join(sub_folder, plot_filename)
        # plt.savefig(save_path_full)
        plt.savefig(save_path_full, dpi=300, format='pdf')
        plt.close()
        # plt.show()
    

    # lira
    def liRA_offline(self):
        
        
        

        # Question? does the loaders have both members and non-members
        #  yes, 
        # self.attack_train_loader # Contains raw samples (made loader) used to train the target model
        # self.attack_test_loader #  Contains raw samples (made loader) used to test the target model,
        # also called target samples, these are used to obtain test PVs to test attack model
        
        # here the test.p contains conf_ob: confobs = (f(x)y)
        # train is the confs of target model that did see target point (x,y), in this case its in test_loader
        from scipy.stats import norm

        outputs_list = []
        members_list = []
        targets_list = []

        # Load data from the saved file (train.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "train.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_list.append(output.cpu())    # output: [batch, num_classes]
                    members_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs = torch.cat(outputs_list, dim=0)   # shape: (total_samples, num_classes)
        all_members = torch.cat(members_list, dim=0)     # shape: (total_samples,)
        all_targets = torch.cat(targets_list, dim=0)     # shape: (total_samples,)

        
       
        out_signals = all_outputs     # non-members (out)
        
        
        mean_out = np.median(out_signals.numpy(), 1).reshape(-1, 1)

       
        std_out = np.std(out_signals.numpy(), 1).reshape(-1, 1)

        print("Estimated distribution parameters:")
        # print("Mean In-Signal:", mean_in)
        print("Mean Out-Signal:", mean_out)
        # print("Std In-Signal:", std_in)
        print("Std Out-Signal:", std_out)
        # exit()
        # Now, for each sample, compute the negative log-likelihood under the two distributions.
        # Here, sc (signal observed) is the correct confidence for each sample.
        
        outputs_test_list = []
        members_test_list = []
        targets_test_list = []

        # Load data from the saved file (test.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_test_list.append(output.cpu())    # output: [batch, num_classes]
                    members_test_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_test_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs_test = torch.cat(outputs_test_list, dim=0)   # shape: (total_samples, num_classes)
        all_members_test = torch.cat(members_test_list, dim=0)     # shape: (total_samples,)
        all_targets_test = torch.cat(targets_test_list, dim=0)     # shape: (total_samples,)

        # print("Loaded data from test.p:")
        # print(f"Outputs shape: {all_outputs_test.shape}")
        # print(f"Members shape: {all_members_test.shape}")
        # print(f"Targets shape: {all_targets_test.shape}")

        # exit()
        sc = all_outputs_test

        # mean_out = np.median(out_signals, 1).reshape(-1, 1)
        
        
        # std_out = np.std(out_signals, 1).reshape(-1, 1)

        # If running in "offline" mode, you might choose to ignore the in-part
        # (i.e., set pr_in=0) and only use the out-distribution.
        

        # print(f"Shape of all_outputs_test: {all_outputs_test.shape}")
        # print(f"Shape of mean_out: {mean_out.shape}")
        # print(f"Shape of std_out: {std_out.shape}")
        # exit()
        prediction = []
        answers = []

      
        pr_in = 0
        
        pr_out = -norm.logpdf(all_outputs_test, mean_out, std_out + 1e-30) # gaussian approximation
        score = pr_in - pr_out

        prediction = np.array(score.mean(1))

        # prediction_2 = np.array(score.mean(1))
        prediction_2 = np.where(-score.mean(1) >= 0, 1, 0)
        # For each sample, the membership score is given by 'score'.
        # (You can decide on a threshold—for example, if score < 0, predict member.)
        # print("First 10 LiRA scores:", score[:10])
        # print(f"size of score: {score.shape}")
        # print("Predictions (first 10):", prediction[:10])
        # print("Predictions size:", prediction.shape)
        # # Check if all prediction values are probabilities and print
        
        # # Check if all prediction values are less than 1
        # print(f"Max of prediction: {np.max(prediction)}")
        # # Print a few predictions to compare
        # print("First 10 Predictions (Threshold Applied):", prediction_2[:10])
        # print("First 10 Predictions (Raw Scores):", prediction[:10])
        # print(f"Max of prediction_2: {np.max(prediction_2)}")
        # print(f"Min of prediction_2: {np.min(prediction_2)}")

        # correct = predicted.eq(all_members_test).sum().item()
        answers = np.array(all_members_test.reshape(-1, 1), dtype=bool)
        fpr_list, tpr_list, thresholds = roc_curve(answers.ravel(), (-prediction).ravel())
        auc_score = auc(fpr_list, tpr_list)
        # bcm = BinaryConfusionMatrix().to(self.device)
        # conf_mat = bcm((-prediction).ravel(), all_members_test.ravel())
        

       

        acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
        
        recall = np.max(tpr_list)
        print(f"acc: {acc}, recall: {recall:.3f}")
        # prec = np.sum((prediction > 0) & (answers > 0)) / np.sum(prediction > 0)
        # recall = np.sum((prediction > 0) & (answers > 0)) / np.sum(answers > 0)
        # print(f"Precision: {prec:.3f}, Recall: {recall:.3f}")
        # print(f"Precision: {prec:.3f}, Recall: {recall:.3f}")
        # print("First few FPR values:", fpr_list[:10])
        # print("First few TPR values:", tpr_list[:10])
        
          
        # Plot the ROC curve
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr_list, tpr_list, label="ROC curve (AUC = %0.2f)" % auc(fpr_list, tpr_list), lw=2, color='blue')
        # plt.plot([0, 1], [0, 1], 'k--', lw=2, color='red')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("False Positive Rate", fontsize=14)
        # plt.ylabel("True Positive Rate", fontsize=14)
        # plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight="bold")
        # plt.legend(loc="lower right", fontsize=12)
        # plt.grid(True, linestyle="--", alpha=0.5)
        # plt.tight_layout()
        # plt.show()
        # print(f"Correct predictions: {correct}")
        # exit()
        
        return fpr_list, tpr_list, thresholds, auc_score


    def liRA_offline_mul(self):
        
        
        # load N target models (closely mimicking shadow models) trained already
        # the output of these trained models is outsignals
        # for each loaded target model, get their corresponding training dataset
        # use its corresponding dataset to generate outsignals 
        # concatinate for all N models, 
        # for each dataset the N will be different 
        # for testing we will target model trained on the entire dataset

        

        # Question? does the loaders have both members and non-members
        #  yes, 
        # self.attack_train_loader # Contains raw samples (made loader) used to train the target model
        # self.attack_test_loader #  Contains raw samples (made loader) used to test the target model,
        # also called target samples, these are used to obtain test PVs to test attack model
        
        # here the test.p contains conf_ob: confobs = (f(x)y)
        # train is the confs of target model that did see target point (x,y), in this case its in test_loader
        from scipy.stats import norm


        outputs_list = []
        members_list = []
        targets_list = []

        # Load data from the saved file (train.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "train.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_list.append(output.cpu())    # output: [batch, num_classes]
                    members_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs = torch.cat(outputs_list, dim=0)   # shape: (total_samples, num_classes)
        all_members = torch.cat(members_list, dim=0)     # shape: (total_samples,)
        all_targets = torch.cat(targets_list, dim=0)     # shape: (total_samples,)

        
       
        out_signals = all_outputs     # non-members (out)
        
        
        mean_out = np.median(out_signals.numpy(), 1).reshape(-1, 1)

       
        std_out = np.std(out_signals.numpy(), 1).reshape(-1, 1)

        print("Estimated distribution parameters:")
        # print("Mean In-Signal:", mean_in)
        print("Mean Out-Signal:", mean_out)
        # print("Std In-Signal:", std_in)
        print("Std Out-Signal:", std_out)
        # exit()
        # Now, for each sample, compute the negative log-likelihood under the two distributions.
        # Here, sc (signal observed) is the correct confidence for each sample.
        
        outputs_test_list = []
        members_test_list = []
        targets_test_list = []

        # Load data from the saved file (test.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_test_list.append(output.cpu())    # output: [batch, num_classes]
                    members_test_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_test_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs_test = torch.cat(outputs_test_list, dim=0)   # shape: (total_samples, num_classes)
        all_members_test = torch.cat(members_test_list, dim=0)     # shape: (total_samples,)
        all_targets_test = torch.cat(targets_test_list, dim=0)     # shape: (total_samples,)

        print("Loaded data from test.p:")
        print(f"Outputs shape: {all_outputs_test.shape}")
        print(f"Members shape: {all_members_test.shape}")
        print(f"Targets shape: {all_targets_test.shape}")

        # exit()
        sc = all_outputs_test

        # mean_out = np.median(out_signals, 1).reshape(-1, 1)
        
        
        # std_out = np.std(out_signals, 1).reshape(-1, 1)

        # If running in "offline" mode, you might choose to ignore the in-part
        # (i.e., set pr_in=0) and only use the out-distribution.
        

        prediction = []
        answers = []

      
        pr_in = 0
        
        pr_out = -norm.logpdf(all_outputs_test, mean_out, std_out + 1e-30) # gaussian approximation
        score = pr_in - pr_out

        prediction = np.array(score.mean(1))

        # prediction_2 = np.array(score.mean(1))
        prediction_2 = np.where(-score.mean(1) >= 0, 1, 0)
        # For each sample, the membership score is given by 'score'.
        # (You can decide on a threshold—for example, if score < 0, predict member.)
        # print("First 10 LiRA scores:", score[:10])
        # print(f"size of score: {score.shape}")
        # print("Predictions (first 10):", prediction[:10])
        # print("Predictions size:", prediction.shape)
        # # Check if all prediction values are probabilities and print
        
        # # Check if all prediction values are less than 1
        # print(f"Max of prediction: {np.max(prediction)}")
        # # Print a few predictions to compare
        # print("First 10 Predictions (Threshold Applied):", prediction_2[:10])
        # print("First 10 Predictions (Raw Scores):", prediction[:10])
        # print(f"Max of prediction_2: {np.max(prediction_2)}")
        # print(f"Min of prediction_2: {np.min(prediction_2)}")

        # correct = predicted.eq(all_members_test).sum().item()
        answers = np.array(all_members_test.reshape(-1, 1), dtype=bool)
        fpr_list, tpr_list, betas = roc_curve(answers.ravel(), (-prediction).ravel())
        # bcm = BinaryConfusionMatrix().to(self.device)
        # conf_mat = bcm((-prediction).ravel(), all_members_test.ravel())
        

        
        acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
        
        recall = np.max(tpr_list)
        print(f"acc: {acc}, recall: {recall:.3f}")
        # prec = np.sum((prediction > 0) & (answers > 0)) / np.sum(prediction > 0)
        # recall = np.sum((prediction > 0) & (answers > 0)) / np.sum(answers > 0)
        # print(f"Precision: {prec:.3f}, Recall: {recall:.3f}")
        # print(f"Precision: {prec:.3f}, Recall: {recall:.3f}")
        # print("First few FPR values:", fpr_list[:10])
        # print("First few TPR values:", tpr_list[:10])
        
          
        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_list, tpr_list, label="ROC curve (AUC = %0.2f)" % auc(fpr_list, tpr_list), lw=2, color='blue')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, color='red')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight="bold")
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
        # print(f"Correct predictions: {correct}")
        exit()
    
    def enh_mia(self):
       
        # Attack-R with linear interpolation (Ye et al.) https://arxiv.org/pdf/2111.09679.pdf
        # Taken and repadated from https://github.com/yuan74/ml_privacy_meter/blob/2022_enhanced_mia/research/2022_enhanced_mia/plot_attack_via_reference_or_distill.py
        # if len(target_signal.shape) == 2:
        #     sc = target_signal[target_indices,0].reshape(-1, 1) # 50k x 1 , no augmentation
        #     out_signals = out_signals[target_indices,:,0]
        # else:


        outputs_test_list = []
        members_test_list = []
        targets_test_list = []

        # Load data from the saved file (test.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_test_list.append(output.cpu())    # output: [batch, num_classes]
                    members_test_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_test_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs_test = torch.cat(outputs_test_list, dim=0)   # shape: (total_samples, num_classes)
        all_members_test = torch.cat(members_test_list, dim=0)     # shape: (total_samples,)
        all_targets_test = torch.cat(targets_test_list, dim=0)     # shape: (total_samples,)

        print("Loaded data from test.p:")
        print(f"Outputs shape: {all_outputs_test.shape}")
        print(f"Members shape: {all_members_test.shape}")
        print(f"Targets shape: {all_targets_test.shape}")

        # exit()
        sc = all_outputs_test
        
        outputs_list = []
        members_list = []
        targets_list = []

        # Load data from the saved file (train.p)
        with torch.no_grad():
            with open(self.ATTACK_SETS + "train.p", "rb") as f:
                while True:
                    try:
                        output, prediction, members, targets = pickle.load(f)
                    except EOFError:
                        break
                    outputs_list.append(output.cpu())    # output: [batch, num_classes]
                    members_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
                    targets_list.append(targets.cpu())     # true labels for each sample

        # Concatenate batches to get one tensor per item.
        all_outputs = torch.cat(outputs_list, dim=0)   # shape: (total_samples, num_classes)
        all_members = torch.cat(members_list, dim=0)     # shape: (total_samples,)
        all_targets = torch.cat(targets_list, dim=0)     # shape: (total_samples,)

        #
        # Now split the signals by membership.
        # Here we assume that 'members' is 1 for training (member) and 0 for non-member.
        # in_signals = correct_confidences[all_members.bool()]      # members (in)
        out_signals = all_outputs     # non-members (out)


        # sc = target_signal[target_indices] # 50k x 1
        
        def from_correct_logit_to_loss(array): # convert correct logit to the cross entropy loss
            return np.log((1+np.exp(array))/np.exp(array)) # positive
        
        losses = from_correct_logit_to_loss(out_signals).T.numpy() # shape nb_models x nb_target, ref lossses
        check_losses = from_correct_logit_to_loss(sc).T.numpy() # shape nb_target x 1, target losses

        
        dummy_min = np.zeros((1, len(losses[0]))) # shape 1 x nb_target

        dummy_max = dummy_min + 1000 # shape 1 x nb_target

        dat_reference_or_distill = np.sort(np.concatenate((losses, dummy_max, dummy_min), axis=0), axis=0) # shape nb_models + 2 x nb_target 

        prediction = np.array([])
        
        discrete_alpha = np.linspace(0, 1, len(dat_reference_or_distill))
        for i in range(len(dat_reference_or_distill[0])):
            losses_i =  dat_reference_or_distill[:, i]

            # Create the interpolator
            pr = np.interp(check_losses[0,i], losses_i, discrete_alpha)
            
            prediction = np.append(prediction, pr)

       

        answers = np.array(all_members_test.reshape(-1, 1), dtype=bool)
        # answers = np.array(all_members_test.reshape(-1, 1), dtype=bool)
        fpr_list, tpr_list, betas = roc_curve(answers.ravel(), (prediction).ravel())

         # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_list, tpr_list, label="ROC curve (AUC = %0.2f)" % auc(fpr_list, tpr_list), lw=2, color='blue')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, color='red')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight="bold")
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

        print("OKAY")
        exit()



    def metric_results(fpr_list, tpr_list, thresholds):
        fprs = [0.01,0.001,0.0001,0.00001,0.0] # 1%, 0.1%, 0.01%, 0.001%, 0%
        tpr_dict = {}
        thresholds_dict = {}
        acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
        roc_auc = auc(fpr_list, tpr_list)

        for fpr in fprs:
            tpr_dict[fpr] = tpr_list[np.where(fpr_list <= fpr)[0][-1]] # tpr at fpr
            thresholds_dict[fpr] = thresholds[np.where(fpr_list <= fpr)[0][-1]] # corresponding threshold

        return roc_auc, acc, tpr_dict, thresholds_dict

    def get_attack_dataset_without_shadow(train_set, test_set, batch_size):
        mem_length = int(len(train_set)*0.45)
        nonmem_length = int(len(test_set)*0.45)
        mem_train, mem_test, _ = torch.utils.data.random_split(train_set, [mem_length, mem_length, len(train_set)-(mem_length*2)])
        nonmem_train, nonmem_test, _ = torch.utils.data.random_split(test_set, [nonmem_length, nonmem_length, len(test_set)-(nonmem_length*2)])
        mem_train, mem_test, nonmem_train, nonmem_test = list(mem_train), list(mem_test), list(nonmem_train), list(nonmem_test)

        for i in range(len(mem_train)):
            mem_train[i] = mem_train[i] + (1,)
        for i in range(len(nonmem_train)):
            nonmem_train[i] = nonmem_train[i] + (0,)
        for i in range(len(nonmem_test)):
            nonmem_test[i] = nonmem_test[i] + (0,)
        for i in range(len(mem_test)):
            mem_test[i] = mem_test[i] + (1,)
            
        attack_train = mem_train + nonmem_train
        attack_test = mem_test + nonmem_test

        attack_trainloader = torch.utils.data.DataLoader(
            attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
        attack_testloader = torch.utils.data.DataLoader(
            attack_test, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)

        return attack_trainloader, attack_testloader


def get_attack_dataset_with_shadow(target_train, target_test,  batch_size):
    # mem_train, nonmem_train, mem_test, nonmem_test = list(shadow_train), list(shadow_test), list(target_train), list(target_test)

    mem_train = list(target_train)
    nonmem_test = list(target_test)

    # for i in range(len(mem_train)):
    #     mem_train[i] = mem_train[i] + (1,)
    # for i in range(len(nonmem_train)):
    #     nonmem_train[i] = nonmem_train[i] + (0,)
    # for i in range(len(nonmem_test)):
    #     nonmem_test[i] = nonmem_test[i] + (0,)
    # for i in range(len(mem_test)):
    #     mem_test[i] = mem_test[i] + (1,)

    

    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test), len(nonmem_test))

    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
    mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
    non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
    
    attack_train = mem_train + non_mem_train
    attack_test = mem_test + non_mem_test

    # attack_trainloader = torch.utils.data.DataLoader(
    #     attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
    # attack_testloader = torch.utils.data.DataLoader(
    #     attack_test, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)


    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)


    attack_train = mem_train
    attack_test = nonmem_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)

    return attack_trainloader, attack_testloader


# Combine the data from the shdaow and target model outputs to get attack_train and attack_test
# perform some statistical analysis on the target model outputs for the training samples and testing samples (can also be from shadow data)
# the anaylysis must be performed by selecting a dataset for which the target model is not generalizing well, meaning it is not overfitted
# we can also analyse the it with highly over fitted version for the same data and see the differnce between pvs for the same data once overfitted and once not overiftteed

def enahanced_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, batch_size):
    mem_train, nonmem_train, mem_test, nonmem_test = list(shadow_train), list(shadow_test), list(target_train), list(target_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)


    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test), len(nonmem_test))

    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
    mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
    non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
    
    attack_train = mem_train + non_mem_train
    attack_test = mem_test + non_mem_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)

    return attack_trainloader, attack_testloader

def dataloader_to_dataset(dataloader):
    data_list = []
    
    for batch in dataloader:
        data, labels = batch  # assuming the data is returned as (data, labels)
        data_list.append(data)
    
    # Concatenate all the batches into one dataset
    full_data = torch.cat(data_list, dim=0)
    
    return full_data

# def save_best_checkpoint(val_loss, attack_model, perturb_model, cosine_threshold, entropy_threshold, checkpoint_path='checkpoint.pt'):
#     """
#     Saves a checkpoint containing the best attack model, perturb model, and corresponding thresholds.
#     """
#     checkpoint = {
#         'val_loss': val_loss,
#         'attack_model_state_dict': attack_model.state_dict(),
#         'perturb_model_state_dict': perturb_model.state_dict(),
#         'cosine_threshold': cosine_threshold,
#         'entropy_threshold': entropy_threshold
#     }
#     torch.save(checkpoint, checkpoint_path)
#     print("Checkpoint saved to", checkpoint_path)

def save_best_checkpoint(val_loss, attack_model, perturb_model, cosine_threshold, entropy_threshold, checkpoint_path):
    checkpoint = {
        'val_loss': val_loss,
        'attack_model_state_dict': attack_model.state_dict(),
        'perturb_model_state_dict': perturb_model.state_dict(),
        'cosine_threshold': cosine_threshold,
        'entropy_threshold': entropy_threshold
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# Combined attack
# def attack_mode0_com(TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, perturb_model, get_attack_set, num_classes, mode):
def attack_mode0_com(TARGET_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, perturb_model, num_classes, mode, dataset_name, attack_name, entropy_dis_dr, apcmia_cluster, arch, acc_gap):
    MODELS_PATH = ATTACK_PATH + "_meminf_"+attack_name+"_.pth"
    Perturb_MODELS_PATH = ATTACK_PATH + "_perturb_model.pth"

    RESULT_PATH = ATTACK_PATH + "_meminf_attack0_com.p"
    RESULT_PATH_csv = ATTACK_PATH + "_meminf_attack0_com.csv"
    
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode0__com"
    ATTACK_SETS_PV_CSV = ATTACK_PATH + "_meminf_attack_pvs.csv"

    fpr_tpr_file_path = ATTACK_PATH + "_FPR_TPR_" + attack_name + "_.csv"

    # from datetime import datetime
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # fpr_tpr_file_path = ATTACK_PATH + f"_FPR_TPR_{attack_name}_{timestamp}.csv"

    
    MODELS_PATH_att_per_thr = ATTACK_PATH + "_attack_pertubr_thresholds_"+attack_name+".pth" # will store all 

    # MODELS_PATH, RESULT_PATH, ATTACK_PATH
    #! for weak shadow_model change model architecture to just taking the PV int simple NN and 
    
    # ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode0__com"
    # file_path = ATTACK_SETS + f"_Results-Mean_mode-{attack_name}_.csv"

    print(f"MODELS_PATH: {MODELS_PATH}, \nRESULT_PATH: {RESULT_PATH}, \nATTACK_PATH: {ATTACK_SETS}")
    
    epoch_data = []
    train_accuracy_list = []
    test_accuracy_list = []
    res_list = []
    cosine_entropy_threshold_list = []
    
    
        
    attack = attack_for_blackbox_com_NEW(TARGET_PATH, Perturb_MODELS_PATH, ATTACK_SETS,ATTACK_SETS_PV_CSV, attack_trainloader, attack_testloader, target_model, attack_model,perturb_model, device, dataset_name, attack_name, num_classes, acc_gap)
           
    if 1:
        
        get_attack_set = 1;
        if get_attack_set:
            attack.delete_pickle()
            # # output = output.cpu().detach().numpy()
            #         # output size: torch.Size([64, 10]), prediction size: torch.Size([64, 1]), members: torch.Size([64]), batch of 64
            #         # prediction: a specific sample in the batch is predicted correct (1) or predicted wrong (0)
            #         # output: Not PVs but raw 10 logits (based on the number of classes)
            #         # print(f"output size: {output.shape}, prediction size: {prediction.shape}, members: {members.shape}")
            #         # print(output)
            #         # print(prediction)
            #         # print(members)
            #         # break;
            #         pickle.dump((output, prediction, members, targets), f)
                    # target are corresponding class la
            dataset = attack.prepare_dataset() # uses shahow model to first obtain PVs and and the, combines it into [Pv, prediction, members, targets]
            # attack.prepare_dataset_analyse()

        # exit()
        
        epochs = 100
        tr_sum = 0.0;
        ts_sum = 0.0;

        checkpoint_files = []  # List to store unique checkpoint filenames
        res_list = []  # store final test metrics per epoch

        # if attack_name != "lira": # news
            
        threshold_progress = []  # list of tuples (cosine_threshold, entropy_threshold)
        test_loss_progress = []  # list of average test losses
        
        # initialize the early_stopping object
        if dataset_name == "purchase":
            patience = 7
        else:
            patience = 15

        early_stopping = EarlyStopping(patience=patience, verbose=True)

        for ep in range(epochs):
            flag = 1 if ep == (epochs - 1) else 0
            print("Epoch %d:" % (ep + 1))
            # Train for one epoch
            res_train = attack.train(flag, RESULT_PATH, RESULT_PATH_csv, mode)
            # Test for one epoch and get metrics (the last element is avg test loss)
            res_test, fpr, tpr = attack.test(flag, RESULT_PATH, mode)
            # Extract current threshold values (after sigmoid)
            current_cosine_threshold = torch.sigmoid(attack.cosine_threshold).item()
            current_entropy_threshold = torch.sigmoid(attack.Entropy_quantile_threshold).item()
                                    
            res_list.append({'epoch': ep + 1,                                
                            'test_acc': res_test[0]*100,
                            'test_prec': res_test[1]*100,
                            'test_recall': res_test[2]*100,
                            'test_f1': res_test[3]*100,
                            'test_auc': res_test[4]*100,
                            'test_loss': res_test[-1],
                            'cosine_threshold': current_cosine_threshold,
                            'entropy_threshold': current_entropy_threshold})

            # heyyyyy
            early_stopping(res_test[-1], attack.attack_model)

            # Suppose early_stopping has an attribute best_loss that is updated when improvement occurs:
            if res_test[-1] == early_stopping.best_val_loss:
                # Save the best checkpoint with both models and current thresholds.
                # save_best_checkpoint(
                #     res_test[-1],
                #     attack.attack_model,
                #     attack.perturb_model,
                #     current_cosine_threshold,
                #     current_entropy_threshold,
                #     checkpoint_path='checkpoint.pt'
                # )
                checkpoint_path = f'checkpoint_epoch_{ep+1}.pt'
                
                save_best_checkpoint(
                    res_test[-1],
                    attack.attack_model,
                    attack.perturb_model,
                    current_cosine_threshold,
                    current_entropy_threshold,
                    checkpoint_path=checkpoint_path
                )
                checkpoint_files.append(checkpoint_path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        if checkpoint_files:
            last_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=True)
            attack.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
            attack.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
            best_cosine_threshold = checkpoint['cosine_threshold']
            best_entropy_threshold = checkpoint['entropy_threshold']
            print(f"Loaded last checkpoint: {last_checkpoint}")
            # print(f"Best Cosine Threshold: {best_cosine_threshold:.4f}, Best Entropy Threshold: {best_entropy_threshold:.4f}")
            # exit()
            
        # checkpoint = torch.load('checkpoint.pt', weights_only=True)
        # attack.attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
        # attack.perturb_model.load_state_dict(checkpoint['perturb_model_state_dict'])
        # best_cosine_threshold = checkpoint['cosine_threshold']
        # best_entropy_threshold = checkpoint['entropy_threshold']

        print(f"Best Cosine Threshold: {best_cosine_threshold:.4f}, Best Entropy Threshold: {best_entropy_threshold:.4f}")
        # exit()
        # fpr, tpr, thresholds, roc_auc = attack.compute_roc_curve_apcmia(attack.attack_model, attack.perturb_model,best_cosine_threshold,best_entropy_threshold)
        # herererer
        # load the last checkpoint with the best model
        # attack.attack_model.load_state_dict(torch.load('checkpoint.pt', weights_only=True))
        
        # print few of the fpr and tpr and then exit for debugging

        # print(f"FPR: {fpr[:10]}")
        # print(f"TPR: {tpr[:10]}")
        # # print(f"Thresholds: {thresholds[:10]}")

        # exit()
        # Optionally, save the threshold and loss progress to a CSV.
        df = pd.DataFrame(res_list)
        file_path = ATTACK_SETS + f"_Results-Mean_mode-{attack_name}_.csv" 
        df.to_csv(file_path, index=False)
        
        
        #  Save models
        # attack.saveModel(MODELS_PATH)
        # # print("Saved Attack Model")
        # attack.save_pertub_Model(Perturb_MODELS_PATH)
        # print(f"saved pertrub model")

        # attack.test_saved_model(MODELS_PATH_att_per_thr, plot=True, save_path=None)
        # exit()
        # ;;;;;;
        attack.save_att_per_thresholds_models(last_checkpoint, MODELS_PATH_att_per_thr)
        print(f'models and thresholds are savedQ')
        
        # the following function will load the thresholds from MODELS_PATH_att_per_thr
        
        if attack_name == "apcmia":
            print(f'computing ROC for apcmia')
            attack.test_saved_model_apcmia(attack.attack_model, attack.perturb_model,best_cosine_threshold,best_entropy_threshold)
            # and dataset_name != "adult"
            if(dataset_name != "cifar10" and dataset_name != "cifar100" and dataset_name != "stl10" and dataset_name != "purchase" and dataset_name != "texas" and dataset_name != "adult" ):
                fpr, tpr, thresholds, roc_auc = attack.compute_roc_curve_apcmia(attack.attack_model, attack.perturb_model,best_cosine_threshold,best_entropy_threshold)
            # fpr, tpr, thresholds, roc_auc = attack.compute_roc_curve_apcmia(attack.attack_model, attack.perturb_model,best_cosine_threshold,best_entropy_threshold)
            # original_entropies, perturbed_entropies = attack.compute_entropy_distribution(attack.attack_model, attack.perturb_model,best_cosine_threshold,best_entropy_threshold, entropy_dis_dr)
            # attack.compute_entropy_distribution_new(attack.attack_model, attack.perturb_model,best_cosine_threshold,best_entropy_threshold, entropy_dis_dr)
            attack.compute_entropy_distribution_new_norm(attack.attack_model, attack.perturb_model,best_cosine_threshold,best_entropy_threshold, entropy_dis_dr)


        else:
            # attack.test_saved_model_rest(MODELS_PATH_att_per_thr, plot=True, save_path=None)
            attack.test_saved_model_rest(attack.attack_model)
            # exit()
            fpr, tpr, thresholds, roc_auc = attack.compute_roc_curve_rest(attack.attack_model)

        # Save fpr and tpr to a CSV file
        df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        df_fpr_tpr.to_csv(fpr_tpr_file_path, index=False) # roc call won't work now cuz name has added timestamp
        print(f"saved ROC curve info")

        
    # attack.compute_cosine_similarity_distribution(MODELS_PATH_att_per_thr)
    
    # elif attack_name == "lira": # calling LiRA_offline attack
    #     print("LiRA offline attack in process")
    #     fpr, tpr, thresholds, roc_auc = attack.liRA_offline()
    #     df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    #     df_fpr_tpr.to_csv(fpr_tpr_file_path, index=False)
    #     print(f"saved ROC curve info")
    #     # exit()
    # elif mode==5:
    #     attack.enh_mia()
    #     # exit()
        # else:
        #     print("mode is not -1 or -2")
    
    if attack_name == "apcmia" and apcmia_cluster:

          # 1. Create the root directory "cluster_results" if it doesn't exist.
        cluster_root = f"cluster_results/{arch}/"
        if not os.path.exists(cluster_root):
            os.makedirs(cluster_root)

        # 2. Create a subdirectory for this dataset (e.g., "test") inside "cluster_results".
        sub_folder = os.path.join(cluster_root, dataset_name)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        for target_class in range(num_classes):
            attack.visualize_transformed_pvs_classwise(target_class, attack.attack_model, attack.perturb_model, best_cosine_threshold, best_entropy_threshold, sub_folder)
            # exit()
    # return res_train, res_test

