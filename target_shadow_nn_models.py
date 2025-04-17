import os
import torch
import pandas
import torchvision
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.autograd import Variable
from torch.utils.data import random_split, ConcatDataset
from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple
torch.manual_seed(0)
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

   
    
class simpleNN(nn.Module):
    def __init__(self, input_size, num_classes=30):
        super(simpleNN, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(128, num_classes),
        )


    def forward(self, X_Batch):
        x = self.classifier(X_Batch)
        return x



class simpleNN_Target_purchase(nn.Module):
    def __init__(self, input_size, num_classes=30):
        super(simpleNN_Target_purchase, self).__init__()
        
        self.classifier = nn.Sequential(
            
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),      
            nn.Linear(100, num_classes),
            
        )


    def forward(self, X_Batch):
        x = self.classifier(X_Batch)
        return x


class simpleNN_Target_texas(nn.Module):
    def __init__(self, input_size, num_classes=100):
        super(simpleNN_Target_texas, self).__init__()
     
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Continue replacing BatchNorm1d with LayerNorm or GroupNorm as needed
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes)
        )

    def forward(self, X_Batch):
        return self.classifier(X_Batch)
    
class simpleNN_Shaddow_purchase(nn.Module):
    def __init__(self, input_size, num_classes=30):
        super(simpleNN_Shaddow_purchase, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),  # Dropout with probability 50%
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes),
        )


    def forward(self, X_Batch):
        x = self.classifier(X_Batch)
        return x

class Adult(nn.Module):
    def __init__(self, input_size=12, num_classes=2):
        super(Adult, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, X_Batch):
        return self.classifier(X_Batch)


class UTKFaceDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.processed_path = os.path.join(self.root, 'UTKFace/processed/')
        
        
        self.files = os.listdir(self.processed_path)
        print("self.root: ", self.root)    
        print("in the UTKFace dataset class constructor", self.processed_path)
        print("self files: ", self.files)
        # exit()
        
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []
        for txt_file in self.files:
            txt_file_path = os.path.join(self.processed_path, txt_file)
            with open(txt_file_path, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    if len(attrs) < 4 or int(attrs[2]) >= 4  or '' in attrs:
                        continue
                    self.lines.append(image_name+'jpg')


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        attrs = self.lines[index].split('_')

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])
        # print("in the __getitem__ method")
        
        image_path = os.path.join(self.root, 'UTKFace/raw/', self.lines[index]+'.chip.jpg').rstrip()
        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target


# Define the Perturbation Model
class PerturbationModel(nn.Module):
    def __init__(self, class_num,  device,  hidden_dim=128, layer_dim=1, output_dim=1, batch_size=64):
        super(PerturbationModel, self).__init__()

        self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.5),
			nn.Linear(1, 512),
			nn.ReLU(),
			nn.Linear(512, 64),
          
		)
        self.model = nn.Sequential(
            # nn.Linear(class_num, 128),  # First hidden layer with 64 neurons
            # nn.ReLU(),                 # Activation function
            # nn.BatchNorm1d(128),        # Batch normalization
            
            nn.Linear(class_num, 256),         # Second hidden layer with 32 neurons
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            
            nn.Linear(256, 128),         # Second hidden layer with 32 neurons
            nn.ReLU(),

            nn.Linear(128, 64),         # Second hidden layer with 32 neurons
            nn.ReLU(),
            # nn.BatchNorm1d(32),

            nn.Linear(64, 32),         # Second hidden layer with 32 neurons
            nn.ReLU(),

            nn.Linear(32, class_num), # Output layer matches input size
            nn.Sigmoid()               # Sigmoid ensures perturbation values are bounded (0, 1)
        )
    
    def forward(self, PV_batch, target_label_batch):
         return self.model(PV_batch)
       
 
from torch.utils.data import Dataset, DataLoader
import pickle

class AttackDataset(Dataset):
    def __init__(self, pickle_path):
        outputs = []
        predictions = []
        members = []
        targets = []
        with open(pickle_path, 'rb') as f:
            while True:
                try:
                    # Each pickle load returns a batch of (output, prediction, members, targets)
                    out, pred, mem, targ = pickle.load(f)
                    outputs.append(out)
                    predictions.append(pred)
                    members.append(mem)
                    targets.append(targ)
                except EOFError:
                    break
        # Concatenate all batches along the first dimension
        self.outputs = torch.cat(outputs, dim=0)
        self.predictions = torch.cat(predictions, dim=0)
        self.members = torch.cat(members, dim=0)
        self.targets = torch.cat(targets, dim=0)

    def __len__(self):
        return self.outputs.size(0)

    def __getitem__(self, idx):
        return self.outputs[idx], self.predictions[idx], self.members[idx], self.targets[idx]



class CombinedShadowAttack(nn.Module):
    def __init__(self, class_num,  device, mode, attack_name,  hidden_dim=128, layer_dim=1, output_dim=1, batch_size=64):
        
        super(CombinedShadowAttack, self).__init__()
        
        # batch_size = 2
        self.h_size_1 = 256
        self.h_size_2 = 128
        self.h_size_3 = 50
        
        self.lstm1 = nn.LSTM(class_num, self.h_size_1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(self.h_size_1, self.h_size_2, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lstm3 = nn.LSTM(self.h_size_2, self.h_size_3, batch_first=True)
        
        self.hidden2label = nn.Linear(self.h_size_3, 2)
        
        self.input_dim = class_num
        self.batch_size = batch_size
        
        self.batch_size = 64
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device = device
        
        self.mode = mode
        self.attack_name = attack_name
        
        
        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()
        self.hidden3 = self.init_hidden3()
        
        
        self.Output_NSH = nn.Sequential(
			nn.Linear(class_num, 10),
			nn.ReLU(),
			nn.Linear(10, 30),
            nn.ReLU(),
			nn.Linear(30, 10),
		)
        
        self.label_NSH = nn.Sequential(
			nn.Linear(class_num, 100),
			nn.ReLU(),
			nn.Linear(100, 5),
		)
        
        self.final_NSH = nn.Sequential(
			nn.Linear(5+10, 100),
			nn.ReLU(),
			nn.Linear(100, 50),
            nn.ReLU(),
			nn.Linear(50, 2),
		)
        
        
        
        self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 512),
			nn.ReLU(),
			nn.Linear(512, 64),
            # nn.ReLU(),
			# nn.Linear(256, 64),
		)
        
        self.Output_Component_meMIA = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
            nn.ReLU(),
			nn.Linear(256, 128),
            nn.ReLU(),
			nn.Linear(128, 64),
		)
        
        self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.5),
			nn.Linear(1, 512),
			nn.ReLU(),
			nn.Linear(512, 64),
          
		)

        self.meMIA_Encoder_Component_joint = nn.Sequential(
			nn.Linear(self.h_size_3+64+64, 512), #mine
			nn.ReLU(),
			# nn.Dropout(p=0.5),
			nn.Linear(512, 256),
			nn.ReLU(),
			# nn.Dropout(p=0.5),
			nn.Linear(256, 128),
			nn.ReLU(),
			# nn.Dropout(p=0.5),
            nn.Linear(128, 2),
		)
        
        self.mia_Encoder_Component = nn.Sequential(
			nn.Linear(class_num, 512), #mia
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
            nn.Linear(128, 2),
           
		)
        
        self.meMIA_Encoder_Component = nn.Sequential(
			nn.Linear(class_num+64, 512), #meMIA
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
            nn.Linear(128, 2),
           
		)
        self.Encoder_Component = nn.Sequential(
			nn.Linear(class_num+64, 512), #mia_actual
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
            nn.Linear(128, 2),
           
		)
        
        self.pertubed_attack_Component = nn.Sequential(
        nn.Linear(class_num+64, 512), 
        nn.ReLU(),
        # nn.BatchNorm1d(512),

        nn.Linear(512, 256),
        nn.ReLU(),
        # nn.BatchNorm1d(256),
        
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
        # nn.BatchNorm1d(128)
        
    )

   
        
    def init_hidden1(self):
        return (Variable(torch.zeros(1, self.batch_size, self.h_size_1).to(self.device)),
                Variable(torch.zeros(1, self.batch_size, self.h_size_1).to(self.device)))
    def init_hidden2(self):
        return (Variable(torch.zeros(1, self.batch_size, self.h_size_2).to(self.device)),
                Variable(torch.zeros(1, self.batch_size, self.h_size_2).to(self.device)))
    def init_hidden3(self):
        return (Variable(torch.zeros(1, self.batch_size, self.h_size_3).to(self.device)),
                Variable(torch.zeros(1, self.batch_size, self.h_size_3).to(self.device)))
   
    def pertubed_attack(self, output, prediction):
        Prediction_Component_result = self.Prediction_Component(prediction)
        return self.pertubed_attack_Component(torch.cat((Prediction_Component_result, output), 1))
    

    def get_embeddings(self, output, prediction, label):
        """
        For the apcmia attack, returns feature embeddings for contrastive loss computation.
        Here, we first compute the concatenated features (Prediction_Component output concatenated with output),
        then run them through all but the final layer of the pertubed_attack_Component.
        """
        if self.attack_name == "apcmia":
            Prediction_Component_result = self.Prediction_Component(prediction)
            features = torch.cat((Prediction_Component_result, output), 1)
            # Get a list of layers from the sequential model:
            layers = list(self.pertubed_attack_Component.children())
            # Run through all layers except the final one:
            for layer in layers[:-1]:
                features = layer(features)
            return features  # This is the embedding representation
        else:
            raise NotImplementedError("get_embeddings is implemented only for apcmia attack.")
            
    def forward(self, output, prediction, label):
        
        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()
        self.hidden3 = self.init_hidden3()
       
        if self.attack_name == "apcmia":
            # print("apcmia here!")
            # exit()
            return self.pertubed_attack(output, prediction)
        
        elif self.attack_name == "nsh":
            # ! NSH attack
            
            label_one_hot_encoded = torch.nn.functional.one_hot(label.to(torch.int64), self.input_dim).float().to(self.device)
            # print(f"size of label: {label_one_hot_encoded.size()}")
            # print(f"size of label: {label_one_hot_encoded.dtype}")
            
            # exit()
            
            out_nsh = self.Output_NSH(output)#ouput --> class_num
            lable_nsh =  self.label_NSH(label_one_hot_encoded)
            
            combined_nsh = torch.cat((out_nsh, lable_nsh), 1)
            final_result = self.final_NSH(combined_nsh)
            return final_result
        
        elif self.attack_name == "seqmia":
            x = output.view(self.batch_size, 1, self.input_dim)
            
            x1, self.hidden1 = self.lstm1(x, self.hidden1)
            x2 = self.dropout1(x1[:, -1, :])
            x2 = x2.view(self.batch_size, 1, x2.size()[1])
            x3, self.hidden2 = self.lstm2(x2, self.hidden2)
            x4 = self.dropout2(x3[:, -1, :])
            x4 = x4.view(self.batch_size, 1, x4.size()[1])
            x5, self.hidden2 = self.lstm3(x4, self.hidden3)
            
            final_result  = self.hidden2label(x5[:, -1, :])
            return final_result


        elif self.attack_name == "mia":
            #! mia
            # print("mia here!")
            # exit()
            Prediction_Component_result = self.Prediction_Component(prediction) #ouput --> class_num 64
            # output = self.Output_Component(output) #64
            final_result = self.Encoder_Component(torch.cat((Prediction_Component_result, output), 1))
            final_result = self.mia_Encoder_Component(output)
            return final_result

        elif self.attack_name == "memia":
            # ! Mine combined architecture
            label_one_hot_encoded = torch.nn.functional.one_hot(label.to(torch.int64), self.input_dim).float()
            
            x = output.view(self.batch_size, 1, self.input_dim)
            x1, self.hidden1 = self.lstm1(x, self.hidden1)
            # x2 = self.dropout1(x1[:, -1, :])
            x2 = x1[:, -1, :]
            x2 = x2.view(self.batch_size, 1, x2.size()[1])
            x3, self.hidden2 = self.lstm2(x2, self.hidden2)
            # x4 = self.dropout2(x3[:, -1, :])
            x4 = x3[:, -1, :]
            x4 = x4.view(self.batch_size, 1, x4.size()[1])
            x5, self.hidden2 = self.lstm3(x4, self.hidden3)
            
            output = self.Output_Component_meMIA(output) #64
            # exit()
            Prediction_Component_result = self.Prediction_Component(prediction) #ouput --> class_num|64
            final_inputs = torch.cat((x5[:, -1, :],Prediction_Component_result, output), 1)
            final_result = self.meMIA_Encoder_Component_joint(final_inputs)
            return final_result

            


class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, conv_params):
        super(ConvBlock, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        avg_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        conv_layers = []
        conv_layers.append(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))

        conv_layers.append(nn.ReLU())

        if avg_pool_size > 1:
            conv_layers.append(nn.AvgPool2d(kernel_size=avg_pool_size))

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FcBlock(nn.Module):
    def __init__(self, fc_params, flatten):
        super(FcBlock, self).__init__()
        input_size = int(fc_params[0])
        output_size = int(fc_params[1])

        fc_layers = []
        if flatten:
            fc_layers.append(Flatten())
        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd
import math

class VGG16(nn.Module):
    def __init__(self,input_channel, num_classes ):
        super(VGG16, self).__init__()

        self.input_size = 64
        self.num_classes = num_classes
        self.conv_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        self.fc_layer_sizes = [512, 512]

        self.max_pool_sizes = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        self.conv_batch_norm = True
        self.init_weights = True
        self.augment_training = False
        self.num_output = 1

        

        self.init_conv = nn.Sequential()

        self.layers = nn.ModuleList()
        # input_channel = 3
        cur_input_size = self.input_size
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size / 2)
            conv_params = (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            self.layers.append(ConvBlock(conv_params))
            input_channel = channel

        fc_input_size = cur_input_size * cur_input_size * self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width

        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        fwd = self.init_conv(x)

        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
