import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable


from meminf import *
from dataloader import *


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


    

   
        
from datetime import datetime
def target_train_func(PATH, device, train_set, test_set, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, arch):
    print("Training model: train set shape", len(train_set), " test set shape: ", len(test_set), ", device: ", device)
    print(f"dataset Name: {dataset_name}")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = target_train_class(train_loader, test_loader, dataset_name,  target_model, device, use_DP, noise, norm, delta, arch)
    
    acc_train = 0
    acc_test = 0
    for i in range(60):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("target training")
        acc_train = model.train()
        print("target testing")
        acc_test = model.test()
        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)
    FILE_PATH_target = PATH + "_target.pth"
   

    
    model.saveModel(FILE_PATH_target)

    print("Saved target model!!!")
   
    print("Finished training!!!")

     # Save the accuracies to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = PATH + f"_accs_{timestamp}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['acc_train', 'acc_test', 'overfitting'])
        writer.writerow([acc_train, acc_test, overfitting])
    print(f"Saved accuracies to {csv_file}")

    return overfitting
import wandb
import pytorch_lightning as pl

# train_lira_func(PATH, device, train_ds, test_ds, target_model, batch_size, dataset_name, shadow_id, keep_bool, n_shadows)
def train_lira_func(PATH, device, train_set, test_set, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, shadow_id, keep_bool, arch):
    print("Training model: train set shape", len(train_set), " test set shape: ", len(test_set), ", device: ", device)
    print(f"dataset Name: {dataset_name}")
    # train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=1)
    # test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=1)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = target_train_class(train_loader, test_loader, dataset_name,  target_model, device, use_DP, noise, norm, delta, arch)
    

    acc_train = 0
    acc_test = 0
    for i in range(60):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("target training")
        acc_train = model.train()
        # print("target testing")
        # acc_test = model.test()
        # overfitting = round(acc_train - acc_test, 6)
        # print('The overfitting rate is %s' % overfitting)
    # FILE_PATH_target = PATH + "_target.pth"
    
    acc_test = model.test()
    overfitting = round(acc_train - acc_test, 6)
    print('The overfitting rate is %s' % overfitting)

    print("Saved target model!!!")
   
    print("Finished training!!!")
    # exit()
    saveDIR = f"exp/{arch}/{dataset_name}" 

    savedir = os.path.join(saveDIR, str(shadow_id))
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/keep.npy", keep_bool)
    # torch.save(m.state_dict(), savedir + "/model.pt")
    model.saveModel(savedir + "/model.pt")
    # exit()

    
    # def run():
    
    
    for shadow_id in range(n_shadows):
        print(f"Training shadow {shadow_id}")
        # exit()

        seed = np.random.randint(0, 1000000000)
        seed ^= int(time.time())
        pl.seed_everything(seed)

        debug = True
        wandb.init(project="lira", mode="disabled" if debug else "online")
        # wandb.config.update(args)

        # train_ds = train_set
        # test_ds = test_set

        size = len(train_set)
        np.random.seed(seed)
        if n_shadows is not None:
            np.random.seed(0)
            keep = np.random.uniform(0, 1, size=(n_shadows, size))
            order = keep.argsort(0)
            keep = order < int(pkeep * n_shadows)
            keep = np.array(keep[shadow_id], dtype=bool)
            keep = keep.nonzero()[0]
        else:
            keep = np.random.choice(size, size=int(pkeep * size), replace=False)
            keep.sort()
        keep_bool = np.full((size), False)
        keep_bool[keep] = True

        train_ds = torch.utils.data.Subset(train_set, keep)
        # train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=1)
        # test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=1)
        # exit()
        train_lira_func(PATH, device, train_ds, test_set, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, shadow_id, keep_bool, arch)

        


def create_subsets(train_dataset, N):
    """
    Randomly create N subsets from train_dataset so that
    each sample appears in exactly N/2 of them.

    Args:
        train_dataset: A list (or similar) of training samples, e.g. [(x1, y1), (x2, y2), ...]
        N (int): Number of subsets/models to create. Must be even.
    Returns:
        subsets (list of lists): subsets[i] is a list of samples for model i.
    """
    assert N % 2 == 0, "N must be even so each sample can appear in N/2 subsets."
    M = len(train_dataset)

    # Create empty "bins" for each subset
    subsets = [[] for _ in range(N)]

    for i in range(M):
        # Randomly pick which N/2 subsets this sample belongs to
        chosen = random.sample(range(N), k=N//2)
        for subset_idx in chosen:
            subsets[subset_idx].append(train_dataset[i])

    return subsets


def prepare_attack_data_for_target(train_subset, test_set, mem_ratio=0.45, nonmem_ratio=0.45):
    """
    For a given target model, split its training subset into two halves and also split the test_set.
    
    Then append a membership flag:
      - For samples from the training subset, assign membership 1.
      - For samples from the test set, assign membership 0.
      
    Return:
      attack_train: (member part from training subset + non-member part from test_set)
      attack_test:  a second half of the training subset + second half of test_set.
      
    (The idea is that each target model is used to generate its own confidence signals
    with the known membership labels.)
    """
    # Split the training subset (members) into two parts
    mem_length = int(len(train_subset) * mem_ratio)
    mem_train, mem_test, _ = torch.utils.data.random_split(
        train_subset, [mem_length, mem_length, len(train_subset) - 2 * mem_length]
    )
    # Split the test set (non-members) into two parts
    nonmem_length = int(len(test_set) * nonmem_ratio)
    nonmem_train, nonmem_test, _ = torch.utils.data.random_split(
        test_set, [nonmem_length, nonmem_length, len(test_set) - 2 * nonmem_length]
    )
    
    # Convert to lists (if not already)
    mem_train = list(mem_train)
    mem_test = list(mem_test)
    nonmem_train = list(nonmem_train)
    nonmem_test = list(nonmem_test)
    
    # Append membership flag: add a new element to each sample tuple:
    mem_train = [sample + (1,) for sample in mem_train]
    mem_test  = [sample + (1,) for sample in mem_test]
    nonmem_train = [sample + (0,) for sample in nonmem_train]
    nonmem_test  = [sample + (0,) for sample in nonmem_test]
    
    # Combine the parts into attack training and attack test sets.
    attack_train = mem_train + nonmem_train
    attack_test  = mem_test + nonmem_test
    return attack_train, attack_test

def generate_confidences(PATH ,model, attack_train, attack_test, device, batch_size=64):
    """
    Given a trained model and an attack dataset (each sample is (x, y, membership)),
    compute the output (confidence) for the correct class for each sample.
    
    Returns:
      confidences: List of scalar confidence values.
      memberships: List of membership flags.
      targets: List of true labels.
    """

    
    state_dict = torch.load(PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

 
    combined_set = attack_train + attack_test
    loader = torch.utils.data.DataLoader(combined_set, batch_size=batch_size, shuffle=False)
    
    
    all_confidences = []
    all_members = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels, memberships in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # outputs shape: [batch_size, num_classes]
            # Here we assume you want the full output vector (the raw confidence) per sample.
            probs = F.softmax(outputs, dim=1)
            
            # Append each batch result (keep the batch dimension)
            all_confidences.append(probs.cpu().numpy())  # shape: (batch_size, num_classes)
            all_members.append(memberships.numpy())        # shape: (batch_size,) if memberships is 1D
            all_targets.append(labels.cpu().numpy())       # shape: (batch_size,) if labels is 1D

    # Concatenate all batches along the first axis
    all_confidences = np.concatenate(all_confidences, axis=0)  # shape: (total_samples, num_classes)
    all_members = np.concatenate(all_members, axis=0).reshape(-1, 1)  # reshape to (total_samples, 1)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 1)  # reshape to (total_samples, 1)
    

    return all_confidences, all_members, all_targets

def generate_confidences_full(PATH ,model, attack_test, device, batch_size=64):
    """
    Given a trained model and an attack dataset (each sample is (x, y, membership)),
    compute the output (confidence) for the correct class for each sample.
    
    Returns:
      confidences: List of scalar confidence values.
      memberships: List of membership flags.
      targets: List of true labels.
    """

    
    state_dict = torch.load(PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

 
    combined_set =  attack_test
    loader = torch.utils.data.DataLoader(combined_set, batch_size=batch_size, shuffle=False)
    
    
    all_confidences = []
    all_members = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels, memberships in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # outputs shape: [batch_size, num_classes]
            # Here we assume you want the full output vector (the raw confidence) per sample.
            probs = F.softmax(outputs, dim=1)
            
            # Append each batch result (keep the batch dimension)
            all_confidences.append(probs.cpu().numpy())  # shape: (batch_size, num_classes)
            all_members.append(memberships.numpy())        # shape: (batch_size,) if memberships is 1D
            all_targets.append(labels.cpu().numpy())       # shape: (batch_size,) if labels is 1D

    # Concatenate all batches along the first axis
    all_confidences = np.concatenate(all_confidences, axis=0)  # shape: (total_samples, num_classes)
    all_members = np.concatenate(all_members, axis=0).reshape(-1, 1)  # reshape to (total_samples, 1)
    all_targets = np.concatenate(all_targets, axis=0).reshape(-1, 1)  # reshape to (total_samples, 1)

    return all_confidences, all_members, all_targets

def train_multiple_targets_lira(PATH,train_dataset, test_dataset, target_model, device, batch_size,use_DP, noise, norm, delta, dataset_name,N):
    # heeeer
    print(f"PATH: {PATH +f"_lira_model_{0}"+ "_target.pth"}")

    # PATH +f"lira_model_{i}"+ "_target.pth"

    combined_signals_path = PATH + "_LiRA_mul.npz"
    # exit()
    """
    Train N different target models for LiRA, each on a random half-subset,
    ensuring each sample is in exactly N/2 of those subsets.

    Args:
        train_dataset (list): Full training data, e.g. list of (x, y).
        test_dataset (list): Full test data, used as the same test set for all models.
        N (int): Number of target models (must be even).
        create_model_func (callable): A function that returns a *fresh* untrained model.
        PATH_prefix (str): Base path for saving each model. We'll append _0, _1, etc.
        device: Torch device (e.g. "cuda" or "cpu").
        batch_size, use_DP, noise, norm, delta, dataset_name: Additional config for training.
    """

    # 1) Create N subsets from the training data
    subsets = create_subsets(train_dataset, N)
    print("done partition")
    # exit()
    # 2) Train a model on each subset

    all_conf_train = []
    all_mem_train  = []
    all_targ_train = []

    for i in range(1):
        print(f"\n--- Training Model {i+1}/{N} ---")
        subset_i = subsets[i]

        # Create a fresh untrained model
        model_i = target_model

        # Provide a unique path for each model
        # path_i = f"{PATH_prefix}_{i}"
        path_i = PATH +f"_lira_model_{i}"+ "_target.pth"

        # Now train
        # i am here
        acc_train, acc_test, overfitting, trained_model = target_train_func_lira_mul(
            PATH=path_i,
            device=device,
            train_set=subset_i,
            test_set=test_dataset,
            target_model=model_i,
            batch_size=batch_size,
            use_DP=use_DP,
            noise=noise,
            norm=norm,
            delta=delta,
            dataset_name=dataset_name
        )

        print(f"Finished Model {i}, Acc_Train: {acc_train}, Acc_Test: {acc_test}, Overfitting: {overfitting}")
        
        # Now, for this target model, prepare the attack data
        attack_train, attack_test = prepare_attack_data_for_target(subset_i, test_dataset)
        print(f"Attack train size: {len(attack_train)}, Attack test size: {len(attack_test)}")
        # ffffffff
        conf_train, mem_train, targ_train = generate_confidences(path_i, target_model, attack_train, attack_test , device, batch_size)
        # Print a few samples and their sizes
        print("Sample confidences:", conf_train[:5])
        print("Sample memberships:", mem_train[:5])
        print("Sample targets:", targ_train[:5])
        print("Size of confidences:", len(conf_train))
        print("Size of memberships:", len(mem_train))
        print("Size of targets:", len(targ_train))
        # Print shapes of the generated confidences, memberships, and targets
        print("Shape of confidences:", np.array(conf_train).shape)
        print("Shape of memberships:", np.array(mem_train).shape)
        print("Shape of targets:", np.array(targ_train).shape)
        # # Save the attack datasets (for later use when generating confidence outputs)
        # with open(path_i + "_attack_train.p", "wb") as f:
        #     pickle.dump(attack_train, f)
        # with open(path_i + "_attack_test.p", "wb") as f:
        #     pickle.dump(attack_test, f)
        # print(f"Saved attack data for target model {i}.")


        # Accumulate for combined signals:
        all_conf_train.extend(conf_train)
        all_mem_train.extend(mem_train)
        all_targ_train.extend(targ_train)
       
        # exit()

    # exit()
    # Also train one target model on the full training set:
    full_path = PATH +"_lira_model_FULL"+ "_target.pth"
    # train_dataset, 
    # test_dataset, 
    # acc_train, acc_test, overfitting, trained_model_full =target_train_func_full(
    #         PATH=full_path,
    #         device=device,
    #         train_set=train_dataset,
    #         test_set=test_dataset,
    #         target_model=target_model,
    #         batch_size=batch_size,
    #         use_DP=use_DP,
    #         noise=noise,
    #         norm=norm,
    #         delta=delta,
    #         dataset_name=dataset_name
    # )

    # full_attack_train, full_attack_test = prepare_attack_data_for_target(train_dataset, test_dataset)
    # full_conf_test, full_mem_test, full_targ_test = generate_confidences_full(full_path, target_model, full_attack_test, device, batch_size) # only need conf for test samples
    
    
    # print("Size of confidences (full):", len(full_conf_test))
    # print("Size of memberships (full):", len(full_mem_test))
    # print("Size of targets (full):", len(full_targ_test))
    # exit()


    full_conf_test = []
    full_mem_test  = []
    full_targ_test = []
    

    # trained_model = target_model
    print(f"Training Target model on N subsets")
    path = PATH + f"_lira_model_" + "_target.pth"

    for i in range(1):
        print(f"\n--- Training on Subset {i+1}/{N} ---")
        subset_i = subsets[i]
        
        # Provide a unique path for saving this iteration's model state (if desired)
        
        
        # Train the model on the current subset
        acc_train, acc_test, overfitting, trained_model = target_train_func_lira_mul(
            PATH=path,
            device=device,
            train_set=subset_i,
            test_set=test_dataset,
            target_model=target_model,  # use the same model (updated iteratively)
            batch_size=batch_size,
            use_DP=use_DP,
            noise=noise,
            norm=norm,
            delta=delta,
            dataset_name=dataset_name
        )
        # Update target_model with the newly trained weights

        state_dict = torch.load(path, map_location=device, weights_only=True)
        target_model.load_state_dict(state_dict)
        target_model.to(device)
        target_model.train()
        
        # load the trained model from path
        target_model = target_model

        print(f"Finished subset {i}, Acc_Train: {acc_train}, Acc_Test: {acc_test}, Overfitting: {overfitting}")
        
        # Now, for this target model, prepare the attack data
        attack_train, attack_test = prepare_attack_data_for_target(subset_i, test_dataset)
        # print(f"Attack train size: {len(attack_train)}, Attack test size: {len(attack_test)}")
        # ffffffff
        conf_train, mem_train, targ_train = generate_confidences(path_i, target_model, attack_train, attack_test , device, batch_size)
        # # Print a few samples and their sizes
        # print("Sample confidences:", conf_train[:5])
        # print("Sample memberships:", mem_train[:5])
        # print("Sample targets:", targ_train[:5])
        # print("Size of confidences:", len(conf_train))
        # print("Size of memberships:", len(mem_train))
        # print("Size of targets:", len(targ_train))
        # # Print shapes of the generated confidences, memberships, and targets
        # print("Shape of confidences:", np.array(conf_train).shape)
        # print("Shape of memberships:", np.array(mem_train).shape)
        # print("Shape of targets:", np.array(targ_train).shape)
        # # Save the attack datasets (for later use when generating confidence outputs)
        # with open(path_i + "_attack_train.p", "wb") as f:
        #     pickle.dump(attack_train, f)
        # with open(path_i + "_attack_test.p", "wb") as f:
        #     pickle.dump(attack_test, f)
        # print(f"Saved attack data for target model {i}.")


        # Accumulate for combined signals:
        full_conf_test.extend(conf_train)
        full_mem_test.extend(mem_train)
        full_targ_test.extend(targ_train)



    # save_path = PATH + '_lira_results.npz'
    np.savez(combined_signals_path,
             all_conf_train=all_conf_train,
             all_mem_train=all_mem_train,
             all_targ_train=all_targ_train,
             full_conf_test=full_conf_test,
             full_mem_test=full_mem_test,
             full_targ_test=full_targ_test)
    # print("Shape of all_conf_train:", np.array(all_conf_train).shape)
    # print("Shape of all_mem_train:", np.array(all_mem_train).shape)
    # print("Shape of all_targ_train:", np.array(all_targ_train).shape)
    # print("Shape of full_conf_test:", np.array(full_conf_test).shape)
    # print("Shape of full_mem_test:", np.array(full_mem_test).shape)
    # print("Shape of full_targ_test:", np.array(full_targ_test).shape)
    # exit()
    return all_conf_train, all_mem_train, all_targ_train, full_conf_test, full_mem_test, full_targ_test

def attack_lira(all_conf_train, all_mem_train, all_targ_train, full_conf_test, full_mem_test, full_targ_test):
    # wwwwwwwwww

    from scipy.stats import norm

        # outputs_list = []
        # members_list = []
        # targets_list = []

        # # Load data from the saved file (train.p)
        # with torch.no_grad():
        #     with open(self.ATTACK_SETS + "train.p", "rb") as f:
        #         while True:
        #             try:
        #                 output, prediction, members, targets = pickle.load(f)
        #             except EOFError:
        #                 break
        #             outputs_list.append(output.cpu())    # output: [batch, num_classes]
        #             members_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
        #             targets_list.append(targets.cpu())     # true labels for each sample

        # # Concatenate batches to get one tensor per item.
        # all_outputs = torch.cat(outputs_list, dim=0)   # shape: (total_samples, num_classes)
        # all_members = torch.cat(members_list, dim=0)     # shape: (total_samples,)
        # all_targets = torch.cat(targets_list, dim=0)     # shape: (total_samples,)
    # print("Type and shape of all_conf_train:", type(all_conf_train), np.array(all_conf_train).shape)
    # print("Type and shape of all_mem_train:", type(all_mem_train), np.array(all_mem_train).shape)
    # print("Type and shape of all_targ_train:", type(all_targ_train), np.array(all_targ_train).shape)
    # print("Type and shape of full_conf_test:", type(full_conf_test), np.array(full_conf_test).shape)
    # print("Type and shape of full_mem_test:", type(full_mem_test), np.array(full_mem_test).shape)
    # print("Type and shape of full_targ_test:", type(full_targ_test), np.array(full_targ_test).shape)
    # exit()
    # all_conf_train
    out_signals = all_conf_train     # non-members (out)
    
    
    mean_out = np.median(out_signals, 1).reshape(-1, 1)

    
    std_out = np.std(out_signals, 1).reshape(-1, 1)

    # print("Estimated distribution parameters:")
    # # print("Mean In-Signal:", mean_in)
    # print("Mean Out-Signal:", mean_out[:5])
    # # print("Std In-Signal:", std_in)
    # print("Std Out-Signal:", std_out[:5])
    # exit()
    # Now, for each sample, compute the negative log-likelihood under the two distributions.
    # Here, sc (signal observed) is the correct confidence for each sample.
    
    # outputs_test_list = []
    # members_test_list = []
    # targets_test_list = []

    # # get
    # # Load data from the saved file (test.p)
    # with torch.no_grad():
    #     with open(self.ATTACK_SETS + "test.p", "rb") as f:
    #         while True:
    #             try:
    #                 output, prediction, members, targets = pickle.load(f)
    #             except EOFError:
    #                 break
    #             outputs_test_list.append(output.cpu())    # output: [batch, num_classes]
    #             members_test_list.append(members.cpu())     # membership flag, e.g. 1 for member, 0 for non-member
    #             targets_test_list.append(targets.cpu())     # true labels for each sample

    # # Concatenate batches to get one tensor per item.
    # all_outputs_test = torch.cat(outputs_test_list, dim=0)   # shape: (total_samples, num_classes)
    # all_members_test = torch.cat(members_test_list, dim=0)     # shape: (total_samples,)
    # all_targets_test = torch.cat(targets_test_list, dim=0)     # shape: (total_samples,)

    # print("Loaded data from test.p:")
    # print(f"Outputs shape: {all_outputs_test.shape}")
    # print(f"Members shape: {all_members_test.shape}")
    # print(f"Targets shape: {all_targets_test.shape}")

    # exit()
    # xxxxxxxxxxx
    # get the ouput confidences for test_dataset of full_target_model and the corresponding members

    # sc = all_outputs_test
    # full_conf_test, full_mem_test
    all_outputs_test = full_conf_test

    
    
    print("Shape of all_outputs_test:", np.array(all_outputs_test).shape)
    print("Shape of mean_out:", np.array(mean_out).shape)
    print("Shape of std_out:", np.array(std_out).shape)
    # exit()
    prediction = []
    answers = []

    
    pr_in = 0
    
    pr_out = -norm.logpdf(all_outputs_test, mean_out, std_out + 1e-30) # gaussian approximation
    score = pr_in - pr_out

    # print("Score (few samples):", score[:5])
    # print("Type of score:", type(score))
    # print("Shape of score:", score.shape)
    # exit()
    prediction = np.array(score.mean(1))

    # prediction_2 = np.array(score.mean(1))
    prediction_2 = np.where(-score.mean(1) >= 0, 1, 0)
    

    # correct = predicted.eq(all_members_test).sum().item()
    answers = np.array(full_mem_test.reshape(-1, 1), dtype=bool)
    fpr_list, tpr_list, betas = roc_curve(answers.ravel(), (-prediction).ravel())
    # bcm = BinaryConfusionMatrix().to(self.device)
    # conf_mat = bcm((-prediction).ravel(), all_members_test.ravel())
    

    
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
    
    recall = np.max(tpr_list)
    print(f"acc: {acc}, recall: {recall:.3f}")
    
    
        
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
    

def target_train_func_lira_mul(PATH, device, train_set, test_set, target_model, batch_size, use_DP, noise, norm, delta, dataset_name):
    print("Training model: train set shape", len(train_set), "test set shape:", len(test_set), ", device:", device)
    print(f"Dataset Name: {dataset_name}")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = target_train_class(train_loader, test_loader, dataset_name, target_model, device, use_DP, noise, norm, delta)
    
    for i in range(60):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("Target training")
        acc_train = model.train()
        print("Target testing")
        acc_test = model.test()
        overfitting = round(acc_train - acc_test, 6)
        print("Overfitting rate:", overfitting)
    
    FILE_PATH_target = PATH
    model.saveModel(FILE_PATH_target)
    print("Saved target model at", FILE_PATH_target)
    return acc_train, acc_test, overfitting, model

def target_train_func_full(PATH, device, train_set, test_set, target_model, batch_size, use_DP, noise, norm, delta, dataset_name):
    """
    Trains one target model on the entire training set.
    """
    print("Training FULL target model: train set size:", len(train_set), " test set size:", len(test_set), ", device:", device)
    print(f"Dataset Name: {dataset_name}")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = target_train_class(train_loader, test_loader, dataset_name, target_model, device, use_DP, noise, norm, delta)
    
    for i in range(60):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("Full target training")
        acc_train = model.train()
        print("Full target testing")
        acc_test = model.test()
        overfitting = round(acc_train - acc_test, 6)
        print("Overfitting rate:", overfitting)
    
    FILE_PATH_full = PATH
    model.saveModel(FILE_PATH_full)
    print("Saved full target model at", FILE_PATH_full)
    return acc_train, acc_test, overfitting, model



# def shadow_train_func(PATH, device, shadow_model, batch_size, train_loader, test_loader, use_DP, noise, norm, loss, optimizer, delta):

def shadow_train_func(PATH, device, train_set, test_set, shadow_model, batch_size, use_DP, noise, norm, delta, dataset_name):

    print("Training shadow model: train set shape", len(train_set), "test set shape:", len(test_set), ", device:", device)
    print(f"Dataset Name: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = shadow_train_class(train_loader, test_loader, dataset_name, shadow_model, device, use_DP, noise, norm, delta)

    acc_train = 0
    acc_test = 0
    for i in range(60):
        print("<======================= Epoch " + str(i + 1) + " =======================>")
        print("shadow training")
        acc_train = model.train()
        print("shadow testing")
        acc_test = model.test()
        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)

    FILE_PATH = PATH + "_shadow.pth"
    model.saveModel(FILE_PATH)
    print("Saved shadow model!!!")
    print("Finished training!!!")

    return acc_train, acc_test, overfitting


# def test_meminf(TARGET_PATH, device, num_classes, target_train, target_test, dataset_name, batch_size, target_model, train_rnn, train_shadow, use_DP, noise, norm, delta, mode)

def test_meminf(PATH, device, num_classes, target_train, target_test, batch_size,  target_model, mode, dataset_name, attack_name, entropy_dis_dr, apcmia_cluster, arch, acc_gap):
    


    if attack_name == "lira" or attack_name == "memia" or attack_name == "seqmia" or attack_name == "nsh" or attack_name == "apcmia" or attack_name == "mia" or attack_name == "m_lira":
        # attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, batch_size)
        attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size)
    
        
        attack_model = CombinedShadowAttack(num_classes, device, mode, attack_name, hidden_dim=128, layer_dim=1, output_dim=1, batch_size=batch_size)
        
        perturb_model = PerturbationModel(num_classes, device, hidden_dim=128, layer_dim=1, output_dim=1, batch_size=batch_size)
        
        attack_mode0_com(PATH + "_target.pth", PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, perturb_model, num_classes, mode, dataset_name, attack_name, entropy_dis_dr, apcmia_cluster, arch, acc_gap)

    else:
        raise Exception("Wrong attack name")


def str_to_bool(string):
    if isinstance(string, bool):
       return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_fpr_tpr_for_all_attacks(dataset_name, directory="."):
    """
    Loads CSV files of the form:
        dataset_name_FPR_TPR_{attack_name}_.csv
    for the given dataset_name, where attack_name is in:
        ["apcmia", "mia", "seqmia", "memia", "nsh", "lira"].
    Returns a dict: { attack_name: (fpr_list, tpr_list) }.
    """
    attack_names = ["apcmia", "memia", "m_lira", "seqmia", "mia", "nsh"]
    # attack_names = ["apcmia","m_lira",]

    fpr_tpr_dict = {}

    for attack in attack_names:
        filename = f"{dataset_name}_FPR_TPR_{attack}_.csv"
        filepath = os.path.join(directory, filename)
        
        print(filepath)
        if os.path.isfile(filepath):
            try:
                df = pd.read_csv(filepath)
                fpr_list = df["FPR"].tolist()
                tpr_list = df["TPR"].tolist()
                fpr_tpr_dict[attack] = (fpr_list, tpr_list)
                print(f"Loaded {attack} from {filepath}")
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                fpr_tpr_dict[attack] = ([], [])
        else:
            print(f"File not found for attack '{attack}': {filepath}")
            fpr_tpr_dict[attack] = ([], [])
    # exit()
    return fpr_tpr_dict

def metric_results_new(fpr_list, tpr_list, attack_name, dataset_name, arch, directory="./tprs_at/"):
    """
    Calculates ROC AUC, accuracy (computed from ROC data), and TPR values at specified FPR thresholds from ROC data.
    Then updates a master CSV that accumulates three metrics for each (dataset, attack) combination:
      - TPR@0.001
      - Accuracy (acc)
      - ROC AUC
    The master CSV has one row per attack method and groups of columns for each dataset.
    
    Parameters:
      fpr_list: List of false positive rates.
      tpr_list: List of true positive rates.
      attack_name: Name of the attack (e.g., "apcMIA", "LiRa", etc.).
      dataset_name: Name of the dataset (e.g., "CIFAR-10", "STL-10", etc.). (Case-insensitive.)
      arch: Architecture (e.g., "vgg16", "cnn", etc.).
      directory: Base directory in which to store CSV files (default "./tprs_at/").
    
    Returns:
      roc_auc: ROC AUC computed from the fpr_list and tpr_list.
      acc: Accuracy computed from the ROC data.
      tpr_dict: Dictionary mapping FPR thresholds to TPR values.
    """
    # Build directory: e.g., ./tprs_at/<arch>/<dataset_name>/
    directory = os.path.join(directory, arch, dataset_name)
    os.makedirs(directory, exist_ok=True)

    # Convert lists to numpy arrays.
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)

    # Compute an "accuracy" from ROC data as: acc = max(1 - (FPR + (1 - TPR))/2)
    acc = np.max(1 - (fpr_array + (1 - tpr_array)) / 2)
    
    # Compute ROC AUC.
    roc_auc = auc(fpr_array, tpr_array)
    
    # Define FPR thresholds of interest.
    fprs = [0.01, 0.001, 0.0001, 0.00001, 0.0]
    tpr_dict = {}
    for threshold in fprs:
        indices = np.where(fpr_array <= threshold)[0]
        if len(indices) > 0:
            tpr_val = tpr_array[indices[-1]]
        else:
            tpr_val = None
        tpr_dict[threshold] = tpr_val
    
    # Round TPR values to 3 decimals.
    tpr_dict = {k: round(v, 3) if v is not None else None for k, v in tpr_dict.items()}
    
    print(f"Dataset: {dataset_name}, Attack: {attack_name}, Arch: {arch}")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"TPR at 0.001 FPR: {tpr_dict[0.001]}")
    
    # Save all threshold/TPR pairs for this dataset+attack
    df_all = pd.DataFrame(list(tpr_dict.items()), columns=["FPR", "TPR"])
    df_all.sort_values(by="FPR", ascending=False, inplace=True)
    file_name_all = f"{dataset_name}_tprAT_{attack_name}.csv"
    path_all = os.path.join(directory, file_name_all)
    df_all.to_csv(path_all, index=False)
    print(f"TPR values saved to {path_all}")

    # --- Update the master CSV ---
    # We want columns as follows: 
    # Method, CIFAR-10_tpr001, STL-10_tpr001, CIFAR-100_tpr001, UTKFace_tpr001, FMNIST_tpr001,
    #         CIFAR-10_acc, STL-10_acc, CIFAR-100_acc, UTKFace_acc, FMNIST_acc,
    #         CIFAR-10_auc, STL-10_auc, CIFAR-100_auc, UTKFace_auc, FMNIST_auc
    master_cols = ["Method",
                   "CIFAR-10_tpr001", "STL-10_tpr001", "CIFAR-100_tpr001", "UTKFace_tpr001", "FMNIST_tpr001",
                   "CIFAR-10_acc",   "STL-10_acc",   "CIFAR-100_acc",   "UTKFace_acc",   "FMNIST_acc",
                   "CIFAR-10_auc",   "STL-10_auc",   "CIFAR-100_auc",   "UTKFace_auc",   "FMNIST_auc"]
    
    # Mapping from attack names to display names for the "Method" column.
    attack_map = {
        "apcmia": "apcMIA",
        "lira":   "LiRa",
        "memia":  "meMIA",
        "seqmia": "seqMIA",
        "nsh":    "NSH",
        "mia":    "MIA",
    }
    
    # Mapping from dataset_name (lower case) to the three master CSV columns.
    dataset_map = {
        "cifar10":  ("CIFAR-10_tpr001", "CIFAR-10_acc", "CIFAR-10_auc"),
        "stl10":    ("STL-10_tpr001",   "STL-10_acc",   "STL-10_auc"),
        "cifar100": ("CIFAR-100_tpr001", "CIFAR-100_acc", "CIFAR-100_auc"),
        "utkface":  ("UTKFace_tpr001",   "UTKFace_acc",   "UTKFace_auc"),
        "fmnist":   ("FMNIST_tpr001",    "FMNIST_acc",    "FMNIST_auc")
    }
    
    method_row = attack_map.get(attack_name.lower())
    dataset_key = dataset_name.lower()
    if method_row is None or dataset_key not in dataset_map:
        print(f"Warning: Unrecognized attack '{attack_name}' or dataset '{dataset_name}'.")
        return roc_auc, acc, tpr_dict
    col_tpr, col_acc, col_auc = dataset_map[dataset_key]
    
    # Define master CSV path: store it in the parent directory of the dataset folder for this architecture.
    master_csv = os.path.join(os.path.dirname(directory), f"master_all_{arch}.csv")
    
    # Create the master DataFrame if it doesn't exist.
    if os.path.exists(master_csv):
        df_master = pd.read_csv(master_csv)
    else:
        df_master = pd.DataFrame(columns=master_cols)
    
    # Check if a row for the current method exists.
    if method_row in df_master["Method"].values:
        # Update the corresponding dataset columns for the attack method.
        df_master.loc[df_master["Method"] == method_row, col_tpr] = round(tpr_dict[0.001], 3)
        df_master.loc[df_master["Method"] == method_row, col_acc] = round(acc, 3)
        df_master.loc[df_master["Method"] == method_row, col_auc] = round(roc_auc, 3)
    else:
        # Create a new row with None values, then update the current method's columns.
        new_row = {col: None for col in master_cols}
        new_row["Method"] = method_row
        new_row[col_tpr] = round(tpr_dict[0.001], 3)
        new_row[col_acc] = round(acc, 3)
        new_row[col_auc] = round(roc_auc, 3)
        df_master = pd.concat([df_master, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the updated master CSV.
    df_master.to_csv(master_csv, index=False)
    print(f"Master CSV updated at: {master_csv}")
    
    return roc_auc, acc, tpr_dict

def metric_results_mlp(fpr_list, tpr_list, attack_name, dataset_name, arch, directory="./tprs_at/"):
    """
    Calculates ROC AUC, accuracy (computed from ROC data), and TPR values at specified FPR thresholds 
    from ROC data. Then updates a master CSV that accumulates three metrics for each (attack, dataset) combination:
      - TPR@0.001
      - Accuracy (acc)
      - ROC AUC
      
    The master CSV is structured with one row per attack method and three groups of columns (one group per metric) 
    for each dataset. In our case, we have four datasets: Location, Adult, Texas-100, and Purchase-100.
    
    Parameters:
      fpr_list: List of false positive rates.
      tpr_list: List of true positive rates.
      attack_name: Name of the attack (e.g., "apcMIA", "LiRa", etc.).
      dataset_name: Name of the dataset (e.g., "Location", "Adult", "Texas-100", "Purchase-100"). Case-insensitive.
      arch: Architecture (e.g., "vgg16", "cnn", etc.).
      directory: Base directory for storing CSV files.
    
    Returns:
      roc_auc: ROC AUC computed from the fpr_list and tpr_list.
      acc: Accuracy computed from the ROC data.
      tpr_dict: Dictionary mapping FPR thresholds to TPR values.
    """
    # Build directory: e.g., ./tprs_at/<arch>/<dataset_name>/
    directory = os.path.join(directory, arch, dataset_name)
    os.makedirs(directory, exist_ok=True)

    # Convert lists to numpy arrays.
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)

    # Compute accuracy from ROC data as: acc = max(1 - (FPR + (1 - TPR))/2)
    acc = np.max(1 - (fpr_array + (1 - tpr_array)) / 2)
    
    # Compute ROC AUC.
    roc_auc = auc(fpr_array, tpr_array)
    
    # Define FPR thresholds of interest.
    fprs = [0.01, 0.001, 0.0001, 0.00001, 0.0]
    tpr_dict = {}
    for threshold in fprs:
        indices = np.where(fpr_array <= threshold)[0]
        if len(indices) > 0:
            tpr_val = tpr_array[indices[-1]]
        else:
            tpr_val = None
        tpr_dict[threshold] = tpr_val
    
    # Round TPR values to 3 decimals.
    tpr_dict = {k: round(v, 3) if v is not None else None for k, v in tpr_dict.items()}
    
    print(f"Dataset: {dataset_name}, Attack: {attack_name}, Arch: {arch}")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"TPR at 0.001 FPR: {tpr_dict[0.001]}")
    
    # Save CSV with all (FPR, TPR) pairs for this dataset+attack.
    df_all = pd.DataFrame(list(tpr_dict.items()), columns=["FPR", "TPR"])
    df_all.sort_values(by="FPR", ascending=False, inplace=True)
    file_name_all = f"{dataset_name}_tprAT_{attack_name}.csv"
    path_all = os.path.join(directory, file_name_all)
    df_all.to_csv(path_all, index=False)
    print(f"TPR values saved to {path_all}")

    # --- Update the master CSV ---
    # Define master CSV columns for four datasets:
    master_cols = ["Method",
                   "Location_tpr001", "Adult_tpr001", "Texas-100_tpr001", "Purchase-100_tpr001",
                   "Location_acc",   "Adult_acc",   "Texas-100_acc",   "Purchase-100_acc",
                   "Location_auc",   "Adult_auc",   "Texas-100_auc",   "Purchase-100_auc"]
    
    # Mapping from attack names (lower case) to display names for the "Method" column.
    attack_map = {
        "apcmia": "apcMIA",
        "lira":   "LiRa",
        "memia":  "meMIA",
        "seqmia": "seqMIA",
        "nsh":    "NSH",
        "mia":    "MIA",
    }
    
    # Mapping from dataset_name (lower case) to master CSV columns (tpr001, acc, auc).
    dataset_map = {
        "location":    ("Location_tpr001",    "Location_acc",    "Location_auc"),
        "adult":       ("Adult_tpr001",       "Adult_acc",       "Adult_auc"),
        "texas":   ("Texas-100_tpr001",   "Texas-100_acc",   "Texas-100_auc"),
        "purchase":("Purchase-100_tpr001","Purchase-100_acc","Purchase-100_auc")
    }
    
    method_row = attack_map.get(attack_name.lower())
    dataset_key = dataset_name.lower()
    if method_row is None or dataset_key not in dataset_map:
        print(f"Warning: Unrecognized attack '{attack_name}' or dataset '{dataset_name}'.")
        return roc_auc, acc, tpr_dict
    col_tpr, col_acc, col_auc = dataset_map[dataset_key]
    
    # Define master CSV path: stored in the parent directory of the current dataset folder (per architecture).
    master_csv = os.path.join(os.path.dirname(directory), f"master_all_{arch}.csv")
    
    # Load master CSV if it exists; otherwise, create a new DataFrame.
    if os.path.exists(master_csv):
        df_master = pd.read_csv(master_csv)
    else:
        df_master = pd.DataFrame(columns=master_cols)
    
    # Check if a row for the current method already exists.
    if method_row in df_master["Method"].values:
        # Update the corresponding columns for the given dataset.
        df_master.loc[df_master["Method"] == method_row, col_tpr] = round(tpr_dict[0.001], 3)
        df_master.loc[df_master["Method"] == method_row, col_acc] = round(acc, 3)
        df_master.loc[df_master["Method"] == method_row, col_auc] = round(roc_auc, 3)
    else:
        # Create a new row with None in all master columns, then set the current method's values.
        new_row = {col: None for col in master_cols}
        new_row["Method"] = method_row
        new_row[col_tpr] = round(tpr_dict[0.001], 3)
        new_row[col_acc] = round(acc, 3)
        new_row[col_auc] = round(roc_auc, 3)
        df_master = pd.concat([df_master, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the updated master CSV.
    df_master.to_csv(master_csv, index=False)
    print(f"Master CSV updated at: {master_csv}")
    
    return roc_auc, acc, tpr_dict


def metric_results(fpr_list, tpr_list, attack_name, dataset_name, arch, directory="./tprs_at/"):
    """
    Calculates ROC AUC, accuracy, and TPR values at specified FPR thresholds from ROC data.
    Then it saves two CSV files:
      1) A CSV containing all threshold/TPR pairs for this (dataset, attack) combination.
      2) A master CSV (all_tpr_0.001.csv) that accumulates the TPR@0.001 for each dataset,
         with one row per dataset and one column per attack method.

    Parameters:
      fpr_list: List of false positive rates.
      tpr_list: List of true positive rates.
      attack_name: Name of the attack (e.g., "apcMIA", "LiRa", etc.).
      dataset_name: Name of the dataset (e.g., "FMNIST", "UTKFace", etc.).
      arch: Architecture (e.g., "vgg16", "cnn", etc.).
      directory: Base directory in which to store the CSV files (default "./tprs_at/").
    
    Returns:
      roc_auc: ROC AUC value.
      acc: Accuracy computed from the ROC curve data.
      tpr_dict: Dictionary mapping each specified FPR threshold to its TPR.
    """
    # Build directory: ./tprs_at/<arch>/<dataset_name>/
    directory = os.path.join(directory, arch, dataset_name)
    os.makedirs(directory, exist_ok=True)

    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)

    # Compute accuracy and ROC AUC from ROC curve data.
    acc = np.max(1 - (fpr_array + (1 - tpr_array)) / 2)
    roc_auc = auc(fpr_array, tpr_array)

    # Define FPR thresholds of interest.
    fprs = [0.01, 0.001, 0.0001, 0.00001, 0.0]
    tpr_dict = {}
    for threshold in fprs:
        indices = np.where(fpr_array <= threshold)[0]
        if len(indices) > 0:
            tpr_val = tpr_array[indices[-1]]
        else:
            tpr_val = None
        tpr_dict[threshold] = tpr_val

    # Round TPR values to 3 decimals.
    tpr_dict = {k: round(v, 3) if v is not None else None for k, v in tpr_dict.items()}

    print(f"Dataset: {dataset_name}, Attack: {attack_name}, Arch: {arch}")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"TPR at 0.001 FPR: {tpr_dict[0.001]}")
    print("All TPR values:")
    for threshold, value in tpr_dict.items():
        print(f"  FPR {threshold}: TPR = {value}")

    # -------------------------------------------------------------------
    # 1) Save a CSV with all threshold/TPR pairs for this dataset+attack.
    # -------------------------------------------------------------------
    df_all = pd.DataFrame(list(tpr_dict.items()), columns=["FPR", "TPR"])
    df_all.sort_values(by="FPR", ascending=False, inplace=True)
    file_name_all = f"{dataset_name}_tprAT_{attack_name}.csv"
    path_all = os.path.join(directory, file_name_all)
    df_all.to_csv(path_all, index=False)
    print(f"TPR values saved to {path_all}")

    # -------------------------------------------------------------------
    # 2) Update a master CSV (one per architecture) that accumulates TPR@0.001 values.
    #    The master CSV has one row per dataset and one column for each attack.
    # -------------------------------------------------------------------
    # Master CSV is stored under the arch folder (one level up from the dataset folder)

    l_fpr = 0.001;
    master_csv = os.path.join(os.path.dirname(directory), f"all_tpr_{l_fpr}.csv")

    # Define expected columns. (You can add more columns if needed.)
    columns = ["Dataset", "apcMIA", "LiRa", "meMIA", "seqMIA", "NSH", "MIA"]

    # Mapping from attack names to CSV column names.
        # Mapping from attack names (in lower case) to CSV column names.
    attack_col_map = {
        "apcmia": "apcMIA",
        "lira":   "LiRa",
        "memia":  "meMIA",
        "seqmia": "seqMIA",
        "nsh":    "NSH",
        "mia":    "MIA",
    }

    # Get the TPR value for FPR=0.001.
    tpr_0_001 = tpr_dict[l_fpr]

    # Load master CSV if it exists; otherwise, create a new DataFrame.
    if os.path.exists(master_csv):
        df_master = pd.read_csv(master_csv)
    else:
        df_master = pd.DataFrame(columns=columns)

    # Check if a row for this dataset already exists.
    if dataset_name in df_master["Dataset"].values:
        # Update the row for this dataset.
        col_name = attack_col_map.get(attack_name.lower())
        if col_name is not None:
            df_master.loc[df_master["Dataset"] == dataset_name, col_name] = tpr_0_001
    else:
        # Create a new row.
        new_row = {col: None for col in columns}
        new_row["Dataset"] = dataset_name
        col_name = attack_col_map.get(attack_name.lower())
        if col_name is not None:
            new_row[col_name] = tpr_0_001
        df_master = pd.concat([df_master, pd.DataFrame([new_row])], ignore_index=True)

    # Save the master CSV.
    df_master.to_csv(master_csv, index=False)
    print(f"Master CSV updated at: {master_csv}")

    return roc_auc, acc, tpr_dict

# plot_roc_curves_for_attacks(fpr_tpr_data, dataset_name, threshold_curves_pth, arch)
def plot_roc_curves_for_attacks(fpr_tpr_dict, dataset_name, save_path, arch):
    """
    Plots ROC curves for each attack on the same figure.
    `
    Parameters:
      - fpr_tpr_dict: dict of {attack_name: (fpr_list, tpr_list)}
      - dataset_name: Name of the dataset (string)
      - save_path: If provided, saves the figure to the given path (e.g., "roc_plot.pdf")
    """

    
        
    # print(f"Save path for ROC curves: {save_path}")
    # exit(
    # from matplotlib.font_manager import FontProperties
    # Update rcParams with your settings
    # Update rcParams with your settings
    # Update rcParams with your settings
    size = 30
    params = {
        'axes.labelsize': size,
        'font.size': size,
        'legend.fontsize': size,
        'xtick.labelsize': size,
        'ytick.labelsize': size,
        'figure.figsize': [10, 9],
        "font.family": "arial",
    }

    # params = {
    # 'axes.labelsize': size,
    # 'axes.labelweight': 'bold',   # Make axis labels bold
    # 'font.size': size,
    # 'font.weight': 'bold',        # Make all fonts bold
    # 'legend.fontsize': size,
    # 'xtick.labelsize': size,
    # 'ytick.labelsize': size,
    # 'figure.figsize': [10, 9],
    # "font.family": "arial",
    # }
    
    plt.rcParams.update(params)

    # Define a color palette for consistency
    attack_colors = {
        "apcmia": "#0d0478",   # Blue
        "mia":    "#9467bd",   # Red
        "seqmia": "#2ca02c",   # Green
        "memia":  "#d62728",   # Purple
        "nsh":    "#ff7f0e",   # Orange
        "lira":   "#8c564b"    # Brownish
    }

   
    # Define line styles for each attack method
    attack_linestyles = {
        "apcmia": "-",
        "mia":    "--",
        "seqmia": "-.",
        "memia":  ":",
        "nsh":    "-",
        "lira":   "--"
    }

    # Assume fpr_tpr_dict, dataset_name, arch, and save_path are defined.
    # fpr_tpr_dict: a dictionary mapping attack names to (fpr_list, tpr_list)

    for attack_name, (fpr, tpr) in fpr_tpr_dict.items():
        if not fpr or not tpr:
            continue
        # Normalize attack names to desired display names:
        if attack_name.upper() == "M_LIRA":
            display_name = "LiRA"
        elif attack_name.upper() == "MEMIA":
            display_name = "meMIA"
        elif attack_name.upper() == "SEQMIA":
            display_name = "seqMIA"
        elif attack_name.upper() == "APCMIA":
            display_name = "apcMIA"
        elif attack_name.upper() == "NSH":
            display_name = "NSH"
        elif attack_name.upper() == "MIA":
            display_name = "MIA"
        else:
            display_name = attack_name

        # Retrieve marker and line style; default if not found.
        # marker = attack_markers.get(display_name.lower(), "o")
        linestyle = attack_linestyles.get(display_name.lower(), "-")
        color = attack_colors.get(display_name.lower(), "black")
        
        
        
        # Plot the ROC curve using specified style
        plt.plot(fpr, tpr,
                label=rf"{display_name}",
                color=color, lw=3.5, linestyle=linestyle, markersize=6)
        
        # If needed, update metrics in master CSV (function call remains the same)
        metric_results_new(fpr, tpr, display_name, dataset_name, arch)

    # Plot the diagonal (random classifier)
    plt.plot([0, 1], [0, 1], ls="--", color="gray", lw=2)

    # Set labels and other figure properties
    plt.xlabel("False Positive Rate", fontsize=size)
    plt.ylabel("True Positive Rate", fontsize=size)
    legend = plt.legend(loc="lower right")
    frame = legend.get_frame()
    frame.set_edgecolor('0.91')
    plt.subplots_adjust(left=0.60)
    plt.tight_layout()

    # Set axes to log scale
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1.01)

    # Save the figure as a PDF with 300 dpi
    filename = f"{dataset_name}_roc_curves_{arch}.pdf"
    filepath = os.path.join(save_path, filename)
    print(f"File path for roc curve: {filepath}")
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(filepath, format='pdf', dpi=300, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    # plt.show()


                    

def load_plot_thresholds(base_dir, threshold_save_path):
    """
    Recursively finds all CSV files ending with 
    '_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv' under base_dir.
    For each file:
      - Infers the architecture from the directory structure,
      - Loads the CSV and slices data up to (and including) the row with minimal test_loss,
      - Smooths the curves (using a rolling average),
      - Plots the cosine threshold (\tau_c) and entropy threshold (\tau_e) on the left y-axis,
        and test loss on the right y-axis,
      - Saves the resulting plot in a subfolder (named by architecture) under threshold_save_path.
    """
    os.makedirs(threshold_save_path, exist_ok=True)

    # Plot settings
    size = 20
    params = {
       'axes.labelsize': size,
       'font.size': size,
       'legend.fontsize': size,
       'xtick.labelsize': size,
       'ytick.labelsize': size,
       # 'text.usetex': False,
       'figure.figsize': [10, 8],
       "font.family": "arial",
    }
    plt.rcParams.update(params)

    # Recursively walk through the base_dir
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv"):
                filepath = os.path.join(root, file)

                # Determine the architecture (e.g., "./demoloader/trained_model/cnn/cifar10/..." -> "cnn")
                rel_path = os.path.relpath(root, base_dir)
                arch = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path

                # Create subfolder for the architecture if it doesn't exist
                arch_save_path = os.path.join(threshold_save_path, arch)
                os.makedirs(arch_save_path, exist_ok=True)

                # Infer dataset name from the file name (e.g., "cifar10_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv" -> "cifar10")
                suffix = "_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv"
                dataset_name = file.replace(suffix, "")

                print(f"Loading: {filepath}")
                df = pd.read_csv(filepath)

                required_cols = {"epoch", "cosine_threshold", "entropy_threshold", "test_loss"}
                if not required_cols.issubset(df.columns):
                    print(f"Skipping {file} -- missing one of {required_cols}.")
                    continue

                # Slice the dataframe up to (and including) the row with minimal test_loss
                min_idx = df["test_loss"].idxmin()
                df = df.iloc[:min_idx + 1]

                # Apply smoothing (rolling average with window=3)
                epochs     = df["epoch"]
                cos_thresh = df["cosine_threshold"].rolling(window=3, min_periods=1).mean()
                ent_thresh = df["entropy_threshold"].rolling(window=3, min_periods=1).mean()
                test_loss  = df["test_loss"].rolling(window=3, min_periods=1).mean()

                # Create the plot
                fig, ax1 = plt.subplots()
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Threshold Value")
                # Plot the smoothed cosine threshold with label \tau_c
                line_cos = ax1.plot(epochs, cos_thresh, label=r"$\tau_c$",
                                      marker="o", color="tab:blue")
                # Plot the smoothed entropy threshold with label \tau_e
                line_ent = ax1.plot(epochs, ent_thresh, label=r"$\tau_e$",
                                      marker="s", color="tab:orange")

                # Create a second y-axis for test_loss
                ax2 = ax1.twinx()
                ax2.set_ylabel("Attack Loss")
                line_loss = ax2.plot(epochs, test_loss, label="Test Loss",
                                     marker="^", color="tab:green")

                # Combine legends from both axes
                lines = line_cos + line_ent + line_loss
                labels = [l.get_label() for l in lines]
                legend = ax1.legend(lines, labels, loc="upper center",
                                    bbox_to_anchor=(0.5, 1.15), ncol=3)
                legend.get_frame().set_facecolor("0.95")
                legend.get_frame().set_edgecolor("0.91")

                plt.grid(linestyle="dotted")
                plt.tight_layout()

                # Save the figure, e.g., "cifar10_thresholds_and_loss_min.png"
                out_name = f"{dataset_name}_thresholds_and_loss_min.png"
                out_path = os.path.join(arch_save_path, out_name)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"Saved plot for '{dataset_name}' in arch '{arch}' -> {out_path}")


def load_plot_thresholds_sub(base_dir, threshold_save_path):
    """
    Recursively finds all CSV files ending with 
    '_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv' under base_dir.
    
    For each file:
      - Infers the architecture from the directory structure,
      - Loads the CSV and slices data up to (and including) the row with minimal test_loss,
      - Smooths the curves (using a rolling average with window=3),
      - Plots the cosine threshold (\tau_c) and entropy threshold (\tau_e) in the top subplot,
        and the test loss in the bottom subplot (with a shared x-axis),
      - Saves the resulting plot in a subfolder (named by architecture) under threshold_save_path.
    """

    print(f"Base directory PLOTTING: {base_dir}")

    os.makedirs(threshold_save_path, exist_ok=True)

    # Plot settings
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

    # Walk recursively through base_dir
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv"):
                filepath = os.path.join(root, file)

                # Determine the architecture from the relative path
                # Expected structure: ./demoloader/trained_model/<arch>/<dataset>/...
                rel_path = os.path.relpath(root, base_dir)
                arch = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path

                # Create a subfolder for the architecture
                arch_save_path = os.path.join(threshold_save_path, arch)
                os.makedirs(arch_save_path, exist_ok=True)

                # Infer dataset name by removing the known suffix from the filename.
                suffix = "_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv"
                dataset_name = file.replace(suffix, "")

                print(f"Loading: {filepath}")
                df = pd.read_csv(filepath)

                # Ensure the required columns exist
                required_cols = {"epoch", "cosine_threshold", "entropy_threshold", "test_loss"}
                if not required_cols.issubset(df.columns):
                    print(f"Skipping {file} -- missing one of {required_cols}.")
                    continue

                # Slice the dataframe up to (and including) the row with minimal test_loss
                min_idx = df["test_loss"].idxmin()
                df = df.iloc[:min_idx + 1]

                # Smooth curves with a rolling average (window=3)
                epochs     = df["epoch"]
                cos_thresh = df["cosine_threshold"].rolling(window=3, min_periods=1).mean()
                ent_thresh = df["entropy_threshold"].rolling(window=3, min_periods=1).mean()
                test_loss  = df["test_loss"].rolling(window=3, min_periods=1).mean()

                # Create subplots: two rows sharing the same x-axis
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
                # fig.suptitle(f"{dataset_name} Thresholds and Test Loss", fontsize=size+2)

                # Top subplot: plot thresholds
                ax1.plot(epochs, cos_thresh, label=r"$\tau_c$", marker="o", color="tab:blue")
                ax1.plot(epochs, ent_thresh, label=r"$\tau_e$", marker="s", color="tab:orange")
                ax1.set_ylabel("Threshold Value")
                ax1.legend(loc="best")
                ax1.grid(linestyle="dotted")

                # Bottom subplot: plot test loss
                ax2.plot(epochs, test_loss, label="Test Loss", marker="^", color="tab:green")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Attack Loss")
                ax2.legend(loc="best")
                ax2.grid(linestyle="dotted")

                # Adjust layout to ensure no clipping and reserve space for the suptitle
                plt.tight_layout(rect=[0, 0, 1, 0.95])

                # Save the figure
                out_name = f"{dataset_name}_thresholds_and_loss_min_subplot.pdf"
                out_path = os.path.join(arch_save_path, out_name)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"Saved plot for '{dataset_name}' in arch '{arch}' -> {out_path}")


def load_plot_thresholds_bestEp(base_dir, threshold_save_path):
    """
    Recursively finds all CSV files ending with 
    '_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv' under base_dir.
    
    For each file:
      - Infers the architecture from the directory structure,
      - Loads the CSV and slices data up to (and including) the row with minimal test_loss,
      - Smooths the curves (using a rolling average with window=3),
      - Plots the cosine threshold (\tau_c) and entropy threshold (\tau_e) in the top subplot,
        and the test loss in the bottom subplot (with a shared x-axis),
      - Highlights the best epoch (where test_loss is minimal) with a vertical dashed red line and annotation,
      - Saves the resulting plot in a subfolder (named by architecture) under threshold_save_path.
    """
    os.makedirs(threshold_save_path, exist_ok=True)

    # Plot settings
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

    # Walk recursively through base_dir
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv"):
                filepath = os.path.join(root, file)

                # Determine the architecture from the relative path
                rel_path = os.path.relpath(root, base_dir)
                arch = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path

                # Create a subfolder for the architecture
                arch_save_path = os.path.join(threshold_save_path, arch)
                os.makedirs(arch_save_path, exist_ok=True)

                # Infer dataset name by removing the known suffix from the filename.
                suffix = "_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv"
                dataset_name = file.replace(suffix, "")

                print(f"Loading: {filepath}")
                df = pd.read_csv(filepath)

                required_cols = {"epoch", "cosine_threshold", "entropy_threshold", "test_loss"}
                if not required_cols.issubset(df.columns):
                    print(f"Skipping {file} -- missing one of {required_cols}.")
                    continue

                # Slice the dataframe up to (and including) the row with minimal test_loss
                min_idx = df["test_loss"].idxmin()
                df = df.iloc[:min_idx + 1]

                # Smooth curves with a rolling average (window=3)
                epochs     = df["epoch"]
                cos_thresh = df["cosine_threshold"].rolling(window=5, min_periods=1).mean()
                ent_thresh = df["entropy_threshold"].rolling(window=5, min_periods=1).mean()
                test_loss  = df["test_loss"].rolling(window=5, min_periods=1).mean()

                # Get the best epoch (minimum test loss)
                best_epoch = epochs.iloc[min_idx]
                best_loss  = test_loss.iloc[min_idx]

                # Create subplots: two rows sharing the same x-axis
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
                # fig.suptitle(f"{dataset_name} Thresholds and Test Loss", fontsize=size+2)

                # Top subplot: plot thresholds
                ax1.plot(epochs, cos_thresh, label=r"$\tau_c$", marker="o", color="tab:blue")
                ax1.plot(epochs, ent_thresh, label=r"$\tau_e$", marker="s", color="tab:orange")
                ax1.set_ylabel("Threshold Value")
                ax1.legend(loc="best")
                ax1.grid(linestyle="dotted")

                # Bottom subplot: plot test loss
                ax2.plot(epochs, test_loss, label="Test Loss", marker="^", color="tab:green")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Attack Loss")
                ax2.legend(loc="best")
                ax2.grid(linestyle="dotted")

                # Highlight the best epoch with a vertical line and annotation on the test loss plot
                ax2.axvline(x=best_epoch, color="red", linestyle="--", linewidth=2, label="Best Epoch")
                # Annotate the best epoch
                ax2.annotate(f"Best Epoch\n({best_epoch})", xy=(best_epoch, best_loss),
                             xytext=(best_epoch, best_loss + 0.05 * best_loss),
                             arrowprops=dict(facecolor="red", shrink=0.05),
                             horizontalalignment="center", color="red", fontsize=size-2)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                out_name = f"{dataset_name}_thresholds_and_loss_min_subplot.png"
                out_path = os.path.join(arch_save_path, out_name)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"Saved plot for '{dataset_name}' in arch '{arch}' -> {out_path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('-a', '--attributes', type=str, default="race", help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('-dn', '--dataset_name', type=str, default="location")
    parser.add_argument('-at', '--attack_type', type=int, default=0)
    parser.add_argument('-tm', '--train_model', action='store_true')
    parser.add_argument('-ts', '--train_shadow', action='store_true')
    parser.add_argument('-trnn', '--train_rnn', action='store_true')
    parser.add_argument('-ud', '--use_DP', action='store_true')
    parser.add_argument('-ne', '--noise', type=float, default=1.3)
    parser.add_argument('-nm', '--norm', type=float, default=1.5)
    parser.add_argument('-d', '--delta', type=float, default=1e-5)
    parser.add_argument('-m', '--mode', type=int, default=0)

    parser.add_argument('-dsize', '--DSize', type=int, default=30000)
    
    parser.add_argument('-an', '--attack_name', type=str, default="mia")
    parser.add_argument('-plt', '--plot', action='store_true')
    parser.add_argument('-roc', '--plot_results', type=str, default="roc")

    parser.add_argument('-arch', '--arch', type=str, default="cnn")
    
    parser.add_argument('-l_tr', '--lira_train', action='store_true')
    parser.add_argument('-l_inf', '--lira_inference', action='store_true')
    parser.add_argument('-l_roc', '--lira_roc', action='store_true')
    parser.add_argument('-n_queries', '--aug', type=int, default=2)
    # parser.add_argument('-l_plt', '--lira_inference', action='store_true')
    parser.add_argument('-plt_cls', '--apcmia_cluster', action='store_true')

    args = parser.parse_args()

    print(args.DSize)
    # exit()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")

    # exit()
    dataset_name = args.dataset_name
    dataset_name = dataset_name.lower()
    arch = args.arch
    arch = arch.lower()
    
    attr = args.attributes
    if "_" in attr:
        attr = attr.split("_")
    root = "./data"
    use_DP = args.use_DP
    noise = args.noise
    norm = args.norm
    delta = args.delta
    mode = args.mode
    apcmia_cluster = args.apcmia_cluster
    
    attack_name = args.attack_name
    attack_name = attack_name.lower()
    # print(f"Attack Name (lowercase): {attack_name}")
    # exit()
    train_shadow = args.train_shadow
    train_rnn = args.train_rnn

    # First, check if the dataset requires mlp
    if dataset_name.lower() in ('location', 'texas', 'adult', 'purchase'):
        if arch.lower() != 'mlp':
            print("For datasets 'location', 'texas', 'adult', 'purchase', only the 'mlp' architecture is allowed!")
            exit()

    # Then, if arch is among the allowed architectures, set up the paths.
    if arch.lower() in ('vgg16', 'cnn', 'wrn', 'mlp'):
        TARGET_ROOT = f"./demoloader/trained_model/{arch}/{dataset_name}/"
        roc_curves_pth = f"./roc_curves/{arch}/{dataset_name}/"
        entropy_dis_dr = f"./entropy_dis/{arch}/{dataset_name}/"
        threshold_curves_pth = f"./thresh_curves/{arch}/{dataset_name}/" # ./demoloader/trained_model/cnn/cifar10/adult_meminf_attack_mode0__com_Results-Mean_mode-apcmia_.csv
        # roc_curves_pth = f"./thresh_curves/{arch}/{dataset_name}/"
        
    else:
        print("Incorrect architecture type! Provide one of these => ['vgg16', 'cnn', 'wrn', 'mlp']")
        exit()

    if not os.path.exists(TARGET_ROOT):
        print(f"Create directory named {TARGET_ROOT}")
        os.makedirs(TARGET_ROOT)

    MODEL_SAVE_PATH = TARGET_ROOT + dataset_name
    print("Target_patth: ",  MODEL_SAVE_PATH)

    
    if dataset_name.lower() == "purchase":
        batch_size = 64
    else:
        batch_size = 64
    

    from datetime import datetime
    import pandas as pd
    if args.plot:
        if args.plot_results.lower() == "roc":
            print("roc")

            # save fpr_tpr_data to a csv file at filename = f"{dataset_name}_roc_curves_{arch}.pdf", filepath = os.path.join(roc_curves_pth, filename) with time stamp,
            

            fpr_tpr_data = load_fpr_tpr_for_all_attacks(dataset_name, directory=TARGET_ROOT)
           
            # Save fpr_tpr_data to a CSV file
            if "apcmia" in fpr_tpr_data:
                fpr, tpr = fpr_tpr_data["apcmia"]
                df_roc = pd.DataFrame({"FPR": fpr, "TPR": tpr})

                fpr_array = np.array(fpr)
                tpr_array = np.array(tpr)

                # Compute accuracy and ROC AUC from ROC curve data.
                acc = np.max(1 - (fpr_array + (1 - tpr_array)) / 2)
                roc_auc = auc(fpr_array, tpr_array)

                fprs = [0.01, 0.001, 0.0001, 0.00001, 0.0]
                tpr_dict = {}
                for threshold in fprs:
                    indices = np.where(fpr_array <= threshold)[0]
                    if len(indices) > 0:
                        tpr_val = tpr_array[indices[-1]]
                    else:
                        tpr_val = None
                    tpr_dict[threshold] = tpr_val

                print(f"TPR at 0.001 FPR: {tpr_dict[0.001]}")

                # Create the metrics DataFrame
                metrics = {
                    "AUC": [roc_auc],
                    "Accuracy": [acc],
                    "TPR @ 0.1% FPR": [tpr_dict[0.001]],
                    "TPR @ 1% FPR": [tpr_dict[0.01]],
                }
                df_metrics = pd.DataFrame(metrics)

                # Build the output path with timestamp
                os.makedirs(roc_curves_pth, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{dataset_name}_roc_curves_{arch}_{timestamp}.xlsx"
                filepath = os.path.join(roc_curves_pth, filename)

                # Save both DataFrames to different sheets in the same Excel file.
                with pd.ExcelWriter(filepath) as writer:
                    df_roc.to_excel(writer, sheet_name="ROC_Curves", index=False)
                    df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
            else:
                print("apcmia data not found.")
             # 2. Plot the ROC curves for all attacks in a single figure.
            print(f"ROC saved to {roc_curves_pth}")
            # exit()

            

            plot_roc_curves_for_attacks(fpr_tpr_data, dataset_name, roc_curves_pth, arch)
            
        elif args.plot_results.lower() == "th":

            print(f"attack name is {attack_name}")
            if attack_name == "apcmia":
                print("plotting thresholds for apcmia")

                base_directory = "./demoloader/trained_model"  # top-level directory to search
                output_dir     = "./threshold_plots"          # where to save the figures
                # load_plot_thresholds(base_directory, output_dir)
                load_plot_thresholds_sub(base_directory, output_dir)
                # scatter_3d_thresholds(TARGET_ROOT, threshold_curves_pth)
                # load_plot_thresholds_bestEp(base_directory, output_dir)
                exit()
            else:

                print(f"can't plot thresholds for {attack_name}, try apcmia")
                exit()

            
        else:
            print("Incorrect plot argument")
        exit()
    
 
    # num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(dataset_name, attr, root, device)
    num_classes, target_train, target_test, target_model =  prepare_dataset(dataset_name, attr, root, device, arch, args.DSize)
    # combined_signals_path = MODEL_SAVE_PATH + "_LiRA_mul.npz"

    if args.train_model:
        print("Training Target model")
        
        acc_gap = target_train_func(MODEL_SAVE_PATH, device, target_train, target_test, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, arch)
        print("acc_gap: ", acc_gap)
        # exit()

    
    N=4
    aug = 2
    n_qury = 1
    if attack_name == "m_lira": 
            print("LiRA inference.")
            
            if args.lira_train:
                lira_train_models(MODEL_SAVE_PATH, device, target_train, target_test, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, arch, n_shadows=N, pkeep=0.5)
            
            if args.lira_inference:
                lira_inference(device, target_train, target_test, target_model, batch_size, dataset_name, arch, n_qury)

            if args.lira_roc:
                    
                # lira_scores()
                # lira_score(dataset_name, target_train)
                # print("Type of target_train:", type(target_train))
                lira_process = Lira_score_process(MODEL_SAVE_PATH, attack_name, target_train, dataset_name, aug, arch, ntest=1)

                # lira_process.enhanced_mia_attack()
                lira_process.lira_score()
                lira_process.fig_fpr_tpr()
                # lira_process.enhanced_mia_attack()
                # lira_process.rmia_attack_loss_reference()

            
            exit()

    if args.attack_type == 0:
        # acc_gap = 0.335
        acc_gap = 0.009418
        test_meminf(MODEL_SAVE_PATH, device, num_classes, target_train, target_test, batch_size,  target_model, mode, dataset_name, attack_name, entropy_dis_dr, apcmia_cluster, arch, acc_gap)
        
    else:
        sys.exit("we have not supported this mode yet! 0c0")

if __name__ == "__main__":
    main()
