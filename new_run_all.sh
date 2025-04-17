# # Image-based datasets (using CNN):
# python main.py --attack_type 0 --dataset_name fmnist --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name utkface --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name stl10 --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name cifar10 --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name cifar100 --arch cnn --train_model

# # Other datasets (using MLP):
# python main.py --attack_type 0 --dataset_name location --arch mlp --train_model
# python main.py --attack_type 0 --dataset_name purchase --arch mlp --train_model
# python main.py --attack_type 0 --dataset_name texas --arch mlp --train_model
# python main.py --attack_type 0 --dataset_name adult --arch mlp --train_model


# # -----------------------# FOR CNN architecture-----------------------------------------------

# For fmnist (cnn):   
# # python main.py --attack_type 0 --dataset_name fmnist --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name fmnist --attack_name mia --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name seqmia --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name memia --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name nsh --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_train --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_inference --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name apcmia --arch cnn 
# python main.py --plot --plot_results roc --dataset_name fmnist



# # For utkface (cnn)
# # python main.py --attack_type 0 --dataset_name utkface --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name utkface --attack_name mia --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name seqmia --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name memia --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name nsh --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_train --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_inference --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name apcmia --arch cnn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name utkface


# # For stl10 (cnn)
# # python main.py --attack_type 0 --dataset_name stl10 --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name stl10 --attack_name mia --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name seqmia --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name memia --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name nsh --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_train --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_inference --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name apcmia --arch cnn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name stl10


# # For cifar10 (cnn)
# # python main.py --attack_type 0 --dataset_name cifar10 --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name mia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name seqmia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name memia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name nsh --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_train --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_inference --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name apcmia --arch cnn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name cifar10


# # For cifar100 (cnn)
# # python main.py --attack_type 0 --dataset_name cifar100 --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name mia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name seqmia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name memia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name nsh --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_train --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_inference --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name apcmia --arch cnn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name cifar100


# # For location (mlp) done
# # python main.py --attack_type 0 --dataset_name location --arch mlp --train_model
# python main.py --attack_type 0 --dataset_name location --attack_name mia --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name seqmia --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name memia --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name nsh --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name m_lira --lira_train --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name m_lira --lira_inference --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name m_lira --lira_roc --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name apcmia --arch mlp --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name location


# # For purchase (mlp)
# # python main.py --attack_type 0 --dataset_name purchase --arch mlp --train_model
# python main.py --attack_type 0 --dataset_name purchase --attack_name mia --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name seqmia --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name memia --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name nsh --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name m_lira --lira_train --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name m_lira --lira_inference --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name m_lira --lira_roc --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name apcmia --arch mlp --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name purchase



# # For texas (mlp)
# # python main.py --attack_type 0 --dataset_name texas --arch mlp --train_model
# python main.py --attack_type 0 --dataset_name texas --attack_name mia --arch mlp
# python main.py --attack_type 0 --dataset_name texas --attack_name seqmia --arch mlp
# python main.py --attack_type 0 --dataset_name texas --attack_name memia --arch mlp
# python main.py --attack_type 0 --dataset_name texas --attack_name nsh --arch mlp
# python main.py --attack_type 0 --dataset_name texas --attack_name m_lira --lira_train --arch mlp
# python main.py --attack_type 0 --dataset_name texas --attack_name m_lira --lira_inference --arch mlp
# python main.py --attack_type 0 --dataset_name texas --attack_name m_lira --lira_roc --arch mlp
# python main.py --attack_type 0 --dataset_name texas --attack_name apcmia --arch mlp --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name texas



# # For adult (mlp)
# # python main.py --attack_type 0 --dataset_name adult --arch mlp --train_model
# python main.py --attack_type 0 --dataset_name adult --attack_name mia --arch mlp
# python main.py --attack_type 0 --dataset_name adult --attack_name seqmia --arch mlp
# python main.py --attack_type 0 --dataset_name adult --attack_name memia --arch mlp
# python main.py --attack_type 0 --dataset_name adult --attack_name nsh --arch mlp
# python main.py --attack_type 0 --dataset_name adult --attack_name m_lira --lira_train --arch mlp
# python main.py --attack_type 0 --dataset_name adult --attack_name m_lira --lira_inference --arch mlp
# python main.py --attack_type 0 --dataset_name adult --attack_name m_lira --lira_roc --arch mlp
# python main.py --attack_type 0 --dataset_name adult --attack_name apcmia --arch mlp --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name adult












# # # -----------------------# FOR vgg16 architecture-----------------------------------------------

# # For fmnist (vgg16):   
# python main.py --attack_type 0 --dataset_name fmnist --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name fmnist --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name apcmia --arch vgg16 
# python main.py --plot --plot_results roc --dataset_name fmnist



# # For utkface (vgg16)
# python main.py --attack_type 0 --dataset_name utkface --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name utkface --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name apcmia --arch vgg16 --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name utkface


# # For stl10 (vgg16)
# python main.py --attack_type 0 --dataset_name stl10 --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name stl10 --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name apcmia --arch vgg16 --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name stl10


# # For cifar10 (vgg16)
# python main.py --attack_type 0 --dataset_name cifar10 --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name apcmia --arch vgg16 --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name cifar10


# # For cifar100 (vgg16)
# python main.py --attack_type 0 --dataset_name cifar100 --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name apcmia --arch vgg16 --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name cifar100


# # For location (mlp) done
# # python main.py --attack_type 0 --dataset_name location --arch mlp --train_model
# python main.py --attack_type 0 --dataset_name location --attack_name mia --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name seqmia --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name memia --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name nsh --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name m_lira --lira_train --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name m_lira --lira_inference --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name m_lira --lira_roc --arch mlp
# python main.py --attack_type 0 --dataset_name location --attack_name apcmia --arch mlp --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name location




# # For purchase (mlp)
# python main.py --attack_type 0 --dataset_name purchase --arch mlp --train_model
# python main.py --attack_type 0 --dataset_name purchase --attack_name mia --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name seqmia --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name memia --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name nsh --arch mlp
# python main.py --attack_type 0 --dataset_name purchase --attack_name m_lira --lira_train --arch mlp
# # python main.py --attack_type 0 --dataset_name purchase --attack_name m_lira --lira_inference --arch mlp
# # python main.py --attack_type 0 --dataset_name purchase --attack_name m_lira --lira_roc --arch mlp
# # python main.py --attack_type 0 --dataset_name purchase --attack_name apcmia --arch mlp --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name purchase



# # # For texas (mlp)
# # python main.py --attack_type 0 --dataset_name texas --arch mlp --train_model
# # python main.py --attack_type 0 --dataset_name texas --attack_name mia --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name seqmia --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name memia --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name nsh --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name m_lira --lira_train --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name m_lira --lira_inference --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name m_lira --lira_roc --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name apcmia --arch mlp --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name texas



# # # For adult (mlp)
# # python main.py --attack_type 0 --dataset_name adult --arch mlp --train_model
# # python main.py --attack_type 0 --dataset_name adult --attack_name mia --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name seqmia --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name memia --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name nsh --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name m_lira --lira_train --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name m_lira --lira_inference --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name m_lira --lira_roc --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name apcmia --arch mlp --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name adult






# # # # -----------------------# FOR WRN architecture-----------------------------------------------

# # # For fmnist (wrn):   
# # python main.py --attack_type 0 --dataset_name fmnist --arch wrn --train_model
# # python main.py --attack_type 0 --dataset_name fmnist --attack_name mia --arch wrn
# # python main.py --attack_type 0 --dataset_name fmnist --attack_name seqmia --arch wrn
# # python main.py --attack_type 0 --dataset_name fmnist --attack_name memia --arch wrn
# # python main.py --attack_type 0 --dataset_name fmnist --attack_name nsh --arch wrn
# # python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_train --arch wrn
# # python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_inference --arch wrn
# # python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_roc --arch wrn
# # python main.py --attack_type 0 --dataset_name fmnist --attack_name apcmia --arch wrn 
# # python main.py --plot --plot_results roc --dataset_name fmnist



# # # For utkface (wrn)
# # python main.py --attack_type 0 --dataset_name utkface --arch wrn --train_model
# # python main.py --attack_type 0 --dataset_name utkface --attack_name mia --arch wrn
# # python main.py --attack_type 0 --dataset_name utkface --attack_name seqmia --arch wrn
# # python main.py --attack_type 0 --dataset_name utkface --attack_name memia --arch wrn
# # python main.py --attack_type 0 --dataset_name utkface --attack_name nsh --arch wrn
# # python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_train --arch wrn
# # python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_inference --arch wrn
# # python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_roc --arch wrn
# # python main.py --attack_type 0 --dataset_name utkface --attack_name apcmia --arch wrn --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name utkface


# # # For stl10 (wrn)
# # python main.py --attack_type 0 --dataset_name stl10 --arch wrn --train_model
# # python main.py --attack_type 0 --dataset_name stl10 --attack_name mia --arch wrn
# # python main.py --attack_type 0 --dataset_name stl10 --attack_name seqmia --arch wrn
# # python main.py --attack_type 0 --dataset_name stl10 --attack_name memia --arch wrn
# # python main.py --attack_type 0 --dataset_name stl10 --attack_name nsh --arch wrn
# # python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_train --arch wrn
# # python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_inference --arch wrn
# # python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_roc --arch wrn
# # python main.py --attack_type 0 --dataset_name stl10 --attack_name apcmia --arch wrn --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name stl10


# # # For cifar10 (wrn)
# # python main.py --attack_type 0 --dataset_name cifar10 --arch wrn --train_model
# # python main.py --attack_type 0 --dataset_name cifar10 --attack_name mia --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar10 --attack_name seqmia --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar10 --attack_name memia --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar10 --attack_name nsh --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_train --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_inference --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_roc --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar10 --attack_name apcmia --arch wrn --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name cifar10


# # # For cifar100 (wrn)
# # python main.py --attack_type 0 --dataset_name cifar100 --arch wrn --train_model
# # python main.py --attack_type 0 --dataset_name cifar100 --attack_name mia --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar100 --attack_name seqmia --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar100 --attack_name memia --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar100 --attack_name nsh --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_train --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_inference --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_roc --arch wrn
# # python main.py --attack_type 0 --dataset_name cifar100 --attack_name apcmia --arch wrn --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name cifar100


# # # For location (mlp) done
# # python main.py --attack_type 0 --dataset_name location --arch mlp --train_model
# # python main.py --attack_type 0 --dataset_name location --attack_name mia --arch mlp
# # python main.py --attack_type 0 --dataset_name location --attack_name seqmia --arch mlp
# # python main.py --attack_type 0 --dataset_name location --attack_name memia --arch mlp
# # python main.py --attack_type 0 --dataset_name location --attack_name nsh --arch mlp
# # python main.py --attack_type 0 --dataset_name location --attack_name m_lira --lira_train --arch mlp
# # python main.py --attack_type 0 --dataset_name location --attack_name m_lira --lira_inference --arch mlp
# # python main.py --attack_type 0 --dataset_name location --attack_name m_lira --lira_roc --arch mlp
# # python main.py --attack_type 0 --dataset_name location --attack_name apcmia --arch mlp --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name location




# # # For purchase (mlp)
# # python main.py --attack_type 0 --dataset_name purchase --arch mlp --train_model
# # python main.py --attack_type 0 --dataset_name purchase --attack_name mia --arch mlp
# # python main.py --attack_type 0 --dataset_name purchase --attack_name seqmia --arch mlp
# # python main.py --attack_type 0 --dataset_name purchase --attack_name memia --arch mlp
# # python main.py --attack_type 0 --dataset_name purchase --attack_name nsh --arch mlp
# # python main.py --attack_type 0 --dataset_name purchase --attack_name m_lira --lira_train --arch mlp
# # python main.py --attack_type 0 --dataset_name purchase --attack_name m_lira --lira_inference --arch mlp
# # python main.py --attack_type 0 --dataset_name purchase --attack_name m_lira --lira_roc --arch mlp
# # python main.py --attack_type 0 --dataset_name purchase --attack_name apcmia --arch mlp --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name purchase



# # # For texas (mlp)
# # python main.py --attack_type 0 --dataset_name texas --arch mlp --train_model
# # python main.py --attack_type 0 --dataset_name texas --attack_name mia --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name seqmia --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name memia --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name nsh --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name m_lira --lira_train --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name m_lira --lira_inference --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name m_lira --lira_roc --arch mlp
# # python main.py --attack_type 0 --dataset_name texas --attack_name apcmia --arch mlp --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name texas



# # # For adult (mlp)
# # python main.py --attack_type 0 --dataset_name adult --arch mlp --train_model
# # python main.py --attack_type 0 --dataset_name adult --attack_name mia --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name seqmia --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name memia --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name nsh --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name m_lira --lira_train --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name m_lira --lira_inference --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name m_lira --lira_roc --arch mlp
# # python main.py --attack_type 0 --dataset_name adult --attack_name apcmia --arch mlp --apcmia_cluster
# # python main.py --plot --plot_results roc --dataset_name adult







# #############################
# # fmnist Experiments
# #############################
# echo "================== fmnist with CNN =================="
# python main.py --attack_type 0 --dataset_name fmnist --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name fmnist --attack_name mia --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name seqmia --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name memia --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name nsh --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_train --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_inference --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name apcmia --arch cnn
# python main.py --plot --plot_results roc --dataset_name fmnist

# echo "================== fmnist with VGG16 =================="
# python main.py --attack_type 0 --dataset_name fmnist --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name fmnist --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name apcmia --arch vgg16
# python main.py --plot --plot_results roc --dataset_name fmnist

# echo "================== fmnist with WRN =================="
# python main.py --attack_type 0 --dataset_name fmnist --arch wrn --train_model
# python main.py --attack_type 0 --dataset_name fmnist --attack_name mia --arch wrn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name seqmia --arch wrn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name memia --arch wrn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name nsh --arch wrn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_train --arch wrn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_inference --arch wrn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_roc --arch wrn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name apcmia --arch wrn
# python main.py --plot --plot_results roc --dataset_name fmnist

# #############################
# # utkface Experiments
# #############################
# echo "================== utkface with CNN =================="
# python main.py --attack_type 0 --dataset_name utkface --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name utkface --attack_name mia --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name seqmia --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name memia --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name nsh --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_train --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_inference --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name apcmia --arch cnn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name utkface

# echo "================== utkface with VGG16 =================="
# python main.py --attack_type 0 --dataset_name utkface --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name utkface --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name apcmia --arch vgg16 --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name utkface

# echo "================== utkface with WRN =================="
# python main.py --attack_type 0 --dataset_name utkface --arch wrn --train_model
# python main.py --attack_type 0 --dataset_name utkface --attack_name mia --arch wrn
# python main.py --attack_type 0 --dataset_name utkface --attack_name seqmia --arch wrn
# python main.py --attack_type 0 --dataset_name utkface --attack_name memia --arch wrn
# python main.py --attack_type 0 --dataset_name utkface --attack_name nsh --arch wrn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_train --arch wrn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_inference --arch wrn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_roc --arch wrn
# python main.py --attack_type 0 --dataset_name utkface --attack_name apcmia --arch wrn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name utkface

# #############################
# # stl10 Experiments
# #############################
# echo "================== stl10 with CNN =================="
# python main.py --attack_type 0 --dataset_name stl10 --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name stl10 --attack_name mia --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name seqmia --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name memia --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name nsh --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_train --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_inference --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name apcmia --arch cnn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name stl10 --arch cnn

# echo "================== stl10 with VGG16 =================="
# python main.py --attack_type 0 --dataset_name stl10 --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name stl10 --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name apcmia --arch vgg16 --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name stl10

# echo "================== stl10 with WRN =================="
# python main.py --attack_type 0 --dataset_name stl10 --arch wrn --train_model
# python main.py --attack_type 0 --dataset_name stl10 --attack_name mia --arch wrn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name seqmia --arch wrn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name memia --arch wrn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name nsh --arch wrn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_train --arch wrn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_inference --arch wrn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_roc --arch wrn
# python main.py --attack_type 0 --dataset_name stl10 --attack_name apcmia --arch wrn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name stl10

# #############################
# # cifar10 Experiments
# #############################
# echo "================== cifar10 with CNN =================="
# python main.py --attack_type 0 --dataset_name cifar10 --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name mia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name seqmia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name memia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name nsh --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_train --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_inference --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name apcmia --arch cnn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name cifar10

# echo "================== cifar10 with VGG16 =================="
# python main.py --attack_type 0 --dataset_name cifar10 --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name apcmia --arch vgg16 --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name cifar10

# echo "================== cifar10 with WRN =================="
# python main.py --attack_type 0 --dataset_name cifar10 --arch wrn --train_model
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name mia --arch wrn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name seqmia --arch wrn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name memia --arch wrn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name nsh --arch wrn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_train --arch wrn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_inference --arch wrn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_roc --arch wrn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name apcmia --arch wrn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name cifar10

# #############################
# # cifar100 Experiments
# #############################
# echo "================== cifar100 with CNN =================="
# python main.py --attack_type 0 --dataset_name cifar100 --arch cnn --train_model
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name mia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name seqmia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name memia --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name nsh --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_train --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_inference --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name apcmia --arch cnn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name cifar100

# echo "================== cifar100 with VGG16 =================="
# python main.py --attack_type 0 --dataset_name cifar100 --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name apcmia --arch vgg16 --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name cifar100


echo "================== fmnist with VGG16 =================="
# python main.py --attack_type 0 --dataset_name fmnist --arch vgg16 --train_model
# python main.py --attack_type 0 --dataset_name fmnist --attack_name mia --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name seqmia --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name memia --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name nsh --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_train --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_inference --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name apcmia --arch vgg16
# python main.py --plot --plot_results roc --dataset_name fmnist

# echo "================== cifar100 with WRN =================="
# python main.py --attack_type 0 --dataset_name cifar100 --arch wrn --train_model
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name mia --arch wrn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name seqmia --arch wrn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name memia --arch wrn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name nsh --arch wrn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_train --arch wrn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_inference --arch wrn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_roc --arch wrn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name apcmia --arch wrn --apcmia_cluster
# python main.py --plot --plot_results roc --dataset_name cifar100

# echo "================== All experiments finished =================="




# # changes to offline mode
# # python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_roc --arch vgg16
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_roc --arch vgg16



# python main.py --plot --plot_results roc --dataset_name stl10 --arch vgg16
# python main.py --plot --plot_results roc --dataset_name cifar10 --arch vgg16
# python main.py --plot --plot_results roc --dataset_name cifar100 --arch vgg16
# python main.py --plot --plot_results roc --dataset_name fmnist --arch vgg16
# python main.py --plot --plot_results roc --dataset_name utkface --arch vgg16



# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_roc --arch wrn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_roc --arch wrn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_roc --arch wrn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_roc --arch wrn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_roc --arch wrn


# python main.py --plot --plot_results roc --dataset_name stl10 --arch wrn
# python main.py --plot --plot_results roc --dataset_name cifar10 --arch wrn
# python main.py --plot --plot_results roc --dataset_name cifar100 --arch wrn
# python main.py --plot --plot_results roc --dataset_name fmnist --arch wrn
# python main.py --plot --plot_results roc --dataset_name utkface --arch wrn


# python main.py --attack_type 0 --dataset_name stl10 --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name cifar100 --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name fmnist --attack_name m_lira --lira_roc --arch cnn
# python main.py --attack_type 0 --dataset_name utkface --attack_name m_lira --lira_roc --arch cnn

# python main.py --plot --plot_results roc --dataset_name stl10 --arch cnn
# python main.py --plot --plot_results roc --dataset_name cifar10 --arch cnn
# python main.py --plot --plot_results roc --dataset_name cifar100 --arch cnn
# python main.py --plot --plot_results roc --dataset_name fmnist --arch cnn
# python main.py --plot --plot_results roc --dataset_name utkface --arch cnn

# python main.py --plot --plot_results roc --dataset_name stl10 --arch vgg16
# python main.py --plot --plot_results roc --dataset_name cifar10 --arch vgg16
# python main.py --plot --plot_results roc --dataset_name cifar100 --arch vgg16
# python main.py --plot --plot_results roc --dataset_name fmnist --arch vgg16
# python main.py --plot --plot_results roc --dataset_name utkface --arch vgg16

# python main.py --plot --plot_results roc --dataset_name purchase --arch mlp
# python main.py --plot --plot_results roc --dataset_name location --arch mlp
# python main.py --plot --plot_results roc --dataset_name adult --arch mlp
# python main.py --plot --plot_results roc --dataset_name texas --arch mlp



python main.py --attack_type 0 --dataset_name location --attack_name apcmia --arch mlp --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name texas --attack_name apcmia --arch mlp --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name purchase --attack_name apcmia --arch mlp --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name adult --attack_name apcmia --arch mlp --apcmia_cluster --apcmia_cluster


python main.py --attack_type 0 --dataset_name fmnist --attack_name apcmia --arch cnn --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name utkface --attack_name apcmia --arch cnn --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name stl10 --attack_name apcmia --arch cnn --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name cifar10 --attack_name apcmia --arch cnn --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name cifar100 --attack_name apcmia --arch cnn --apcmia_cluster --apcmia_cluster


python main.py --attack_type 0 --dataset_name fmnist --attack_name apcmia --arch vgg16 --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name utkface --attack_name apcmia --arch vgg16 --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name stl10 --attack_name apcmia --arch vgg16 --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name cifar10 --attack_name apcmia --arch vgg16 --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name cifar100 --attack_name apcmia --arch vgg16 --apcmia_cluster --apcmia_cluster
python main.py --attack_type 0 --dataset_name fmnist --attack_name apcmia --arch wrn --apcmia_cluster --apcmia_cluster
    



# python main.py --plot --plot_results th --dataset_name cifar100 --arch cnn --attack_name apcmia # this genertes threshold for all
# python main.py --plot --plot_results roc --dataset_name fmnist --arch cnn --attack_name apcmia
# python main.py --plot --plot_results roc --dataset_name location --arch mlp --attack_name apcmia



