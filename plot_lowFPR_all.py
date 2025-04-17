
# # # lowFPR bar pltos for all methods used in the paper vgg16 0.001 fpr--------------
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # ---------------------------
# # # 1. Global Plot Settings
# # # ---------------------------
# # size = 20
# # params = {
# #     'axes.labelsize': size,
# #     'font.size': size,
# #     'legend.fontsize': size,
# #     'xtick.labelsize': size - 5,
# #     'ytick.labelsize': size - 5,
# #     'figure.figsize': [8, 6],   # Adjust as needed
# #     "font.family": "arial",
# # }
# # plt.rcParams.update(params)

# # # ---------------------------
# # # 2. Data Setup
# # # ---------------------------
# # # Dataset labels
# # # labels = ["FMNIST", "UTKFace", "STL-10", "CIFAR-10", "CIFAR-100"]
# # labels = ["CIFAR-10", "STL-10", "CIFAR-100", "UTKFace", "FMNIST"]




# # # for 0.01
# # apcmia = [38.4, 56.4, 20.9, 44.8, 18.1]
# # memia  = [2.3, 18.7, 14.9, 1.6, 2.0]
# # lira   = [6.0, 13.6, 17.7, 12.9, 7.4]
# # seqmia = [2.4, 17.0, 13.0, 1.0, 2.0]
# # mia    = [2.3, 18.1, 12.8, 1.6, 2.0]
# # nsh    = [2.3, 18.7, 13.3, 1.6, 2.1]

# # # # for 0.001
# # # apcmia = [7.1, 21.5, 4.1, 42.0, 3.3]
# # # memia  = [0.2, 2.5, 3.8, 0.1, 0.4]
# # # lira   = [0.5, 1.7, 2.9, 6.1, 2.4]
# # # seqmia = [0.3, 2.4, 2.6, 0.1, 0.4]
# # # mia    = [0.2, 2.5, 3.3, 0.1, 0.4]
# # # nsh    = [0.2, 2.3, 3.3, 0.2, 0.3]



# # # # Convert each value to a percentage (i.e. multiply by 100)
# # # apcmia = [x * 100 for x in apcmia]
# # # lira   = [x * 100 for x in lira]
# # # memia  = [x * 100 for x in memia]
# # # seqmia = [x * 100 for x in seqmia]
# # # nsh    = [x * 100 for x in nsh]
# # # mia    = [x * 100 for x in mia]

# # methods = [apcmia, lira, memia, seqmia, nsh, mia]
# # method_labels = ["apcMIA", "LiRA", "meMIA", "seqMIA", "NSH", "MIA"]
# # colors = ["blue", "green", "red", "orange", "purple", "brown"]

# # # ---------------------------
# # # 3. Grouped Bar Chart
# # # ---------------------------
# # x = np.arange(len(labels))  # positions for each dataset
# # bar_width = 0.13            # width for each bar

# # fig, ax = plt.subplots()

# # # Plot a bar for each method
# # for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
# #     ax.bar(
# #         x + i * bar_width,   # offset each method's bar horizontally
# #         method,
# #         width=bar_width,
# #         label=label,
# #         color=color
# #     )

# # # Set x-axis ticks in the center of the groups
# # ax.set_xticks(x + (len(methods) - 1) * bar_width / 2)
# # ax.set_xticklabels(labels)

# # # Axis labels
# # ax.set_xlabel("Dataset")
# # ax.set_ylabel(f"TPR @ 1% FPR (%)")

# # # Add legend; here we position it above the plot.
# # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# # plt.tight_layout()
# # plt.savefig("bar_plots_vgg16_lowFPR_all_0.01.pdf", dpi=300, bbox_inches="tight")
# # plt.show()




# # # # lowFPR bar pltos for all methods used in the paper CNN 0.01 fpr--------------


# # import numpy as np
# # import matplotlib.pyplot as plt

# # # ---------------------------
# # # 1. Global Plot Settings
# # # ---------------------------
# # size = 20
# # params = {
# #     'axes.labelsize': size,
# #     'font.size': size,
# #     'legend.fontsize': size,
# #     'xtick.labelsize': size - 5,
# #     'ytick.labelsize': size - 5,
# #     'figure.figsize': [8, 6],   # Adjust as needed
# #     "font.family": "arial",
# # }
# # plt.rcParams.update(params)

# # # ---------------------------
# # # 2. Data Setup
# # # ---------------------------
# # # Dataset labels
# # labels = ["CIFAR-10", "STL-10", "CIFAR-100", "UTKFace", "FMNIST"]

# # # Attack accuracies in decimal format (as pro
# # # apcmia =  [0.157, 0.443, 0.494, 0.396, 0.221]
# # # lira   =  [0.074, 0.129, 0.136, 0.06,  0.177]
# # # memia  =  [0.02,  0.016, 0.187, 0.023, 0.149]
# # # seqmia =  [0.02,  0.01,  0.17,  0.024, 0.13]
# # # nsh    =  [0.021, 0.016, 0.187, 0.023, 0.133]
# # # mia    =  [0.02,  0.016, 0.181, 0.023, 0.128]




# # # # # for 0.01

# # # apcmia = [20.1, 39.1, 32.5, 46.0, 19.5]
# # # memia  = [4.9, 11.2, 17.5, 0.6, 0.9]
# # # lira   = [7.8, 5.3, 23.1, 5.6, 3.1]
# # # seqmia = [4.2, 7.6, 11.3, 0.8, 0.0]
# # # mia    = [4.1, 8.1, 12.1, 0.0, 0.9]
# # # nsh    = [5.0, 9.5, 16.0, 0.0, 0.8]




# # # for 0.001


# # apcmia = [2.1, 0.0, 7.5, 45.9, 0.0]
# # memia  = [0.3, 1.5, 3.5, 0.0, 0.0]
# # lira   = [1.0, 0.5, 3.5, 1.3, 0.4]
# # seqmia = [0.0, 1.8, 1.6, 0.2, 0.0]
# # mia    = [0.3, 1.3, 2.3, 0.0, 0.0]
# # nsh    = [0.3, 1.5, 3.5, 0.0, 0.0]


# # # Convert each value to a percentage (i.e. multiply by 100)
# # # apcmia = [x * 100 for x in apcmia]
# # # lira   = [x * 100 for x in lira]
# # # memia  = [x * 100 for x in memia]
# # # seqmia = [x * 100 for x in seqmia]
# # # nsh    = [x * 100 for x in nsh]
# # # mia    = [x * 100 for x in mia]

# # methods = [apcmia, lira, memia, seqmia, nsh, mia]
# # method_labels = ["apcMIA", "LiRA", "meMIA", "seqMIA", "NSH", "MIA"]
# # colors = ["blue", "green", "red", "orange", "purple", "brown"]

# # # ---------------------------
# # # 3. Grouped Bar Chart
# # # ---------------------------
# # x = np.arange(len(labels))  # positions for each dataset
# # bar_width = 0.13            # width for each bar

# # fig, ax = plt.subplots()

# # # Plot a bar for each method
# # for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
# #     ax.bar(
# #         x + i * bar_width,   # offset each method's bar horizontally
# #         method,
# #         width=bar_width,
# #         label=label,
# #         color=color
# #     )

# # # Set x-axis ticks in the center of the groups
# # ax.set_xticks(x + (len(methods) - 1) * bar_width / 2)
# # ax.set_xticklabels(labels)

# # # Axis labels
# # ax.set_xlabel("Dataset")
# # ax.set_ylabel(f"TPR @ 0.1% FPR (%)")

# # # Add legend; here we position it above the plot.
# # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# # plt.tight_layout()
# # plt.savefig("bar_plots_cnn_lowFPR_all_0.001.pdf", dpi=300, bbox_inches="tight")
# # plt.show()




# # # lowFPR bar pltos for all methods used in the paper mlp 0.01 fpr--------------
# import numpy as np
# import matplotlib.pyplot as plt

# # ---------------------------
# # 1. Global Plot Settings
# # ---------------------------
# size = 20
# params = {
#     'axes.labelsize': size,
#     'font.size': size,
#     'legend.fontsize': size,
#     'xtick.labelsize': size - 5,
#     'ytick.labelsize': size - 5,
#     'figure.figsize': [8, 6],   # Adjust as needed
#     "font.family": "arial",
# }
# plt.rcParams.update(params)

# # ---------------------------
# # 2. Data Setup
# # ---------------------------
# labels = ["Location", "Purchase-100", "Texas-100", "Adult"]

# # Attack accuracies (decimals), one list per method
# # apcmia =  [0.008,  0.0045, 0.004,  0.014]
# # lira   =  [0.006,  0.002,  0.002,  0.001]
# # memia  =  [0.004,  0.0,    0.009,  0.0]
# # seqmia =  [0.0,    0.002,  0.002,  0.0]
# # nsh    =  [0.001,  0.0,    0.006,  0.0]
# # mia    =  [0.001,  0.0,    0.003,  0.0]


# # datasets = ["Location", "Purchase-100", "Texas-100", "Adult"]

# apcmia = [0.038, 0.043, 0.043, 0.084]
# lira   = [0.03,  0.017, 0.03,  0.016]
# memia  = [0.013, 0.014, 0.036, 0.011]
# seqmia = [0.012, 0.015, 0.026, 0.0]
# nsh    = [0.011, 0.014, 0.033, 0.011]
# mia    = [0.011, 0.016, 0.024, 0.011]

# for 1%
# apcmia = [6.7, 21.1, 11.2, 4.0]
# memia  = [2.9, 1.1, 3.6, 1.4]
# lira   = [3.0, 1.6, 3.0, 1.7]
# seqmia = [1.1, 0.0, 2.6, 1.5]
# mia    = [1.3, 1.1, 2.4, 1.6]
# nsh    = [3.4, 1.1, 3.3, 1.4]

# for 0.1% 
# apcmia = [2.8, 13.7, 1.0, 0.0]
# memia  = [0.4, 0.0, 0.9, 0.0]
# lira   = [0.6, 0.1, 0.2, 0.2]
# seqmia = [0.2, 0.0, 0.2, 0.2]
# mia    = [0.4, 0.0, 0.3, 0.0]
# nsh    = [0.6, 0.0, 0.6, 0.0]



# # Convert each value to a percentage (multiply by 100)
# apcmia = [x * 100 for x in apcmia]
# lira   = [x * 100 for x in lira]
# memia  = [x * 100 for x in memia]
# seqmia = [x * 100 for x in seqmia]
# nsh    = [x * 100 for x in nsh]
# mia    = [x * 100 for x in mia]

# methods = [apcmia, lira, memia, seqmia, nsh, mia]
# method_labels = ["apcMIA", "LiRA", "meMIA", "seqMIA", "NSH", "MIA"]
# colors = ["blue", "green", "red", "orange", "purple", "brown"]

# # ---------------------------
# # 3. Grouped Bar Chart
# # ---------------------------
# x = np.arange(len(labels))  # positions for each dataset
# bar_width = 0.13            # width of each bar

# fig, ax = plt.subplots()

# for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
#     ax.bar(
#         x + i * bar_width,
#         method,
#         width=bar_width,
#         label=label,
#         color=color
#     )

# # Set x-axis ticks in the center of the grouped bars
# ax.set_xticks(x + (len(methods) - 1) * bar_width / 2)
# ax.set_xticklabels(labels)

# # Axis labels
# ax.set_xlabel("Dataset")
# # ax.set_ylabel("Attack Accuracy (%)")
# ax.set_ylabel(f"TPR @ 1% FPR (%)")

# # Add legend; place it above the plot
# ax.legend(
#     loc='upper center',
#     bbox_to_anchor=(0.5, -0.15),
#     ncol=3
# )

# plt.tight_layout()

# # ---------------------------
# # 4. Save and Show
# # ---------------------------
# plt.savefig("bar_plots_mlp_lowFPR_all_0.01.pdf", dpi=300, bbox_inches="tight")
# plt.show()

