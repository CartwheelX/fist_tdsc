# # # from datetime import datetime


# # # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # print("Current timestamp:", timestamp)
# # # exit()

# # # import matplotlib.pyplot as plt

# # # # Update rcParams as specified
# # # size = 20
# # # params = {
# # #    'axes.labelsize': size,
# # #    'font.size': size,
# # #    'legend.fontsize': size,
# # #    'xtick.labelsize': size-5,
# # #    'ytick.labelsize': size-5,
# # #    # 'text.usetex': False,
# # #    'figure.figsize': [10, 8],
# # #    "font.family": "arial",
# # # }
# # # plt.rcParams.update(params)

# # # # Data from your table
# # # model_acc = [0.643, 0.587, 0.465, 0.304]

# # # apcmMIA_acc = [0.792, 0.755, 0.738, 0.725]
# # # LiRa_acc    = [0.69,  0.681, 0.661, 0.662]
# # # meMIA_acc   = [0.68,  0.643, 0.622, 0.577]
# # # seqMIA_acc  = [0.633, 0.57,  0.515, 0.511]
# # # NSH_acc     = [0.673, 0.654, 0.608, 0.541]
# # # MIA_acc     = [0.637, 0.595, 0.533, 0.512]

# # # # Plot each attack's accuracy vs. model accuracy with specified modifications:
# # # plt.plot(model_acc, apcmMIA_acc, marker='s', color='blue', label='apcMIA', linewidth=2)
# # # plt.plot(model_acc, LiRa_acc,    marker='s', color='green',    label='LiRA', linewidth=2)
# # # plt.plot(model_acc, meMIA_acc,   marker='s', color='red',      label='meMIA', linewidth=2)
# # # plt.plot(model_acc, seqMIA_acc,  marker='s', color='orange',   label='seqMIA', linewidth=2)
# # # plt.plot(model_acc, NSH_acc,     marker='s', color='purple',   label='NSH', linewidth=2)
# # # plt.plot(model_acc, MIA_acc,     marker='s', color='brown',    label='MIA', linewidth=2)

# # # plt.xlabel("Model Accuracy")
# # # plt.ylabel("Attack Accuracy")

# # # # Place legend below the plot
# # # plt.legend(
# # #     loc='lower center',          # legend in lower center
# # #     bbox_to_anchor=(0.5, -0.3),    # adjust vertical position as needed
# # #     ncol=3,                      # arrange legend entries in one row
# # #     fontsize=size
# # # )
# # # plt.tight_layout()

# # # # Save the figure as a PDF with 300 dpi
# # # plt.savefig("attack_vs_model_acc.pdf", dpi=300, bbox_inches="tight")
# # # plt.show()

# # # # # THIS IS USED IN THE FINAL REPORT
# import matplotlib.pyplot as plt

# # # -----------------------------------------------------
# # # 1) Global Plot Settings
# # # -----------------------------------------------------
# # size = 20
# # params = {
# #    'axes.labelsize': size,
# #    'font.size': size,
# #    'legend.fontsize': size,
# #    'xtick.labelsize': size - 5,
# #    'ytick.labelsize': size - 5,
# #    'figure.figsize': [10, 10],
# #    "font.family": "arial",
# # }
# size = 25
# params = {
#     'axes.labelsize': size,
#     'font.size': size,
#     'legend.fontsize': size,
#     'xtick.labelsize': size,
#     'ytick.labelsize': size,
#     'figure.figsize': [10, 9],
#     "font.family": "arial",
# }


# plt.rcParams.update(params)

# # # Define a color palette for consistency
# # attack_colors = {
# #     "apcmia": "#0d0478",   # Blue
# #     "mia":    "#9467bd",   # Red
# #     "seqmia": "#2ca02c",   # Green
# #     "memia":  "#d62728",   # Purple
# #     "nsh":    "#ff7f0e",   # Orange
# #     "lira":   "#8c564b"    # Brownish
# # }


# # plt.rcParams.update(params)

# # -----------------------------------------------------
# # 2) Data
# # -----------------------------------------------------
# sigma = [0.2, 0.3, 0.5, 1.0]

# # apcmMIA_acc   = [0.792, 0.755, 0.738, 0.725]
# # LiRa_acc      = [0.66,  0.661, 0.661, 0.662]
# # meMIA_acc     = [0.68,  0.643, 0.622, 0.577]
# # seqMIA_acc    = [0.633, 0.57,  0.515, 0.511]
# # NSH_acc       = [0.673, 0.654, 0.608, 0.541]
# # MIA_acc       = [0.637, 0.595, 0.533, 0.512]

# apcmMIA_acc = [0.792, 0.755, 0.738, 0.725]
# LiRa_acc    = [0.69,  0.681, 0.661, 0.662]
# meMIA_acc   = [0.68,  0.643, 0.622, 0.577]
# seqMIA_acc  = [0.633, 0.57,  0.515, 0.511]
# NSH_acc     = [0.673, 0.654, 0.608, 0.541]
# MIA_acc     = [0.637, 0.595, 0.533, 0.512]
# apcmMIA_acc_rev = apcmMIA_acc[::-1]   # [0.725, 0.738, 0.755, 0.792]
# LiRa_acc_rev    = LiRa_acc[::-1]      # [0.662, 0.661, 0.681, 0.69]
# meMIA_acc_rev   = meMIA_acc[::-1]     # [0.577, 0.622, 0.643, 0.68]
# seqMIA_acc_rev  = seqMIA_acc[::-1]    # [0.511, 0.515, 0.57, 0.633]
# NSH_acc_rev     = NSH_acc[::-1]       # [0.541, 0.608, 0.654, 0.673]
# MIA_acc_rev     = MIA_acc[::-1]       # [0.512, 0.533, 0.595, 0.637]
# # -----------------------------------------------------
# # 3) Plotting
# # -----------------------------------------------------
# # plt.plot(sigma, model_acc,    marker='o', color='black',   linewidth=2, label='Model')

# # attack_markers = {
# #     "apcMIA": "o",
# #     "meMIA":  "^",
# #     "LiRA":   "*",
# #     "seqMIA": "D",
# #     "MIA":    "s",
# #     "NSH":    "v"
# # }
# plt.plot(sigma, apcmMIA_acc,  marker='o', color='blue',linewidth=3, label='apcMIA')
# plt.plot(sigma, LiRa_acc,     marker='^', color='green',   linewidth=3, label='LiRA')
# plt.plot(sigma, meMIA_acc,    marker='D', color='red',     linewidth=3, label='meMIA')
# plt.plot(sigma, seqMIA_acc,   marker='s', color='orange',  linewidth=3, label='seqMIA')
# plt.plot(sigma, NSH_acc,      marker='v', color='purple',  linewidth=3, label='NSH')
# plt.plot(sigma, MIA_acc,      marker='<', color='brown',   linewidth=3, label='MIA')

# plt.xlabel(r"Noise Multiplier ($\sigma$)")          # X-axis: sigma
# plt.ylabel("Accuracy")           # Y-axis: accuracy
# # No grid or title (remove these lines if you want them)

# # -----------------------------------------------------
# # 4) Legend Below the Plot
# # -----------------------------------------------------
# plt.legend(
#     loc='lower center',
#     bbox_to_anchor=(0.5, -0.45),  # shift legend below the plot
#     ncol=3                       # number of columns in legend
# )

# plt.tight_layout(rect=[0, 0, 1, 0.95])

# # -----------------------------------------------------
# # 5) Save and Show
# # -----------------------------------------------------
# plt.savefig("sigma_vs_accuracy.pdf", dpi=300, bbox_inches="tight")
# plt.show()


# # # # # -----------------------------------------------------GRUOPED BAR CHART-----------------------------------------------------
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt

# # # # # Global rcParams settings
# # # # size = 20
# # # # params = {
# # # #    'axes.labelsize': size,
# # # #    'font.size': size,
# # # #    'legend.fontsize': size,
# # # #    'xtick.labelsize': size-5,
# # # #    'ytick.labelsize': size-5,
# # # #    'figure.figsize': [10, 8],
# # # #    "font.family": "arial",
# # # # }
# # # # plt.rcParams.update(params)

# # # # # Data from your table
# # # # sigma = np.array([0.2, 0.3, 0.5, 1.0])
# # # model_acc     = np.array([0.643, 0.587, 0.465, 0.304])
# # # apcMIA_acc   = np.array([0.792, 0.755, 0.738, 0.725])
# # # LiRa_acc      = np.array([0.69,  0.681, 0.661, 0.662])
# # # meMIA_acc     = np.array([0.68,  0.643, 0.622, 0.577])
# # # seqMIA_acc    = np.array([0.633, 0.57,  0.515, 0.511])
# # # NSH_acc       = np.array([0.673, 0.654, 0.608, 0.541])
# # # MIA_acc       = np.array([0.637, 0.595, 0.533, 0.512])

# # # # All methods (excluding the target model accuracy)
# # # methods = [apcMIA_acc, LiRa_acc, meMIA_acc, seqMIA_acc, NSH_acc, MIA_acc]
# # # method_labels = ['apcMIA', 'LiRa', 'meMIA', 'seqMIA', 'NSH', 'MIA']
# # # colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

# # # # Set up the grouped bar chart
# # # x = np.arange(len(sigma))  # positions for each sigma value
# # # width = 0.12  # width of each bar

# # # fig, ax = plt.subplots()
# # # for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
# # #     ax.bar(x + (i - 2.5)*width, method, width, label=label, color=color)

# # # # Optionally, you can also plot the model accuracy as a line (or as another bar group)
# # # ax.plot(x, model_acc, marker='o', color='black', linewidth=2, label='Model Accuracy')

# # # # Setting axis labels and ticks
# # # ax.set_xlabel(r'Noise Multiplier ($\sigma$)', fontsize=size)
# # # ax.set_ylabel("Accuracy", fontsize=size)
# # # ax.set_xticks(x)
# # # ax.set_xticklabels(sigma)
# # # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.33), ncol=4)
# # # plt.tight_layout()
# # # plt.savefig("grouped_bar_sigma_accuracy.pdf", dpi=300, bbox_inches="tight")
# # # plt.show()

# # # # ------------------------------------------------------ Heatmap ------------------------------------------------------
# # # This code creates a heatmap of attack accuracies against different noise multipliers (sigma values).
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Data arrays
# sigma = [0.2, 0.3, 0.5, 1.0]

# apcMIA_acc   = [0.792, 0.755, 0.738, 0.725]
# LiRa_acc     = [0.660, 0.661, 0.661, 0.662]
# meMIA_acc    = [0.680, 0.643, 0.622, 0.577]
# seqMIA_acc   = [0.633, 0.570, 0.515, 0.511]
# NSH_acc      = [0.673, 0.654, 0.608, 0.541]
# MIA_acc      = [0.637, 0.595, 0.533, 0.512]

# # Data matrix: rows = methods, columns = sigmas
# data = np.array([
#     apcMIA_acc,
#     LiRa_acc,
#     meMIA_acc,
#     seqMIA_acc,
#     NSH_acc,
#     MIA_acc
# ])

# row_labels = ['apcMIA', 'LiRA', 'meMIA', 'seqMIA', 'NSH', 'MIA']
# col_labels = [str(s) for s in sigma]

# # Plot settings
# size = 30
# params = {
#     'axes.labelsize': size,
#     'font.size': size,
#     'legend.fontsize': size,
#     'xtick.labelsize': size,
#     'ytick.labelsize': size,
#     'figure.figsize': [10, 8],
#     'font.family': 'arial',
# }
# plt.rcParams.update(params)

# # Plot heatmap
# fig, ax = plt.subplots()
# sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu",
#             xticklabels=col_labels, yticklabels=row_labels, ax=ax)

# # Label axes
# ax.set_xlabel(r'Noise Multiplier ($\sigma$)')
# ax.set_ylabel("Attack Method")

# # Rotate Y-axis tick labels horizontally
# ax.set_yticklabels(ax.get_yticklabels(), rotation=45)

# # Save and show
# plt.tight_layout()
# plt.savefig("heatmap_sigma_accuracy.pdf", dpi=300, bbox_inches="tight")
# plt.show()
# # # import matplotlib.pyplot as plt
# # import numpy as np

# # # Global settings
# # size = 20
# # params = {
# #    'axes.labelsize': size,
# #    'font.size': size,
# #    'legend.fontsize': size,
# #    'xtick.labelsize': size-5,
# #    'ytick.labelsize': size-5,
# #    'figure.figsize': [10, 8],
# #    "font.family": "arial",
# # }
# # plt.rcParams.update(params)

# # # Data
# # sigma = np.array([0.2, 0.3, 0.5, 1.0])
# # apcMIA_acc = np.array([0.792, 0.755, 0.738, 0.725])
# # LiRa_acc    = np.array([0.66,  0.661, 0.661, 0.662])
# # meMIA_acc   = np.array([0.68,  0.643, 0.622, 0.577])
# # seqMIA_acc  = np.array([0.633, 0.57,  0.515, 0.511])
# # NSH_acc     = np.array([0.673, 0.654, 0.608, 0.541])
# # MIA_acc     = np.array([0.637, 0.595, 0.533, 0.512])

# # # Compute average of the other attacks
# # others_avg = (LiRa_acc + meMIA_acc + seqMIA_acc + NSH_acc + MIA_acc) / 5.0

# # # Compute percentage improvement: (apcMIA/others_avg - 1) * 100
# # perc_improvement = (apcMIA_acc / others_avg - 1) * 100

# # plt.plot(sigma, perc_improvement, marker='o', linestyle='-', color='darkblue', linewidth=2)
# # plt.xlabel(r"$\sigma$")
# # plt.ylabel("Percentage Improvement (%)")
# # plt.title("Percentage Improvement of apcMIA over Other Attacks")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("percentage_improvement.pdf", dpi=300, bbox_inches="tight")
# # plt.show()


# # import numpy as np
# # import matplotlib.pyplot as plt


# # # size = 20
# # # params = {
# # #    'axes.labelsize': size,
# # #    'font.size': size,
# # #    'legend.fontsize': size,
# # #    'xtick.labelsize': size-5,
# # #    'ytick.labelsize': size-5,
# # #    'figure.figsize': [10, 8],
# # #    "font.family": "arial",
# # # }

# # # plt.rcParams.update(params)

# # # sigma = np.array([0.2, 0.3, 0.5, 1.0])
# # # apcMIA_acc = np.array([0.792, 0.755, 0.738, 0.725])
# # # LiRa_acc    = np.array([0.66,  0.661, 0.661, 0.662])
# # # meMIA_acc   = np.array([0.68,  0.643, 0.622, 0.577])
# # # seqMIA_acc  = np.array([0.633, 0.57,  0.515, 0.511])
# # # NSH_acc     = np.array([0.673, 0.654, 0.608, 0.541])
# # # MIA_acc     = np.array([0.637, 0.595, 0.533, 0.512])

# # # # Compute average of other methods
# # # others_avg = (LiRa_acc + meMIA_acc + seqMIA_acc + NSH_acc + MIA_acc) / 5.0

# # # x = np.arange(len(sigma))  # positions for each sigma value
# # # width = 0.25

# # # fig, ax = plt.subplots()
# # # ax.bar(x - width/2, apcMIA_acc, width, color='blue', label='apcMIA')
# # # ax.bar(x + width/2, others_avg, width, color='green', label='Average Others')

# # # ax.set_ylim(0, 1)
# # # ax.set_xticks(x)
# # # ax.set_xticklabels(sigma)
# # # ax.set_xlabel(r"Noise Multiplier ($\sigma$)")
# # # ax.set_ylabel("Attack Accuracy")
# # # ax.legend(loc='upper left')
# # # # plt.title("Grouped Bar Chart: apcMIA vs. Average Others")
# # # plt.tight_layout()
# # # plt.savefig("grouped_bar_sigma_accuracy.pdf", dpi=300, bbox_inches="tight")
# # # plt.show()




# # lowFPR bar pltos for all methods used in the paper vgg16 0.001 fpr--------------
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
# # Dataset labels
# labels = ["FMNIST", "UTKFace", "STL-10", "CIFAR-10", "CIFAR-100"]

# # Attack accuracies in decimal format (as provided)
# apcmia =  [0.03,  0.414, 0.101, 0.062, 0.056]
# lira   =  [0.024, 0.061, 0.017, 0.005, 0.029]
# memia  =  [0.004, 0.001, 0.025, 0.002, 0.038]
# seqmia =  [0.004, 0.001, 0.024, 0.003, 0.026]
# nsh    =  [0.003, 0.002, 0.023, 0.002, 0.033]
# mia    =  [0.004, 0.002, 0.025, 0.002, 0.033]



# # Convert each value to a percentage (i.e. multiply by 100)
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
# bar_width = 0.13            # width for each bar

# fig, ax = plt.subplots()

# # Plot a bar for each method
# for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
#     ax.bar(
#         x + i * bar_width,   # offset each method's bar horizontally
#         method,
#         width=bar_width,
#         label=label,
#         color=color
#     )

# # Set x-axis ticks in the center of the groups
# ax.set_xticks(x + (len(methods) - 1) * bar_width / 2)
# ax.set_xticklabels(labels)

# # Axis labels
# ax.set_xlabel("Dataset")
# ax.set_ylabel(f"TPR @ 0.1% FPR (%)")

# # Add legend; here we position it above the plot.
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# plt.tight_layout()
# plt.savefig("bar_plots_vgg16_lowFPR_all_0.001.pdf", dpi=300, bbox_inches="tight")
# plt.show()



# # # lowFPR bar pltos for all methods used in the paper vgg 0.01 fpr--------------
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
# # labels = ["FMNIST", "UTKFace", "STL-10", "CIFAR-10", "CIFAR-100"]

# # # Attack accuracies in decimal format (as pro
# # apcmia =  [0.157, 0.443, 0.494, 0.396, 0.221]
# # lira   =  [0.074, 0.129, 0.136, 0.06,  0.177]
# # memia  =  [0.02,  0.016, 0.187, 0.023, 0.149]
# # seqmia =  [0.02,  0.01,  0.17,  0.024, 0.13]
# # nsh    =  [0.021, 0.016, 0.187, 0.023, 0.133]
# # mia    =  [0.02,  0.016, 0.181, 0.023, 0.128]

# # # Convert each value to a percentage (i.e. multiply by 100)
# # apcmia = [x * 100 for x in apcmia]
# # lira   = [x * 100 for x in lira]
# # memia  = [x * 100 for x in memia]
# # seqmia = [x * 100 for x in seqmia]
# # nsh    = [x * 100 for x in nsh]
# # mia    = [x * 100 for x in mia]

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


# # # lowFPR bar pltos for all methods used in the paper CNN 0.01 fpr--------------


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
# # Dataset labels
# labels = ["FMNIST", "UTKFace", "STL-10", "CIFAR-10", "CIFAR-100"]

# # Attack accuracies in decimal format (as pro
# # apcmia =  [0.157, 0.443, 0.494, 0.396, 0.221]
# # lira   =  [0.074, 0.129, 0.136, 0.06,  0.177]
# # memia  =  [0.02,  0.016, 0.187, 0.023, 0.149]
# # seqmia =  [0.02,  0.01,  0.17,  0.024, 0.13]
# # nsh    =  [0.021, 0.016, 0.187, 0.023, 0.133]
# # mia    =  [0.02,  0.016, 0.181, 0.023, 0.128]

# apcmia = [0.005,   0.21, 0.017,   0.092, 0.073]
# lira   = [0.004, 0.013, 0.005, 0.01,  0.035]
# memia  = [0.0,   0.0,   0.015, 0.003, 0.035]
# seqmia = [0.0,   0.002, 0.018, 0.0,   0.016]
# nsh    = [0.0,   0.0,   0.015, 0.003, 0.035]
# mia    = [0.0,   0.0,   0.013, 0.003, 0.023]

# for 0.01%
# apcMIA	0.021	0	0.075	0.459	0
# meMIA	0.003	0.015	0.035	0	0
# LiRa	0.01	0.005	0.035	0.013	0.004
# seqMIA	0	0.018	0.016	0.002	0
# MIA	0.003	0.013	0.023	0	0
# NSH	0.003	0.015	0.035	0	0

# for 1%

# apcMIA	0.201	0.391	0.325	0.46	0
# meMIA	0.049	0.112	0.175	0.006	0
# LiRa	0.078	0.053	0.231	0.056	0.004
# seqMIA	0.042	0.076	0.113	0.008	0
# MIA	0.041	0.081	0.121	0	0
# NSH	0.05	0.095	0.16	0	0




# # Convert each value to a percentage (i.e. multiply by 100)
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
# bar_width = 0.13            # width for each bar

# fig, ax = plt.subplots()

# # Plot a bar for each method
# for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
#     ax.bar(
#         x + i * bar_width,   # offset each method's bar horizontally
#         method,
#         width=bar_width,
#         label=label,
#         color=color
#     )

# # Set x-axis ticks in the center of the groups
# ax.set_xticks(x + (len(methods) - 1) * bar_width / 2)
# ax.set_xticklabels(labels)

# # Axis labels
# ax.set_xlabel("Dataset")
# ax.set_ylabel(f"TPR @ 0.1% FPR (%)")

# # Add legend; here we position it above the plot.
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# plt.tight_layout()
# plt.savefig("bar_plots_cnn_lowFPR_all_0.001.pdf", dpi=300, bbox_inches="tight")
# plt.show()


# # # lowFPR bar pltos for all methods used in the paper cnn 0.01 fpr--------------
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
# # labels = ["FMNIST", "UTKFace", "STL-10", "CIFAR-10", "CIFAR-100"]





# # apcmia = [0.197, 0.21, 0.394, 0.296, 0.312]
# # lira   = [0.031, 0.056, 0.053, 0.078, 0.231]
# # memia  = [0.009, 0.006, 0.112, 0.049, 0.175]
# # seqmia = [0.0,   0.008, 0.076, 0.042, 0.113]
# # nsh    = [0.008, 0.0,   0.095, 0.05,  0.16]
# # mia    = [0.009, 0.0,   0.081, 0.041, 0.121]

# # # Convert each value to a percentage (i.e. multiply by 100)
# # apcmia = [x * 100 for x in apcmia]
# # lira   = [x * 100 for x in lira]
# # memia  = [x * 100 for x in memia]
# # seqmia = [x * 100 for x in seqmia]
# # nsh    = [x * 100 for x in nsh]
# # mia    = [x * 100 for x in mia]

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
# # plt.savefig("bar_plots_cnn_lowFPR_all_0.01.pdf", dpi=300, bbox_inches="tight")
# # plt.show()


# # # lowFPR bar pltos for all methods used in the paper mlp 0.001 fpr--------------
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
# # labels = ["Location", "Purchase-100", "Texas-100", "Adult"]

# # # Attack accuracies (decimals), one list per method
# # apcmia =  [0.008,  0.0045, 0.004,  0.014]
# # lira   =  [0.006,  0.002,  0.002,  0.001]
# # memia  =  [0.004,  0.0,    0.009,  0.0]
# # seqmia =  [0.0,    0.002,  0.002,  0.0]
# # nsh    =  [0.001,  0.0,    0.006,  0.0]
# # mia    =  [0.001,  0.0,    0.003,  0.0]


# # # Convert each value to a percentage (multiply by 100)
# # apcmia = [x * 100 for x in apcmia]
# # lira   = [x * 100 for x in lira]
# # memia  = [x * 100 for x in memia]
# # seqmia = [x * 100 for x in seqmia]
# # nsh    = [x * 100 for x in nsh]
# # mia    = [x * 100 for x in mia]

# # methods = [apcmia, lira, memia, seqmia, nsh, mia]
# # method_labels = ["apcMIA", "LiRA", "meMIA", "seqMIA", "NSH", "MIA"]
# # colors = ["blue", "green", "red", "orange", "purple", "brown"]

# # # ---------------------------
# # # 3. Grouped Bar Chart
# # # ---------------------------
# # x = np.arange(len(labels))  # positions for each dataset
# # bar_width = 0.13            # width of each bar

# # fig, ax = plt.subplots()

# # for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
# #     ax.bar(
# #         x + i * bar_width,
# #         method,
# #         width=bar_width,
# #         label=label,
# #         color=color
# #     )

# # # Set x-axis ticks in the center of the grouped bars
# # ax.set_xticks(x + (len(methods) - 1) * bar_width / 2)
# # ax.set_xticklabels(labels)

# # # Axis labels
# # ax.set_xlabel("Dataset")
# # # ax.set_ylabel("Attack Accuracy (%)")
# # ax.set_ylabel(f"TPR @ 0.1% FPR (%)")

# # # Add legend; place it above the plot
# # ax.legend(
# #     loc='upper center',
# #     bbox_to_anchor=(0.5, -0.15),
# #     ncol=3
# # )

# # plt.tight_layout()

# # # ---------------------------
# # # 4. Save and Show
# # # ---------------------------
# # plt.savefig("bar_plots_mlp_lowFPR_all_0.001.pdf", dpi=300, bbox_inches="tight")
# # plt.show()



# # lowFPR bar pltos for all methods used in the paper mlp 0.01 fpr--------------
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
# apcMIA	0.067	0.211	0.112	0.04
# meMIA	0.029	0.011	0.036	0.014
# LiRa	0.03	0.016	0.03	0.017
# seqMIA	0.011	0	0.026	0.015
# MIA	0.013	0.011	0.024	0.016
# NSH	0.034	0.011	0.033	0.014

# for 0.1% 
# apcMIA	0.028	0.137	0.01	0
# meMIA	0.004	0	0.009	0
# LiRa	0.006	0.001	0.002	0.002
# seqMIA	0.002	0	0.002	0.002
# MIA	0.004	0	0.003	0
# NSH	0.006	0	0.006	0



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


# # # Plotting ROC curves from Excel files ------------------------------------------------Abalation Study for VGG16 on CIFAR-10 variying training size
# # # Explicit list of filenames and their corresponding training size labels
# # # Update rcParams as specified

# # import os
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # size = 20
# # params = {
# #     'axes.labelsize': size,
# #     'font.size': size,
# #     'legend.fontsize': size,
# #     'xtick.labelsize': size - 5,
# #     'ytick.labelsize': size - 5,
# #     'figure.figsize': [10, 8],
# #     "font.family": "arial",
# # }
# # plt.rcParams.update(params)

# # # Manually specified list of Excel files and their training sizes.
# # files = [
# #     ("cifar10_roc_curves_vgg16_10000.xlsx", 10000),
# #     ("cifar10_roc_curves_vgg16_15000.xlsx", 15000),
# #     ("cifar10_roc_curves_vgg16_20000.xlsx", 20000),
# #     ("cifar10_roc_curves_vgg16_25000.xlsx", 25000),
# #     ("cifar10_roc_curves_vgg16_30000.xlsx", 30000),
# # ]

# # plt.figure()

# # for filename, train_size in files:
# #     try:
# #         df = pd.read_excel(filename, sheet_name="ROC_Curves")
# #     except Exception as e:
# #         print(f"Error reading {filename}: {e}")
# #         continue

# #     if "FPR" not in df.columns or "TPR" not in df.columns:
# #         print(f"File {filename} is missing required columns, skipping.")
# #         continue

# #     fpr = df["FPR"].values
# #     tpr = df["TPR"].values

# #     plt.plot(fpr, tpr, label=str(train_size), linewidth=2)

# # plt.xlabel("False Positive Rate")
# # plt.ylabel("True Positive Rate")
# # # plt.title("ROC Curves for CIFAR10 (VGG16)")
# # # Create a legend with a common title "Training Size"
# # legend = plt.legend(loc="lower right", title="Training Size")
# # legend.get_frame().set_edgecolor("0.91")
# # plt.plot([0, 1], [0, 1], ls="--", color="gray", label="Random Classifier")

# # plt.subplots_adjust(left=0.60)
# # plt.tight_layout()
# # plt.semilogx()
# # plt.semilogy()
# # plt.xlim(1e-5, 1)
# # plt.ylim(1e-5, 1.01)
# # plt.savefig("vgg16_cifar10_varying_training_size.pdf", dpi=300, bbox_inches="tight")
# # plt.show()


# # import matplotlib.pyplot as plt
# # import numpy as np



# # """
# # Plots 4 subplots comparing two learning-rate selection strategies or approaches:
# # sigmoid-based rate vs. fixed learning rate.
# # - Subplot 1: Accuracy Gap (sigmoid-based rate only)
# # - Subplot 2: Accuracy (sigmoid-based rate vs. fixed learning rate)
# # - Subplot 3: TPR @ 0.1% FPR (sigmoid-based rate vs. fixed learning rate)
# # - Subplot 4: TPR @ 1% FPR (sigmoid-based rate vs. fixed learning rate)
# # """

# # # -------------------------
# # # 1) Hardcode your data
# # # -------------------------
# # train_sizes = np.array([100, 200, 300, 400, 500, 600, 1000, 2000, 5000, 10000])

# # # sigmoid-based rate (e.g., "sigmoid-based scheduling")
# # acc_gap_A =   [0.79,    0.705,   0.736667, 0.6875,   0.684,   0.655,  0.629,   0.5605,  0.4702,  0.3923]
# # acc_A =       [0.978627451, 0.939215686, 0.928630037, 0.936666667, 0.9296, 0.950787402,
# #                 0.941964286, 0.929739317, 0.918939616, 0.899962447]
# # tpr01_A =     [0.029411765, 0.1,         0.169230769, 0.512195122, 0.142857143, 0.244186047,
# #                 0.136160714, 0.099104794, 0.09772423,  0.099075961]
# # tpr1_A =      [0.029411765, 0.1,         0.184615385, 0.597560976, 0.339285714, 0.329457364,
# #                 0.3175,      0.201, 0.29, 0.363663986]

# # # fixed learning rate (e.g., "fixed learning rate")
# # acc_B =       [0.968627451, 0.909803922, 0.926630037, 0.928417761, 0.925642857,   0.956723433,
# #                 0.9375,      0.926264457, 0.906434033, 0.898635223]
# # tpr01_B =     [0.029411765, 0.083333333, 0.169230769, 0.12195122,  0.049107143, 0.313953488,
# #                 0.145089286, 0.098104794, 0.042391789, 0.09562109 ]
# # tpr1_B =      [0.029411765, 0.083333333, 0.246153846, 0.408536585, 0.102678571, 0.46124031,
# #                 0.276785714, 0.191750279, 0.242748773, 0.353663986]

# # # -------------------------
# # # 2) Plot Settings
# # # -------------------------
# # size = 20
# # params = {
# #     'axes.labelsize': size,
# #     'font.size': size,
# #     'legend.fontsize': size,
# #     'xtick.labelsize': size,
# #     'ytick.labelsize': size,
# #     'figure.figsize': [10, 8],
# #     "font.family": "arial",
# # }
# # plt.rcParams.update(params)

# # # Create a 2x2 grid of subplots
# # fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False)
# # ax_gap, ax_acc, ax_tpr01, ax_tpr1 = axes.flatten()

# # # -------------------------
# # # 3) Subplot 1: Accuracy Gap (sigmoid-based rate only)
# # # -------------------------
# # ax_gap.plot(train_sizes, acc_gap_A, marker='o', color='blue', label="Acc Gap (A)")
# # ax_gap.set_title("Accuracy Gap (sigmoid-based rate)")
# # ax_gap.set_xlabel("Training Size")
# # ax_gap.set_ylabel("Acc Gap")
# # ax_gap.grid(True)

# # # -------------------------
# # # 4) Subplot 2: Accuracy (A vs. B)
# # # -------------------------
# # ax_acc.plot(train_sizes, acc_A,   marker='o', color='blue',  label="sigmoid-based rate")
# # ax_acc.plot(train_sizes, acc_B,   marker='s', color='red',   label="fixed learning rate")
# # ax_acc.set_title("Accuracy")
# # ax_acc.set_xlabel("Training Size")
# # ax_acc.set_ylabel("Accuracy")
# # ax_acc.legend(loc="best")
# # ax_acc.grid(True)

# # # -------------------------
# # # 5) Subplot 3: TPR @ 0.1% FPR (A vs. B)
# # # -------------------------
# # ax_tpr01.plot(train_sizes, tpr01_A, marker='o', color='blue',  label="sigmoid-based rate")
# # ax_tpr01.plot(train_sizes, tpr01_B, marker='s', color='red',   label="fixed learning rate")
# # ax_tpr01.set_title("TPR @ 0.1% FPR")
# # ax_tpr01.set_xlabel("Training Size")
# # ax_tpr01.set_ylabel("TPR")
# # ax_tpr01.legend(loc="best")
# # ax_tpr01.grid(True)

# # # -------------------------
# # # 6) Subplot 4: TPR @ 1% FPR (A vs. B)
# # # -------------------------
# # ax_tpr1.plot(train_sizes, tpr1_A, marker='o', color='blue', label="sigmoid-based rate")
# # ax_tpr1.plot(train_sizes, tpr1_B, marker='s', color='red',  label="fixed learning rate")
# # ax_tpr1.set_title("TPR @ 1% FPR")
# # ax_tpr1.set_xlabel("Training Size")
# # ax_tpr1.set_ylabel("TPR")
# # ax_tpr1.legend(loc="best")
# # ax_tpr1.grid(True)

# # plt.tight_layout()
# # plt.show()

# # import numpy as np
# # import matplotlib.pyplot as plt

# # # -------------------------
# # # 1) Hardcoded Data
# # # -------------------------
# # train_sizes = np.array([100, 200, 300, 400, 500, 600, 1000, 2000, 5000, 10000])

# # # Approach A (sigmoid-based scheduling)
# # acc_A    = np.array([0.978627451, 0.939215686, 0.928630037, 0.936666667, 0.9296, 0.950787402,
# #                      0.941964286, 0.929739317, 0.918939616, 0.899962447])
# # tpr01_A  = np.array([0.029411765, 0.1, 0.169230769, 0.512195122, 0.142857143, 0.244186047,
# #                      0.136160714, 0.099104794, 0.09772423,  0.099075961])
# # tpr1_A   = np.array([0.029411765, 0.1, 0.184615385, 0.597560976, 0.339285714, 0.329457364,
# #                      0.3175, 0.201, 0.29, 0.363663986])

# # # Approach B (fixed learning rate)
# # acc_B    = np.array([0.968627451, 0.909803922, 0.926630037, 0.928417761, 0.925642857, 0.956723433,
# #                      0.9375, 0.926264457, 0.906434033, 0.898635223])
# # tpr01_B  = np.array([0.029411765, 0.083333333, 0.169230769, 0.12195122,  0.049107143, 0.313953488,
# #                      0.145089286, 0.098104794, 0.042391789, 0.09562109])
# # tpr1_B   = np.array([0.029411765, 0.083333333, 0.246153846, 0.408536585, 0.102678571, 0.46124031,
# #                      0.276785714, 0.191750279, 0.242748773, 0.353663986])

# # # -------------------------
# # # 2) Compute Differences (A - B)
# # # -------------------------
# # acc_diff   = acc_A - acc_B
# # tpr01_diff = tpr01_A - tpr01_B
# # tpr1_diff  = tpr1_A - tpr1_B

# # # -------------------------
# # # 3) Global Plot Settings
# # # -------------------------
# # size = 20
# # params = {
# #     'axes.labelsize': size,
# #     'font.size': size,
# #     'legend.fontsize': size,
# #     'xtick.labelsize': size,
# #     'ytick.labelsize': size,
# #     'figure.figsize': [10, 12],
# #     "font.family": "arial",
# # }
# # plt.rcParams.update(params)

# # # -------------------------
# # # 4) Plot the Differences in Subplots
# # # -------------------------
# # fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

# # # Subplot 1: Accuracy Difference
# # axes[0].plot(train_sizes, acc_diff, marker='o', color='blue', linewidth=2, label="Accuracy Diff (A-B)")
# # axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
# # axes[0].set_ylabel("Accuracy Diff")
# # axes[0].set_title("Difference between Sigmoid-based and Fixed LR")
# # axes[0].legend(loc="best")
# # axes[0].grid(True)

# # # Subplot 2: TPR @ 0.1% FPR Difference
# # axes[1].plot(train_sizes, tpr01_diff, marker='o', color='green', linewidth=2, label="TPR@0.1% Diff (A-B)")
# # axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
# # axes[1].set_ylabel("TPR@0.1% Diff")
# # axes[1].legend(loc="best")
# # axes[1].grid(True)

# # # Subplot 3: TPR @ 1% FPR Difference
# # axes[2].plot(train_sizes, tpr1_diff, marker='o', color='red', linewidth=2, label="TPR@1% Diff (A-B)")
# # axes[2].axhline(0, color='gray', linestyle='--', linewidth=1)
# # axes[2].set_ylabel("TPR@1% Diff")
# # axes[2].set_xlabel("Training Size")
# # axes[2].legend(loc="best")
# # axes[2].grid(True)

# # plt.tight_layout()
# # plt.show()

# # import matplotlib.pyplot as plt

# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Sample data (replace with your actual values)
# # train_sizes = np.array([100, 200, 300, 400, 500, 600, 1000, 2000, 5000, 10000])
# # # Accuracy for Approach A and B
# # acc_A = np.array([0.9786, 0.9392, 0.9286, 0.9367, 0.9296, 0.9508, 0.9420, 0.9297, 0.9189, 0.9000])
# # acc_B = np.array([0.9686, 0.9098, 0.9266, 0.9284, 0.9256, 0.9567, 0.9375, 0.9263, 0.9064, 0.8986])



# # import numpy as np
# # import matplotlib.pyplot as plt

# # def smooth_curve(data, window=3):
# #     """
# #     Smooths a 1D array using a simple moving average with the specified window.
# #     Uses 'same' mode to return an array of the same length.
# #     """
# #     kernel = np.ones(window) / window
# #     return np.convolve(data, kernel, mode='same')

# # # -------------------------
# # # 1) Hardcoded Data (example values)
# # # -------------------------
# # # -------------------------
# # # 2) Smooth the Data
# # # -------------------------
# # window = 3 # Adjust as needed

# # acc_A     = smooth_curve(acc_A, window)


# # acc_B   = smooth_curve(acc_B, window)


# # # Global plot settings
# # size = 20
# # params = {
# #    'axes.labelsize': size,
# #    'font.size': size,
# #    'legend.fontsize': size,
# #    'xtick.labelsize': size-5,
# #    'ytick.labelsize': size-5,
# #    'figure.figsize': [10, 8],
# #    "font.family": "arial",
# # }
# # plt.rcParams.update(params)





# # # Compute differences
# # acc_diff   = acc_A - acc_B
# # tpr01_diff = np.array([0.02941, 0.1, 0.16923, 0.51220, 0.14286, 0.24419, 0.13616, 0.09910, 0.09772, 0.09908]) - \
# #              np.array([0.02941, 0.08333, 0.16923, 0.12195, 0.04911, 0.31395, 0.14509, 0.09810, 0.04239, 0.09562])
# # tpr1_diff  = np.array([0.02941, 0.1, 0.18462, 0.59756, 0.33929, 0.32946, 0.31750, 0.201, 0.29, 0.36366]) - \
# #              np.array([0.02941, 0.08333, 0.24615, 0.40854, 0.10268, 0.46124, 0.27679, 0.19175, 0.24275, 0.35366])



# # plt.figure(figsize=(10, 8))
# # plt.plot(train_sizes, acc_diff, marker='o', linestyle='-', color='blue', label="Accuracy Diff")
# # plt.plot(train_sizes, tpr01_diff, marker='s', linestyle='-', color='green', label="TPR@0.1% Diff")
# # plt.plot(train_sizes, tpr1_diff, marker='^', linestyle='-', color='red', label="TPR@1% Diff")
# # plt.axhline(0, color='gray', linestyle='--', linewidth=1)
# # plt.xlabel("Training Size")
# # plt.ylabel("Difference (A - B)")
# # plt.title("Difference between Sigmoid-based and Fixed LR Strategies")
# # plt.legend(loc="best")
# # # plt.grid(True)
# # plt.tight_layout()
# # plt.show()


# # # plotting peromanc eidffiernce for cnn, cifar10 varying sizes, to check the varuous learning rate strategies (final)
# import numpy as np
# import matplotlib.pyplot as plt

# def smooth_curve(data, window=3):
#     """
#     Smooths a 1D array using a simple moving average with the specified window.
#     Uses 'same' mode to return an array of the same length.
#     """
#     kernel = np.ones(window) / window
#     return np.convolve(data, kernel, mode='same')

# # -------------------------
# # 1) Hardcoded Data (example values)
# # -------------------------
# train_sizes = np.array([100, 200, 300, 400, 500, 600, 1000, 2000, 5000, 10000])

# # Approach A (sigmoid-based scheduling)
# acc_A = np.array([0.978627451, 0.939215686, 0.928630037, 0.936666667, 0.9296, 0.950787402,
#                   0.941964286, 0.929739317, 0.918939616, 0.899962447])
# tpr01_A = np.array([0.029411765, 0.1,         0.169230769, 0.512195122, 0.142857143, 0.244186047,
#                     0.136160714, 0.099104794, 0.09772423,  0.099075961])
# tpr1_A = np.array([0.029411765, 0.1,         0.184615385, 0.597560976, 0.339285714, 0.329457364,
#                    0.3175,      0.201,       0.29,        0.363663986])

# # Approach B (fixed learning rate)
# acc_B = np.array([0.968627451, 0.909803922, 0.926630037, 0.928417761, 0.925642857, 0.956723433,
#                   0.9375,      0.926264457, 0.906434033, 0.898635223])
# tpr01_B = np.array([0.029411765, 0.083333333, 0.169230769, 0.12195122,  0.049107143, 0.313953488,
#                     0.145089286, 0.098104794, 0.042391789, 0.09562109])
# tpr1_B = np.array([0.029411765, 0.083333333, 0.246153846, 0.408536585, 0.102678571, 0.46124031,
#                    0.276785714, 0.191750279, 0.242748773, 0.353663986])

# # -------------------------
# # 2) Smooth the Data
# # -------------------------
# window = 3  # Adjust window size if needed
# acc_A_smooth    = smooth_curve(acc_A, window)
# tpr01_A_smooth  = smooth_curve(tpr01_A, window)
# tpr1_A_smooth   = smooth_curve(tpr1_A, window)

# acc_B_smooth    = smooth_curve(acc_B, window)
# tpr01_B_smooth  = smooth_curve(tpr01_B, window)
# tpr1_B_smooth   = smooth_curve(tpr1_B, window)

# # -------------------------
# # 3) Global Plot Settings
# # -------------------------
# size = 30
# params = {
#    'axes.labelsize': size,
#    'font.size': size,
#    'legend.fontsize': size,
#    'xtick.labelsize': size,
#    'ytick.labelsize': size,
#    'figure.figsize': [10, 8],
#    "font.family": "arial",
# }
# plt.rcParams.update(params)

# # Compute differences for each metric (Approach A - Approach B)
# acc_diff   = acc_A_smooth - acc_B_smooth
# tpr01_diff = tpr01_A_smooth - tpr01_B_smooth
# tpr1_diff  = tpr1_A_smooth - tpr1_B_smooth

# # -------------------------
# # 4) Create Plot with Background Shading
# # -------------------------
# plt.figure(figsize=(10, 8))

# plt.plot(train_sizes, acc_diff, marker='o', linestyle='-', color='blue', label="Accuracy Diff")
# plt.plot(train_sizes, tpr01_diff, marker='s', linestyle='-', color='green', label="TPR@0.1% Diff")
# plt.plot(train_sizes, tpr1_diff, marker='^', linestyle='-', color='red', label="TPR@1% Diff")
# plt.axhline(0, color='gray', linestyle='--', linewidth=1)

# plt.xlabel("Training Size")
# plt.ylabel("Difference (A - B)")
# # plt.title("Difference between Sigmoid-based and Fixed LR Strategies")

# # After plotting, get the current y-axis limits
# ymin, ymax = plt.ylim()
# # Shade region where difference is positive (A > B) in blue, negative (A < B) in red.
# plt.axhspan(0, ymax, facecolor='blue', alpha=0.1)
# plt.axhspan(ymin, 0, facecolor='red', alpha=0.1)

# plt.legend(loc="best")
# plt.tight_layout()
# # plt.show()


# plt.savefig("lr_strategy_difference.pdf", dpi=300, bbox_inches="tight")
# plt.show()

# # #  the following code is used genere sigmpid plot used in the paper for refeerences

# import numpy as np
# import matplotlib.pyplot as plt

# def get_ent_lr(acc_gap, lower=0.0001, upper=0.1, k=10, mid=0.75):
#     """
#     Returns a learning rate (for the entropy threshold) that varies sigmoidally with the accuracy gap.
#     The output is scaled to lie between `lower` and `upper`.
#     """
#     return lower + (upper - lower) * (1 - 1/(1 + np.exp(-k * (acc_gap - mid))))

# def get_cs_lr(acc_gap, lower=0.001, upper=0.01, k=10, mid=0.5):
#     """
#     Returns a learning rate (for the cosine similarity threshold) that varies sigmoidally with the accuracy gap.
#     The output is scaled to lie between `lower` and `upper`.
#     """
#     return lower + (upper - lower) * (1 - 1/(1 + np.exp(-k * (acc_gap - mid))))

# # Generate a range of accuracy gap values between 0 and 1
# acc_gap_values = np.linspace(0, 1, 100)
# ent_lr_values = get_ent_lr(acc_gap_values)
# cs_lr_values  = get_cs_lr(acc_gap_values)

# # Global plot settings
# size = 30
# params = {
#    'axes.labelsize': size,
#    'font.size': size,
#    'legend.fontsize': size,
#    'xtick.labelsize': size,
#    'ytick.labelsize': size,
#    'figure.figsize': [10, 8],
#    "font.family": "arial",
# }
# plt.rcParams.update(params)

# # Create the plot
# # plt.figure(figsize=(10, 6))
# plt.plot(acc_gap_values, ent_lr_values, label="Entropy LR", color="blue", linewidth=2)
# # plt.plot(acc_gap_values, cs_lr_values,  label="Cosine-Similarity LR", color="red", linewidth=2)
# plt.xlabel("Accuracy Gap (%)")
# plt.ylabel("Learning Rate")
# # plt.title("Learning Rate vs. Accuracy Gap")

# # Highlight the mid point (acc_gap = 0.5)
# mid = 0.5
# mid_value = get_ent_lr(mid)  # Both functions are identical in this example
# plt.axvline(mid, color="gray", linestyle="--", linewidth=1)
# plt.annotate(f"Mid = {mid}\nLR = {mid_value:.3f}", 
#              xy=(mid, mid_value),
#              xytext=(mid + 0.05, mid_value + 0.001),
#              arrowprops=dict(arrowstyle="->", color="blue"),
#              color="blue", fontsize=20)

# # plt.legend(loc="best")
# plt.tight_layout()
# plt.savefig("learning_rate_vs_acc_gap.pdf", dpi=300, bbox_inches="tight")
# plt.show()


# # # #  The following is used to plot roc for magin_vgg16_cl_pert_cifar10_5k (in the paper)
# import pandas as pd
# import matplotlib.pyplot as plt


# size = 30
# params = {
# 'axes.labelsize': size,
# 'font.size': size,
# 'legend.fontsize': size,
# 'xtick.labelsize': size,
# 'ytick.labelsize': size,
# #    'text.usetex': False,
# 'figure.figsize': [10, 9],
# "font.family": "arial",

# }
# plt.rcParams.update(params)

# csv_path = "magin_vgg16_cl_pert_cifar10_5k.csv"

# df = pd.read_csv(csv_path)

# # Extract FPR/TPR columns for each margin
# fpr_0_0 = df["FPR_0.0"].values
# tpr_0_0 = df["TPR_0.0"].values

# fpr_0_2 = df["FPR_0.2"].values
# tpr_0_2 = df["TPR_0.2"].values

# fpr_0_5 = df["FPR_0.5"].values
# tpr_0_5 = df["TPR_0.5"].values

# fpr_0_8 = df["FPR_0.8"].values
# tpr_0_8 = df["TPR_0.8"].values

# fpr_1_0 = df["FPR_1.0"].values
# tpr_1_0 = df["TPR_1.0"].values

# # Plot each margin's ROC curve
# # plt.figure(figsize=(8, 6))

# plt.plot(fpr_0_0, tpr_0_0, label=r"$m=0.1$", linewidth=3.5)
# plt.plot(fpr_0_2, tpr_0_2, label=r"$m=0.2$", linewidth=3.5)
# plt.plot(fpr_0_5, tpr_0_5, label=r"$m=0.5$", linewidth=3.5)
# plt.plot(fpr_0_8, tpr_0_8, label=r"$m=0.8$", linewidth=3.5)
# plt.plot(fpr_1_0, tpr_1_0, label=r"$m=1.0$", linewidth=3.5)

# plt.plot([0, 1], [0, 1], ls="--", color="gray")


# #  legend = plt.legend(loc="lower right")
# # frame = legend.get_frame()
# # frame.set_facecolor('0.97')
# # frame.set_edgecolor('0.91')
# # plt.subplots_adjust(left=0.60)
# plt.tight_layout()

# # plt.semilogx()
# plt.semilogy()
# plt.xlim(1e-5, 1)
# plt.ylim(1e-5, 1.01)

# plt.semilogx()
# plt.semilogy()
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# # plt.title("ROC Curves for Different Margins")
# plt.legend(loc="best")
# # plt.grid(True)
# plt.tight_layout()
# # plt.show()


# plt.savefig("magin_vgg16_cl_pert_cifar10_5k.pdf", dpi=300, bbox_inches="tight")

# plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import auc
# import numpy as np
# # Path to your CSV file
# csv_file = "cl_no_cl_per_no_per_fpr_tpr_comparision.csv"
# # Read CSV into a DataFrame
# df = pd.read_csv(csv_file)

# # Map each technique name to its (fpr_col, tpr_col) in the CSV.
# # Use LaTeX formatting for subscripts if desired.
# columns_map = {
#     r"$\mathrm{Perturb+CL}_{p}$": ("fpr_01", "tpr_01"),
#     r"$\mathrm{Perturb+nCL}_{p}$": ("fpr_02", "tpr_02"),
#     r"$\mathrm{nPerturb+nCL}_{p}$": ("fpr_03", "tpr_03"),
#     r"$\mathrm{nPerturb+CL}_{p}$": ("fpr_04", "tpr_04"),
#     r"$\mathrm{nPerturb+CL}_{np}$": ("fpr_05", "tpr_05"),
# }

# size = 30
# params = {
#     'axes.labelsize': size,
#     'font.size': size,
#     'legend.fontsize': size,
#     'xtick.labelsize': size-5,
#     'ytick.labelsize': size-5,
#     'figure.figsize': [10, 9],
#     "font.family": "arial",
# }
# plt.rcParams.update(params)

# # Loop through each technique, extract FPR/TPR, compute AUC, and plot the curve
# for technique_name, (fpr_col, tpr_col) in columns_map.items():
#     if fpr_col not in df.columns or tpr_col not in df.columns:
#         print(f"Columns {fpr_col} or {tpr_col} not found, skipping {technique_name}.")
#         continue
    
#     # Extract FPR and TPR values from the DataFrame
#     fpr_vals = df[fpr_col].values
#     tpr_vals = df[tpr_col].values
    
#     # Make sure values are numpy arrays
#     fpr_vals = np.array(fpr_vals)
#     tpr_vals = np.array(tpr_vals)

#     # Remove NaNs or Infs
#     valid = ~(np.isnan(fpr_vals) | np.isnan(tpr_vals) | np.isinf(fpr_vals) | np.isinf(tpr_vals))
#     fpr_vals = fpr_vals[valid]
#     tpr_vals = tpr_vals[valid]

#     # Remove duplicate FPR values (keep the max TPR for each FPR)
#     unique_fpr = {}
#     for f, t in zip(fpr_vals, tpr_vals):
#         if f not in unique_fpr or t > unique_fpr[f]:
#             unique_fpr[f] = t

#     fpr_sorted = np.array(sorted(unique_fpr.keys()))
#     tpr_sorted = np.array([unique_fpr[f] for f in fpr_sorted])

#     # Compute AUC
#     if len(fpr_sorted) >= 2:
#         auc_value = auc(fpr_sorted, tpr_sorted)
#     else:
#         auc_value = float("nan")  # Not enough points to compute AUC
        
#     # Compute the AUC using the trapezoidal rule
#     # auc_value = auc(fpr_vals, tpr_vals)
    
#     # Create a legend label that includes the AUC value
#     label_full = f"{technique_name}"
    
#     plt.plot(fpr_vals, tpr_vals, label=label_full, linewidth=3.5)

# # Plot the diagonal line for reference
# plt.plot([0, 1], [0, 1], ls="--", color="gray")

# # Set axis labels and title
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# # plt.title("ROC Curves for Different Techniques")
# plt.legend(loc="lower right")
# # plt.semilogx()
# # plt.semilogy()
# plt.xlim(1e-5, 1)
# plt.ylim(1e-5, 1.01)
# plt.tight_layout()

# # Save the figure as a PDF with 300 dpi
# plt.savefig("cl_no_cl_per_no_per_fpr_tpr_no_comparision_semilogy.pdf", dpi=300, bbox_inches="tight")
# plt.show()



# # the following plots distance_metrics_similarity_detection_strat

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
# Path to your CSV file
csv_file = "distance_metrics_similarity_detection_strat.csv"
# Read CSV into a DataFrame
df = pd.read_csv(csv_file)

# Map each technique name to its (fpr_col, tpr_col) in the CSV.
# Use LaTeX formatting for subscripts if desired.
columns_map = {
    "Cosine Similarity": ("fpr_01", "tpr_01"),
    "Euclidean Distance": ("fpr_02", "tpr_02"),
    "KL Divergence": ("fpr_03", "tpr_03"),
    "Pearson Correlation": ("fpr_04", "tpr_04"),
    "Mahalanobis Distance": ("fpr_05", "tpr_05"),


    
}

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
plt.rcParams.update(params)

# Loop through each technique, extract FPR/TPR, compute AUC, and plot the curve
for technique_name, (fpr_col, tpr_col) in columns_map.items():
    if fpr_col not in df.columns or tpr_col not in df.columns:
        print(f"Columns {fpr_col} or {tpr_col} not found, skipping {technique_name}.")
        continue
    
    # Extract FPR and TPR values from the DataFrame
    fpr_vals = df[fpr_col].values
    tpr_vals = df[tpr_col].values
    
    # Make sure values are numpy arrays
    fpr_vals = np.array(fpr_vals)
    tpr_vals = np.array(tpr_vals)

    # Remove NaNs or Infs
    valid = ~(np.isnan(fpr_vals) | np.isnan(tpr_vals) | np.isinf(fpr_vals) | np.isinf(tpr_vals))
    fpr_vals = fpr_vals[valid]
    tpr_vals = tpr_vals[valid]

    # Remove duplicate FPR values (keep the max TPR for each FPR)
    unique_fpr = {}
    for f, t in zip(fpr_vals, tpr_vals):
        if f not in unique_fpr or t > unique_fpr[f]:
            unique_fpr[f] = t

    fpr_sorted = np.array(sorted(unique_fpr.keys()))
    tpr_sorted = np.array([unique_fpr[f] for f in fpr_sorted])

    # Compute AUC
    if len(fpr_sorted) >= 2:
        auc_value = auc(fpr_sorted, tpr_sorted)
    else:
        auc_value = float("nan")  # Not enough points to compute AUC
        
    # Compute the AUC using the trapezoidal rule
    # auc_value = auc(fpr_vals, tpr_vals)
    
    # Create a legend label that includes the AUC value
    label_full = f"{technique_name}"
    
    plt.plot(fpr_vals, tpr_vals, label=label_full, linewidth=3.5)

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], ls="--", color="gray")

# Set axis labels and title
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.title("ROC Curves for Different Techniques")
plt.legend(loc="lower right")
plt.semilogx()
plt.semilogy()
plt.xlim(1e-5, 1)
plt.ylim(1e-5, 1.01)
plt.tight_layout()

# Save the figure as a PDF with 300 dpi
plt.savefig("distance_metrics_similarity_detection_strat.pdf", dpi=300, bbox_inches="tight")
plt.show()
