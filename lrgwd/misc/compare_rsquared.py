import os

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from lrgwd.utils.io import from_pickle

# MSE = [0.9400329889602671, 0.9545922128356706, 0.9534312628731445, 0.9480089100272873, 0.9096108126187419, 0.7459606027788386, 0.7841387738957375, 0.8554988685144947, 0.869406712096818, 0.8637525650883207, 0.8483577821840436, 0.7952723013090828, 0.7585418389563084, 0.6998809538012278, 0.7165199650669133, 0.646975544826842, 0.17753648588391222, 0.06766231466566698]

# MAE = [0.9215100012711294, 0.9445086662550973, 0.9442184861513464, 0.9342774000321638, 0.8837685208403457, 0.6502402627344941, 0.699168560154172, 0.7953884429711442, 0.8165758965485282, 0.8093460968347096, 0.7882198778631545, 0.7565313881422749, 0.77937499868345, 0.794270742962999, 0.866113681754332, 0.80877625509013, 0.4685345630864879, 0.5656377293628407]

# # LogCosh = [0.9500278547928318, 0.9656422187139331, 0.965462619244026, 0.9598131252514639, 0.9267468643242859, 0.7751309348798543, 0.8017119344131914, 0.8739724889338316, 0.8866521040533402, 0.8811033996586276, 0.871179124895153, 0.8430248685835904, 0.8536340082097484, 0.8475890220793019, 0.8996017532123329, 0.8579803423453516, 0.5418392501773467, 0.47662772620737137]

# # Massive (1mil)
# LogCosh = [0.9496326951693226, 0.9651840824885862, 0.9648676953619554, 0.959792952834377, 0.9264600613700418, 0.776496729252286, 0.8005097508329734, 0.8746857025874119, 0.8861190014216245, 0.8810557998561824, 0.871259631335388, 0.8449381905519774, 0.8600544492363277, 0.8582861931114023, 0.9133762611928817, 0.8742649686448731, 0.6164430086014544, 0.6841250997268723]

# Composite = [0.9284486792739809, 0.9455585211204894, 0.9459576497988462, 0.9405415013505347, 0.8835270077112379, 0.6889759185896543, 0.7277157428604346, 0.8167771341991533, 0.8300069819294771, 0.8134289792123736, 0.8088030481943791, 0.7690307245726562, 0.7908844506747738, 0.7922402056647841, 0.8638568785435826, 0.8503209815367738, 0.5838624686456577, 0.7009526556906135]

# # Year One LogCosh 5 mil
# [0.9494049169484787, 0.964824927538381, 0.9645408256669529, 0.9595914855003685, 0.9262538969785157, 0.7780945194305404, 0.8011783312287044, 0.8749407996524565, 0.8863997527020785, 0.8823888341813839, 0.8723156805835925, 0.8454697513139204, 0.8610355778445342, 0.8593298259891885, 0.913666765735813, 0.8753043210295398, 0.6194315403202093, 0.6885749842988453]

# # year_one = [0.9688072656501203, 0.9775339329269576, 0.9776340320865292, 0.9745280702244162, 0.9571579537983793, 0.8496055751875617, 0.8673528530669082, 0.9248088159878061, 0.9334661060645576, 0.9338282519522377, 0.9237514322244461, 0.9026462912688763, 0.9180432614977277, 0.9160784915545532, 0.9471160124999615, 0.9288664401445216, 0.757751897737308, 0.8426664399538599]
# # year_two = [0.9571628826637478, 0.9674122009127056, 0.9676638035560068, 0.9638843607354981, 0.9387507831192856, 0.7403240535956304, 0.7864752152729604, 0.8522711197261269, 0.8511531883516812, 0.8692136205903124, 0.8178466022519693, 0.7897694087705549, 0.8523805188442078, 0.8634854879468945, 0.8908053564656027, 0.8723555691584719, 0.6067284170868076, 0.8122143585917225]
# # year_three = [0.9550203578641319, 0.966036503106561, 0.9660990205355283, 0.961780669346648, 0.9350844631508187, 0.7687797651155256, 0.8000467714158865, 0.8594936032135084, 0.8820151629352786, 0.89698887502708, 0.8137172227359912, 0.7391549700392955, 0.8714112940862151, 0.8638144477161648, 0.8971934812596527, 0.8742916536755807, 0.6151426119679773, 0.8060236253863958]
# # year_four = [0.9593810710117473, 0.9686658808419019, 0.9680151298206313, 0.9651727189958603, 0.9439543645184347, 0.7995989622376265, 0.8186261088721937, 0.8658247041419107, 0.901181838810153, 0.899990300585051, 0.8317673080174618, 0.8409928076582981, 0.8703034585906374, 0.8509632557747917, 0.9109632714103998, 0.876607485488826, 0.6323153714663838, 0.8129744152176563]

# only_u = [0.945214800465111, 0.9416409336479143, 0.8946224311821449, 0.7217518279625608, 0.7698180829964812, 0.8018145734776743, 0.8213927363732786, 0.8358386960261909, 0.8566897749493433, 0.8960099302647486, 0.8850057248834897, 0.8897681879647245, 0.9038094956226776, 0.8755772444520927, 0.8720253562862261, 0.8746512450365697, 0.8569982328231543, 0.8874044372272766, 0.9023506720710273, 0.9020479419958618, 0.9309363394660255, 0.9419084475610667, 0.9552148360597376, 0.9567123357011231, 0.9472401782161399, 0.9193141335488934, 0.8964964057240633, 0.8789330526077767, 0.8662060149883687, 0.7867599617084736, 0.7897696126328251, 0.8047058302793157, 0.7976876808336006]
# only_v = [0.38931400072093664, 0.3756812381189184, 0.3185673856593284, 0.12497895379980081, 0.13465160888622177, 0.09588119823640175, 0.06245254705674652, 0.06516451481834347, 0.0801185231285939, 0.033019762849312836, 0.005633480130726643, 0.002280554952944063, 0.022002845875598986, 0.03570893608193528, 6.836786550193582e-05, 0.002413127793479563, 0.0005597882837402377, 0.035320465249481514, 0.1045824972588624, 0.2124632724088696, 0.31057801733176343, 0.35576566852405567, 0.37592710799119455, 0.4025743926179225, 0.3821656261598378, 0.2623632832840157, 0.1449929594792116, 0.11071010378877004, 0.12657359390346787, 0.18425315461332956, 0.22784760115139255, 0.1499635543408435, 0.0927469410031183]
# only_uv = [0.9528944837564081, 0.9494433165449714, 0.9157667108602194, 0.7708073259855285, 0.8186025902466196, 0.8578974015183873, 0.8682627860446848, 0.8728080973540633, 0.8751045399222905, 0.9127483949349562, 0.8973317475373042, 0.9012617355314623, 0.9207973860763959, 0.914279648206125, 0.9027244368112818, 0.9069831446899546, 0.8863200231216591, 0.9015877928379488, 0.9161855062982394, 0.9156510502874455, 0.9214496654206138, 0.9403237677600094, 0.9500173781397522, 0.957540416426511, 0.946868102761248, 0.9199220456669501, 0.9003283261717651, 0.8778857788777229, 0.859733040812875, 0.805859858419275, 0.7855708208063507, 0.8269778365488054, 0.8170098954079344]
# full_features = [0.968096101723858, 0.9629162862179804, 0.9359338780377544, 0.8296802457805219, 0.8581975714835819, 0.879644327742136, 0.8932815154877679, 0.8980783569256147, 0.9050982043456887, 0.9241682647523103, 0.909417393853762, 0.918121830220236, 0.9278009213447242, 0.9176942169325333, 0.9177304304611712, 0.9212065681149525, 0.9178218535420961, 0.9272174952874452, 0.9284154879730179, 0.9372586118543167, 0.9467723048248042, 0.9576875694261682, 0.9665098057050374, 0.9714431213418451, 0.9648544738553332, 0.9490431149896886, 0.9325603733909039, 0.9192400501212071, 0.9154073238331839, 0.9116306364854845, 0.9390582020373587, 0.9355448339951927, 0.9307123878997413]

# u_mae = [1.93101423e-05, 2.05977180e-05, 2.22979394e-05, 1.06410513e-05,
#        7.69928544e-06, 5.76063757e-06, 4.43545588e-06, 3.44389435e-06,
#        2.65287399e-06, 1.98886111e-06, 1.57436105e-06, 1.17054123e-06,
#        8.95868808e-07, 7.54191090e-07, 5.82460814e-07, 4.46433509e-07,
#        3.51755124e-07, 2.71542114e-07, 2.22017398e-07, 1.85507415e-07,
#        1.54272460e-07, 1.44008061e-07, 1.34655203e-07, 1.32356891e-07,
#        1.27337129e-07, 1.20841597e-07, 1.06537498e-07, 9.37019323e-08,
#        8.39020782e-08, 7.82651290e-08, 5.27364756e-08, 3.37680683e-08,
#        2.26296037e-08]
# v_mae = [7.70621694e-05, 7.71998373e-05, 6.13742600e-05, 1.93468023e-05,
#        1.59080034e-05, 1.45027961e-05, 1.22781663e-05, 9.21615031e-06,
#        7.33003185e-06, 6.56018108e-06, 5.49015739e-06, 4.19147031e-06,
#        3.28033806e-06, 2.57710150e-06, 2.11043893e-06, 1.65246663e-06,
#        1.23705190e-06, 9.59276093e-07, 7.69286636e-07, 6.20313064e-07,
#        5.62792947e-07, 5.55155243e-07, 5.50728605e-07, 5.33740194e-07,
#        4.82484964e-07, 4.17978693e-07, 3.46837268e-07, 2.84766227e-07,
#        2.41094261e-07, 1.83927688e-07, 1.18704527e-07, 8.58356443e-08,
#        5.78002370e-08]
# uv_mae = [1.73586835e-05, 1.85567996e-05, 1.92857085e-05, 9.21787066e-06,
#        6.49565547e-06, 4.67242308e-06, 3.63203420e-06, 2.91898379e-06,
#        2.39438586e-06, 1.76562441e-06, 1.42424400e-06, 1.07018187e-06,
#        7.96788222e-07, 6.38879342e-07, 5.04373626e-07, 3.89359729e-07,
#        3.14252308e-07, 2.52876409e-07, 2.06339153e-07, 1.72530906e-07,
#        1.54624376e-07, 1.40357736e-07, 1.35737084e-07, 1.29776750e-07,
#        1.25651710e-07, 1.18596252e-07, 1.03798493e-07, 9.23033747e-08,
#        8.39143775e-08, 7.49348396e-08, 5.36787721e-08, 3.22120190e-08,
#        2.15681567e-08]
# full_mae = [1.42361272e-05, 1.57172346e-05, 1.67958728e-05, 7.83749701e-06,
#        5.73197481e-06, 4.25908139e-06, 3.24825074e-06, 2.61104987e-06,
#        2.08400025e-06, 1.63546488e-06, 1.33883417e-06, 9.74494858e-07,
#        7.46260225e-07, 6.17859857e-07, 4.69863760e-07, 3.60532092e-07,
#        2.79531715e-07, 2.25063307e-07, 1.89142900e-07, 1.52930969e-07,
#        1.32531353e-07, 1.22286696e-07, 1.16107768e-07, 1.09889813e-07,
#        1.05362300e-07, 9.75513705e-08, 8.63465533e-08, 7.66584168e-08,
#        6.78899602e-08, 5.53388866e-08, 3.06750773e-08, 2.11315224e-08,
#        1.40577267e-08]
# rand_r_squared =


# fig = plt.figure(figsize=(8,6))
# labels = [1.80e-01, 5.60e-01, 7.20e-01, 9.40e-01, 1.21e+00, 1.57e+00, 2.02e+00, 2.60e+00,
#           3.32e+00, 4.25e+00, 5.40e+00, 6.85e+00, 8.68e+00, 1.09e+01, 1.38e+01, 1.73e+01,
#           2.16e+01, 2.68e+01, 3.32e+01, 4.11e+01, 5.07e+01, 6.22e+01, 7.60e+01, 9.24e+01,
#           1.12e+02, 1.35e+02, 1.62e+02, 1.94e+02, 2.31e+02, 2.73e+02, 3.21e+02, 3.75e+02,
#           4.36e+02]
# plevels = list(range(len(labels)))
# # labels = [.1, .2, .3, .5, .7, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0, 200.0, 300.0]
# plt.plot(full_features, plevels, label="[u, v, omega, hght, temp, lat, lon, slp]")
# plt.plot(only_u, plevels, label="[u]")
# plt.plot(only_uv, plevels, label="[u, v]")
# plt.plot(only_v, plevels, label="[v]")
# plt.plot(rand, plevels, label="rand")
# # plt.plot(full_mae, plevels, label="[u, v, omega, hght, temp, lat, lon, slp]")
# # plt.plot(u_mae, plevels, label="[u]")
# # plt.plot(uv_mae, plevels, label="[u, v]")
# # plt.plot(v_mae, plevels, label="[v]")
# # plt.plot(plevels, MSE, label="MSE R^2")
# # plt.plot(plevels, MAE, label="MAE R^2")
# # plt.plot(plevels, LogCosh, label="LogCosh R^2")
# # plt.plot(plevels, Composite, label="Composite R^2")
# plt.ylabel("Pressure [hPa]", size=14)
# plt.xlabel("$R^2$", size=14)
# # plt.xscale("log")
# plt.xticks(np.arange(0,1.0,.05))
# plt.yticks(np.arange(1,33.0,1.0), labels=labels)
# plt.xlim(0.0, 1.0)
# plt.title("$R^2$ by height per feature variable")

# plt.legend()
# plt.show()


def generate_rsquared_plots(u_experiments, v_experiments, plevels):
       fig, ((ax_rsq_u), (ax_rsq_v)) = plt.subplots(2, 1)
       fig.suptitle("Performance Sensitivity to Feature Selection", fontsize=18)

       linewidth=2
       labelsize=14
       titlesize=16
       marker='D'
       markeredgecolor='black'

       plevels_idx = np.arange(0, len(plevels), 1.0)
       plevels_idx_rev = np.flip(plevels_idx)
       for label, feat_exp in u_experiments.items():
              r_squared = feat_exp["r_squared"]
              print(f"{label}: ", np.average(r_squared))
              ax_rsq_u.plot(feat_exp["r_squared"], plevels_idx_rev, label=label, linewidth=linewidth)

       for label, feat_exp in v_experiments.items():
              r_squared = feat_exp["r_squared"]
              print(f"{label}: ", np.average(r_squared))
              ax_rsq_v.plot(feat_exp["r_squared"], plevels_idx_rev, label=label, linewidth=linewidth)

       # Set Titles
       ax_rsq_u.set_title("$R^2$ gwfu", fontsize=titlesize)
       ax_rsq_v.set_title("$R^2$ gwfv", fontsize=titlesize)

       y_ticks = np.arange(0, len(plevels), 2)
       y_ticklabels = np.flip(plevels[::2])

       y_ticks = [3,7,11,15,19,23,27,31]
       y_ticklabels = list([500.0, 100.0, 50.0, 10.0, 5.0, 1.0, .5, .1])

       y_ticks = [7,15,23,31]
       y_ticklabels = list([100.0, 10.0, 1.0, .1])

       # Set Y and X Labels
       ax_rsq_u.set_ylabel("Pressure [hPa]", fontsize=labelsize)
       ax_rsq_u.set_yticks(y_ticks)
       ax_rsq_u.set_yticklabels(y_ticklabels)


       ax_rsq_v.set_ylabel("Pressure [hPa]", fontsize=labelsize)
       ax_rsq_v.set_yticks(y_ticks)
       ax_rsq_v.set_yticklabels(y_ticklabels)


       ax_rsq_v.set_xticks(np.arange(0, 1.0, .1))
       ax_rsq_v.set_xlabel("$R^2$", fontsize=titlesize)
       ax_rsq_u.set_xticks(np.arange(0, 1.0, .1))
       ax_rsq_u.set_xlabel("$R^2$", fontsize=titlesize)

       ax_rsq_v.tick_params(axis='both', labelsize=labelsize)
       ax_rsq_u.tick_params(axis='both', labelsize=labelsize)

       # for ax in fig.get_axes():
       #        ax.label_outer()

       # uhandles, ulabels = fig.get_axes()[0].get_legend_handles_labels()
       # vhandles, vlabels = fig.get_axes()[1].get_legend_handles_labels()
       # uhandles.extend(vhandles[6:])
       # ulabels.extend(vlabels[6:])
       ax_rsq_u.legend(loc='center', prop={'size': labelsize})
       ax_rsq_v.legend(loc='center', prop={'size': labelsize})
       # fig.legend(uhandles, ulabels, loc='lower center')

       plt.show()

def generate_plots(u_experiments, v_experiments, plevels):
       fig, ((ax_rsq_u, ax_mae_u, ax_rme_u), (ax_rsq_v, ax_mae_v, ax_rme_v)) = plt.subplots(2, 3)
       fig.suptitle("Performance Sensitivity to Feature Selection")

       linewidth=4
       marker='D'
       markeredgecolor='black'

       plevels_idx = np.arange(0, len(plevels), 1.0)
       plevels_idx_rev = np.flip(plevels_idx)
       for label, feat_exp in u_experiments.items():
              r_squared = feat_exp["r_squared"]
              print(f"{label}: ", np.average(r_squared))
              ax_rsq_u.plot(feat_exp["r_squared"], plevels_idx_rev, label=label, linewidth=linewidth)
              ax_mae_u.plot(feat_exp["maes"], plevels_idx_rev, label=label, linewidth=linewidth)
              ax_rme_u.plot(feat_exp["rmse"], plevels_idx_rev, label=label, linewidth=linewidth)

       for label, feat_exp in v_experiments.items():
              r_squared = feat_exp["r_squared"]
              print(f"{label}: ", np.average(r_squared))
              ax_rsq_v.plot(feat_exp["r_squared"], plevels_idx_rev, label=label, linewidth=linewidth)
              ax_mae_v.plot(feat_exp["maes"], plevels_idx_rev, label=label, linewidth=linewidth)
              ax_rme_v.plot(feat_exp["rmse"], plevels_idx_rev, label=label, linewidth=linewidth)

       # Set Titles
       ax_rsq_u.set_title("$R^2$ gwfu")
       ax_mae_u.set_title("MAE gwfu")
       ax_rme_u.set_title("RMSE gwfu")

       ax_rsq_v.set_title("$R^2$ gwfv")
       ax_mae_v.set_title("MAE gwfv")
       ax_rme_v.set_title("RMSE gwfv")

       y_ticks = np.arange(0, len(plevels), 2)
       y_ticklabels = np.flip(plevels[::2])

       # Set Y and X Labels
       ax_rsq_u.set_ylabel("Pressure [hPa]")
       ax_rsq_u.set_yticks(y_ticks)
       ax_rsq_u.set_yticklabels(y_ticklabels)

       ax_mae_u.set_ylabel("Pressure [hPa]")
       ax_mae_u.set_yticks(y_ticks)
       ax_mae_u.set_yticklabels(y_ticklabels)

       ax_rme_u.set_ylabel("Pressure [hPa]")
       ax_rme_u.set_yticks(y_ticks)
       ax_rme_u.set_yticklabels(y_ticklabels)

       ax_rsq_v.set_ylabel("Pressure [hPa]")
       ax_rsq_v.set_yticks(y_ticks)
       ax_rsq_v.set_yticklabels(y_ticklabels)

       ax_mae_v.set_ylabel("Pressure [hPa]")
       ax_mae_v.set_yticks(y_ticks)
       ax_mae_v.set_yticklabels(y_ticklabels)

       ax_rme_v.set_ylabel("Pressure [hPa]")
       ax_rme_v.set_yticks(y_ticks)
       ax_rme_v.set_yticklabels(y_ticklabels)

       ax_mae_v.set_xscale("log")
       ax_rme_v.set_xscale("log")
       ax_mae_u.set_xscale("log")
       ax_rme_u.set_xscale("log")

       ax_rsq_v.set_xticks(np.arange(0, 1.0, .1))
       ax_rsq_v.set_xlabel("$R^2$")
       ax_rsq_u.set_xticks(np.arange(0, 1.0, .1))
       ax_rsq_u.set_xlabel("$R^2$")
       ax_mae_v.set_xlabel("Mean absolute error")
       ax_rme_v.set_xlabel("Root mean squared error")

       # for ax in fig.get_axes():
       #        ax.label_outer()

       uhandles, ulabels = fig.get_axes()[0].get_legend_handles_labels()
       vhandles, vlabels = fig.get_axes()[3].get_legend_handles_labels()
       uhandles.extend(vhandles[6:])
       ulabels.extend(vlabels[6:])
       fig.legend(uhandles, ulabels)

       plt.show()

def generate_box_whiskers(u_experiments, v_experiments):
       # Create a figure instance
       fig, (ax_u, ax_v) = plt.subplots(2)

       u_r_squared, v_r_squared = [], []
       vlabels, ulabels = [], []
       for label, feat_exp in u_experiments.items():
              u_r_squared.append(feat_exp["r_squared"])
              ulabels.append(label)

       for label, feat_exp in v_experiments.items():
              v_r_squared.append(feat_exp["r_squared"])
              vlabels.append(label)

       # Create the boxplot
       bpu = ax_u.boxplot(u_r_squared, patch_artist=True)
       bpv = ax_v.boxplot(v_r_squared, patch_artist=True)

       ax_u.set_yticks(np.arange(0, 1.0, .05))
       ax_v.set_yticks(np.arange(0, 1.0, .05))

       ax_u.set_title("$R^2$ gwfu")
       ax_v.set_title("$R^2$ gwfv")

       ax_u.set_ylabel("$R^2$")
       ax_v.set_ylabel("$R^2$")

       ax_u.set_xticklabels(ulabels)
       ax_v.set_xticklabels(vlabels)

       # adding horizontal grid lines
       ax_u.yaxis.grid(True)
       ax_v.yaxis.grid(True)

       colors = ['tab:blue', "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan", "tab:olive", "tab:pink", "tab:brown"]
       for box, color in zip(bpu['boxes'], colors):
              box.set_facecolor(color)

       for box, color in zip(bpv['boxes'], colors):
              box.set_facecolor(color)

       plt.show()


def aggregate_experiment_results(metrics_path, experiments):
       u_experiments, v_experiments = defaultdict(lambda: {}), defaultdict(lambda: {})
       for experiment, label in experiments.items():
              if experiment not in ["vtemp", "vhght", "vlatlon"]:
                     metrics = from_pickle(
                            os.path.join(metrics_path, "gwfu", experiment, "metrics.pkl")
                     )
                     u_experiments[label]["maes"] = metrics["maes"]
                     u_experiments[label]["rmse"] = metrics["rmse"]
                     u_experiments[label]["r_squared"] = metrics["r_squared"]

              if experiment not in ["utemp", "uhght", "ulatlon"]:
                     metrics = from_pickle(
                            os.path.join(metrics_path, "gwfv", experiment, "metrics.pkl")
                     )
                     v_experiments[label]["maes"] = metrics["maes"]
                     v_experiments[label]["rmse"] = metrics["rmse"]
                     v_experiments[label]["r_squared"] = metrics["r_squared"]

       return u_experiments, v_experiments


plevels = [1.80e-01, 5.60e-01, 7.20e-01, 9.40e-01, 1.21e+00, 1.57e+00, 2.02e+00, 2.60e+00,
          3.32e+00, 4.25e+00, 5.40e+00, 6.85e+00, 8.68e+00, 1.09e+01, 1.38e+01, 1.73e+01,
          2.16e+01, 2.68e+01, 3.32e+01, 4.11e+01, 5.07e+01, 6.22e+01, 7.60e+01, 9.24e+01,
          1.12e+02, 1.35e+02, 1.62e+02, 1.94e+02, 2.31e+02, 2.73e+02, 3.21e+02, 3.75e+02,
          4.36e+02]

omega = "\u03C9"
lat = "\u03BB"
lon = "\u03C6"

experiments = {
       "full_features": f"[u,v,{omega},H,T,{lat},{lon},ps]",
       "only_uv": "[u,v]",
       "minus_uv": f"[{omega},H,T,{lat},{lon},ps]",
       "only_u": "[u]",
       "only_v":"[v]",
       # "random_var": "random",
       "utemp": "[u, T]",
       "uhght": "[u, H]",
       "ulatlon": f"[u, {lat},{lon}]",
       "vtemp": "[v, T]",
       "vhght": "[v, H]",
       "vlatlon": f"[v, {lat},{lon}]",
}

u_experiments, v_experiments = aggregate_experiment_results("evaluate/", experiments)
generate_rsquared_plots(u_experiments, v_experiments, plevels)
#generate_box_whiskers(u_experiments, v_experiments)
