from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt

with netcdf.netcdf_file("../../../netcdf_data/atmos_1day_d11160_plevel.nc") as year_one_qbo, netcdf.netcdf_file(
    "../../../netcdf_data/atmos_1day_d11520_plevel.nc") as year_two_qbo, netcdf.netcdf_file(
    "../../../netcdf_data/atmos_1day_d12240_plevel.nc") as year_three_qbo:

    ucomp_data = [year_one_qbo.variables["gwfu_cgwd"][:], year_two_qbo.variables["gwfu_cgwd"][:], year_three_qbo.variables["gwfu_cgwd"][:]]
    months = [60*i for i in range(24*len(ucomp_data)+1)]
    xticks= list(range(12*len(ucomp_data)))
    plevels = year_one_qbo.variables["level"][:]
    ucomp_data = np.concatenate(ucomp_data, axis=0)

    print(ucomp_data.shape)
    ucomp_monthly_avgs = []
    for i in range(len(months)-1):
        ucomp_monthly_avgs.append(np.average(ucomp_data[months[i]:months[i+1]-1, :, 32, 64], axis=0))

    ucomp_monthly_avgs = np.array(ucomp_monthly_avgs)
    print(ucomp_monthly_avgs.shape)

    plt.imshow(ucomp_monthly_avgs.T, cmap="BrBG")
    plt.xlabel("Month")
    plt.ylabel("Pressure (hPa)")
    plt.yticks(ticks=list(range(len(plevels))), labels=plevels)
    plt.xlim(left=1, right=36)
    plt.xticks(ticks=xticks)
    cbar = plt.colorbar()
    cbar.set_label("ucomp (m/s)")
    plt.title("True QBO")
    plt.show()
    # ucomp_year_one_avg = []
    # ucomp_year_two_avg = []
    # for i in range(len(months)-1):
    #     ucomp_year_one_avg.append(np.squeeze(np.average(ucomp_year_one[months[i]:months[i+1]-1, :, 32, 64], axis=0)))
    #     ucomp_year_two_avg.append(np.squeeze(np.average(ucomp_year_two[months[i]:months[i+1]-1, :, 32, 64], axis=0)))





    # fig = plt.figure(figsize=(8,6))
    # # plevels = list(range(len(ucomp_pos)))
    # plt.plot(ucomp_pos, plevels, label='Year 1')
    # plt.plot(ucomp_neg, plevels, label='Year 2')
    # plt.xlabel("Ucomp (m/s)", size=14)
    # plt.ylabel("PLevels", size=14)
    # plt.title(f"Ucomp vs Plevels")
    # plt.yticks(plevels)
    # # plt.yticks(np.arange(0,1.0,.1))

    # # Make figures full screen
    # # fig.set_size_inches(32, 18)
    # plt.show()
