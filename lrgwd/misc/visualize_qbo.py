from scipy.io import netcdf

with netcdf.netcdf_file("../../netcdf_data/atmos_1day_d11160_plevel.nc") as cdf_data:
    ucomp = cdf.data.variables["ucomp"][:]
    ucomp_vert_col = ucomp[1400, :, 32, ]
