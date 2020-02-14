from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from netCDF4 import Dataset
import matplotlib, time
import matplotlib.pyplot as plt
import numpy as np 


data = Dataset("../atmos_1day_d11160_plevel.nc", "r", format="NETCDF4")
"""
Data includes (Name, Shape): 
Longitude (128,)
Longitude Edges (129,)
Latitude (64, )
Latitude Edges (65, )
Pressure (level) (22, )
Time (1440,)
Height (1440, 22, 64, 128)
Level Pressure (slp) (1440, 64, 128)
Zonal Wind Component ucomp (1440, 22, 64, 128)
Meridional Wind Component vcomp (1440,22,64,128)
Dp/dt omega vertical velocity ()
Temperature  (1440,22,64,128)
Gravity wave forcing on mean zonal flow, gwfu_cgwd (1440, 22, 64, 128)
Gravity wave forcing on mean meridional flow, gwfv_cgwd (1440,22,64,128)
"""

def visualize():
    gwfu = data.variables['gwfu_cgwd']
    gwfv = data.variables['gwfv_cgwd']
    pressure = data.variables['level']
    
    print(pressure[6])
    time_step = 0 # Value between 0 - 1439
    level = 10 # Value between 0 - 21 
    gwfu_plane = np.array(gwfu[time_step][level])
    gwfv_plane = np.array(gwfv[time_step][level])

    # Create Figure
    fig, axs = plt.subplots(2,2)
    # Create Histogram of Data
    bins = 50
    hist = np.histogram(gwfu, bins=50, density=True)
    vals, edges = hist
    axs[0,0].plot(edges[:50], vals)
    axs[0,0].set_title('PDF GWFU')
    """
    Plot GWDFU at each Long for fixed Lat, Time, & Pressure
    """
    axs[0,1].set_ylabel('GWDFU (m/s^2)')
    axs[0,1].set_xlabel('Long (2.8 degree)')
    axs[0,1].set_title('GWFU vs Long. for fixed Lat, Time & Pressure')
    for i in range(0, 3): axs[0,1].plot(gwfu_plane[i])
    """
    Plot GWDFU at each Lat for fixed Long, Time & Pressure
    """
    axs[1,0].set_ylabel('GWDFU (m/s^2)')
    axs[1,0].set_xlabel('Lat') 
    axs[1,0].set_title('GWFFU vs Lat. for Fixed Long, Time & Pressure')
    for i in range(0, 10): axs[1,0].plot(gwfu_plane[:,i])

    axs[1,1].set_ylabel('Mean gwfu')
    axs[1,1].set_xlabel('Pressure Levels')
    axs[1,1].set_title('Mean gwfu vs Pressure')
    means = np.mean(gwfu[0], axis=(1,2))
    for i in range(0,3): axs[1,1].plot(means)

    for ax in axs.flat:
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-6, 10))

    plt.show()

# visualize()

class Animator:
    def __init__(self, X, Y, Xlabel, Ylabel, Zlabel, Title):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d')
        # self.ax.set_zlim3d( -50e-8, 50e-8)
        self.ax.set_zlim3d( -25e-5, 25e-5)
        self.ax.set_ylabel(Ylabel)
        self.ax.set_xlabel(Xlabel)
        self.ax.set_zlabel(Zlabel)
        self.ax.set_title(Title)
        self.ax.w_zaxis.set_major_locator( LinearLocator( 10 ) )
        self.ax.w_zaxis.set_major_formatter( FormatStrFormatter( '%.03f' ) )

        self.X, self.Y = np.meshgrid(X, Y)

        Z = np.zeros(self.X.shape)
        self.surf = self.ax.plot_surface( 
            self.X, self.Y, Z, rstride=1, cstride=1, 
            cmap=cm.jet, linewidth=0, antialiased=False )
        # plt.draw() maybe you want to see this frame?

    def drawNow(self, Z): 
        self.surf.remove()
        self.surf = self.ax.plot_surface( 
            self.X, self.Y, Z, rstride=1, cstride=1, 
            cmap=cm.jet, linewidth=0, antialiased=False )
        plt.draw()                      # redraw the canvas
        self.fig.canvas.flush_events()
        time.sleep(.1)

matplotlib.interactive(True)


def animate_const_level(): 
    """
    This function animates the gwfu values at a constant pressure level. k
    """
    level = 5
    latX = data.variables['lat']
    longY = data.variables['lon']
    Ylabel = 'Longitude (0 - 360)'
    Xlabel = 'Latitude (-90 - 90 )'
    Zlabel = 'GWD on Zonal Flow (m/s^2)'
    title = 'Gravity Wave Drag on Zonal Flow at Fixed Height'
    p = Animator(latX, longY, Xlabel, Ylabel, Zlabel, title)
    gwfu = data.variables['gwfu_cgwd']
    pressure = data.variables['level']
    # print(pressure)
    # print(np.array(pressure))
    # for t in range(1440):
    #     spatial_frame = np.transpose(np.array(gwfu[t][level]))
    #     print("Starting time frame ", t)
    #     p.drawNow(spatial_frame)

def animate_const_lon():
    """
    This function animates gwfu at a const longitude
    """
    lon = 4
    latX = data.variables['lat']
    levelY = data.variables['level']
    Ylabel = 'Pressure Level (Hpa)'
    Xlabel = 'Latitude (-90 - 90 )'
    Zlabel = 'GWD on Zonal Flow (m/s^2)'
    title = 'Gravity Wave Drag on Zonal Flow at Fixed Longitude'
    p = Animator(latX, levelY, Xlabel, Ylabel, Zlabel, title)
    gwfu = data.variables['gwfu_cgwd']
    d = np.array(gwfu[0,:,:,lon])

    for t in range(1440):
        spatial_frame = np.array(gwfu[t,:,:,lon])
        print("Starting time frame ", t)
        p.drawNow(spatial_frame)
    
# animate_const_lon()
# animate_const_level()




def dim_info():
    for key in data.dimensions.keys():
        dim = data.dimensions[key]
        print(dim.group())
