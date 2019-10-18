"""This module contains a pure python implementation of the basic
cross-correlation algorithm for PIV image processing."""

__licence_ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import scipy
import scipy.special
import scipy.interpolate
from scipy import io
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from PIL import Image

class continuous_flow_field:
    def __init__(self,data,inter=False):
        '''
        Checks if the continous flow should be created from a set of data points
        if so it interpolates them for a continuous flow field
        '''
        self.inter = inter
        if inter:
            self.f_U = scipy.interpolate.interp2d(data[:,0],data[:,1],data[:,2])
            self.f_V = scipy.interpolate.interp2d(data[:,0],data[:,1],data[:,3])
   
    
    '''
    Defining a synthetic flow field
    '''
    def f_U(self,x,y):
        #example for synthetic U velocity
        u=2.5+0.5*np.sin((x**2+y**2)/0.01)
        return u
        
    
    def f_V(self,x,y):
        #example for synthetic V velocity
        v=0.5+0.1*np.cos((x**2+y**2)/0.01)
        return v
    
    def get_U_V(self,x,y):
    	#return the U and V velocity at a certain position
        if self.inter:
            return self.f_U(x,y)[0],self.f_V(x,y)[0]
        else:
            return self.f_U(x,y),self.f_V(x,y)
        
    def create_syn_quiver(self,number_of_grid_points,path=None):
    	#return and save a synthetic flow map 
        X,Y = np.meshgrid(np.linspace(0,1,number_of_grid_points),np.linspace(0,1,number_of_grid_points))
        U = np.zeros(X.shape)
        V = np.zeros(Y.shape)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                u,v = self.get_U_V(X[r,c],Y[r,c])
                U[r,c] = u
                V[r,c] = v
        

        m = np.sqrt(np.power(U, 2) + np.power(V, 2))
        fig = pl.quiver(X, Y, U, V,m,clim=[1.5,m.max()],scale=100,width=0.002,headwidth=6,minshaft=2)
        cb = pl.colorbar(fig)
        cb.set_clim(vmin=1.5, vmax=m.max())
    
        
        if not path:
            pl.savefig('syn_quiver.png', dpi=400)
            pl.close()
        else:
            pl.savefig(path + 'syn_quiver.png', dpi=400)
            pl.close()
        
        return X,Y,U,V


def create_synimage_parameters(input_data,x_bound,y_bound,image_size,path='None',inter=False,den=0.008,per_loss_pairs=2,par_diam_mean=15**(1.0/2),par_diam_std=1.5,par_int_std=0.25,dt=0.1):
    """Creates the synthetic image with the synthetic image parameters

    Parameters
    ----------
    input_data: None or numpy array
        If you have data from which to genrate the flow feild the synthetic image.
        It should be passed on as a numpy array with columns being (X grid position,Y grid position,U velocity at (X,Y) grid point,V velocity at (X,Y) grid point)
        Else, pass None and define a synthetic flow field in continuous_flow_field class.

    x_bound,y_bound: list/tuple of floats
        The boundries of interest in the synthetic flow field.

    image_size: list/tuple of ints
        The desired image size in pixels.
    
    path: str('None' for no generating data)
        Path to txt file of input data.

    inter: boolean
        False if no interpolation of input data is needed.
        True if there is data you want to interpolate from.   

    den: float
        Defines the number of particles per image.

    per_loss_pairs: float
        Percentage of synthetic pairs loss.

    par_diam_mean: float
        Mean particle diamter in pixels.

    par_diam_std: float
        Standard deviation of particles diamter in pixels.

    par_int_std: float
        Standard deviation of particles intensities.

    dt: float
        Synthetic time difference between both images.

    Returns
    -------
    ground_truth: continuous_flow_field class
        The synthetic ground truth as a continuous_flow_field class.

    cv:
        Convertion value to convert U,V from pixels/images to meters/seconds.

    x_1,y_1: numpy array
        Position of particles in the first synthetic image.

    U_par,V_par: numpy array
        Velocity speeds for each particle.

    par_diam1: numpy array
        Particle diamters for the first synthetic image.

    par_int1: numpy array
        Particle intensities for the first synthetic image.

    x_2,y_2: numpy array
        Position of particles in the second synthetic image.

    par_diam2: numpy array
        Particle diamters for the second synthetic image.

    par_int2: numpy array
        Particle intensities for the second synthetic image.
    """

    #Data processing
    
    if not path == 'None':
        f = open(path,'r')
        data = f.readlines()
        f.close()
        data = [line.split('\t') for line in data]
        data = np.array(data).astype(float)
        data = np.array([line for line in data.tolist() if 1.2*x_bound[1]>=line[1]>=0.8*x_bound[0] and 1.2*y_bound[1]>=line[2]>=0.8*y_bound[0]])
        
    else:
        data = input_data
        
    if inter:
        cff = continuous_flow_field(data,inter=True)
    else:
        cff = continuous_flow_field(None)
    
    #Creating syn particles
    
    num_of_par = int(image_size[0]*image_size[1]*den)
    num_of_lost_pairs = num_of_par*(per_loss_pairs/100)
    x_1 = np.random.uniform(x_bound[0]*0.8,x_bound[1]*1.2,num_of_par)
    y_1 = np.random.uniform(y_bound[0]*0.8,y_bound[1]*1.2,num_of_par)
    par_diam1 = np.random.normal(par_diam_mean,par_diam_std,num_of_par)
    particleCenters = np.random.uniform(size=num_of_par)-0.5
    par_int1 = np.exp(-particleCenters**2/(2*par_int_std**2))
    U_par = np.zeros(x_1.shape)
    V_par = np.zeros(y_1.shape)
    x_2 = np.zeros(x_1.shape)
    y_2 = np.zeros(y_1.shape)
    par_diam2 = np.zeros(par_diam1.shape)
    par_int2 = np.zeros(par_int1.shape)
    
    def Move_par(i):
        U_par[i],V_par[i] = cff.get_U_V(x_1[i],y_1[i])
        x_2[i] = x_1[i]+U_par[i]*dt
        y_2[i] = y_1[i]+V_par[i]*dt
        par_diam2[i] = par_diam1[i]
        par_int2[i] = par_int1[i]
        
    cpl = 0
    for i in range(num_of_par):
        if cpl<num_of_lost_pairs:
            if -0.4>particleCenters[i] or 0.4<particleCenters[i]:
                per_to_lose = 1-(0.5 - np.abs(particleCenters[i]))/0.1
                if np.random.uniform()<min(per_loss_pairs/10,1)*per_to_lose:
                    x_2[i] = np.random.uniform(x_bound[0]*0.8,x_bound[1]*1.2)
                    y_2[i] = np.random.uniform(y_bound[0]*0.8,y_bound[1]*1.2)
                    par_diam2[i] = np.random.normal(par_diam_mean,par_diam_std)
                    par_int2[i] = np.exp(-(np.random.uniform()-0.5)**2/(2*par_int_std**2))
                    cpl+=1
                else:
                    Move_par(i)
            else:
                Move_par(i)
        else:
            Move_par(i)
    
    print('Requested pair loss:',str(int(num_of_lost_pairs)),' Actual pair loss:',str(cpl))
    xy_1 = np.transpose(np.vstack((x_1,y_1,U_par,V_par,par_diam1,par_int1)))
    xy_2 = np.transpose(np.vstack((x_2,y_2,par_diam2,par_int2)))
    
    #Choosing particles in boundary area
    
    bounded_xy_1 = np.asarray([xy for xy in xy_1 if x_bound[1]>=xy[0]>=x_bound[0] and y_bound[1]>=xy[1]>=y_bound[0]])
    bounded_xy_2 = np.asarray([xy for xy in xy_2 if x_bound[1]>=xy[0]>=x_bound[0] and y_bound[1]>=xy[1]>=y_bound[0]])
    
    #Tranforming coordinates into pixels
    
    x1 = ((bounded_xy_1[:,0]-x_bound[0])/(x_bound[1]-x_bound[0]))*image_size[0]
    y1 = ((bounded_xy_1[:,1]-y_bound[0])/(y_bound[1]-y_bound[0]))*image_size[1]

    x2 = ((bounded_xy_2[:,0]-x_bound[0])/(x_bound[1]-x_bound[0]))*image_size[0]
    y2 = ((bounded_xy_2[:,1]-y_bound[0])/(y_bound[1]-y_bound[0]))*image_size[1]
    
    conversion_value = min((x_bound[1]-x_bound[0])/image_size[0],(y_bound[1]-y_bound[0])/image_size[1])/dt
    
    return cff,conversion_value,x1,y1,bounded_xy_1[:,2],bounded_xy_1[:,3],bounded_xy_1[:,4],bounded_xy_1[:,5],x2,y2,bounded_xy_2[:,2],bounded_xy_2[:,3]



def generate_particle_image(HEIGHT, WIDTH, X, Y, PARTICLE_DIAMETERS, PARTICLE_MAX_INTENSITIES,BIT_DEPTH):
    """Creates the synthetic image with the synthetic image parameters
    Should be run with the parameters of each image (first,second) separately.

    Parameters
    ----------
    HEIGHT, WIDTH: int
        The number of pixels in the desired output image.

    X,Y: numpy array
        The X and Y positions of the particles, created by create_synimage_parameters().

    PARTICLE_DIAMETERS, PARTICLE_MAX_INTENSITIES: numpy array
		The intensities and diameters of the particles, created by create_synimage_parameters().
	
	BIT_DEPTH: int
		The bit depth of the desired output image.

    Returns
    -------
    Image: numpy array
        The desired synthetic image.

    """
    render_fraction = 0.75
    IMAGE_OUT = np.zeros([HEIGHT, WIDTH])

    minRenderedCols = (X - render_fraction * PARTICLE_DIAMETERS).astype(int)
    maxRenderedCols = (np.ceil(X + render_fraction * PARTICLE_DIAMETERS)).astype(int)
    minRenderedRows = (Y - render_fraction * PARTICLE_DIAMETERS).astype(int)
    maxRenderedRows = (np.ceil(Y + render_fraction * PARTICLE_DIAMETERS)).astype(int)

    index_to_render = []

    for i in range(X.size):
        if 1<minRenderedCols[i] and maxRenderedCols[i]< WIDTH and 1<minRenderedRows[i] and maxRenderedRows[i]< HEIGHT:
            index_to_render.append(i)

    for i in range(len(index_to_render)):
        ind = index_to_render[i]
        max_int = PARTICLE_MAX_INTENSITIES[ind]
        par_diam = PARTICLE_DIAMETERS[ind]
        sqrt8 = np.sqrt(8)
        x = X[ind]
        y = Y[ind]

        bl = max(minRenderedCols[ind],0)
        br = min(maxRenderedCols[ind],WIDTH)
        bu = max(minRenderedRows[ind],0)
        bd = min(maxRenderedRows[ind],HEIGHT)

        for c in range(bl,br):
            for r in range(bu,bd):
                IMAGE_OUT[r,c] = IMAGE_OUT[r,c] + max_int * par_diam**2 * np.pi / 32 * \
                ( scipy.special.erf( sqrt8 * (c - x - 0.5) / par_diam ) - scipy.special.erf(sqrt8 * (c - x + 0.5) / par_diam)) * \
                ( scipy.special.erf( sqrt8 * (r - y - 0.5)/ par_diam) - scipy.special.erf(sqrt8 *(r - y + 0.5) / par_diam))

    NOISE_MEAN = 2**(BIT_DEPTH*0.3)
    NOISE_STD = 0.25*NOISE_MEAN
    Noise = NOISE_STD * np.random.randn(HEIGHT, WIDTH) + NOISE_MEAN
    return (IMAGE_OUT*(2**BIT_DEPTH * 2.8**2/8)+Noise).astype(int)[::-1]