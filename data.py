"""Functions for downloading and reading pyJHTDB"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import ctypes
import math
import numpy as np
from six.moves import xrange
import scipy.interpolate as itp
import pyJHTDB
import pyJHTDB.dbinfo
import pyJHTDB.interpolator

Image_size = 9
NUM_b = 9
filename = 'myfile.hdf5'
figname  = 'kin_en_contours'
N = 64
spacing=math.pi*2.**(-9)
xoff = 4.0 * math.pi
yoff = -1 # from the lower boundary
zoff = 1.5 * math.pi - spacing * N/2
y = spacing * np.arange(0, N, 1, dtype='float32') + yoff
z = spacing * np.arange(0, N, 1, dtype='float32') + zoff
# A surface perpendicular to X-axis 
surface = np.empty((N, N, 3),dtype='float32')
surface[:, :, 0] = xoff
surface[:, :, 1] = y[:, np.newaxis]
surface[:, :, 2] = z[np.newaxis, :]

if pyJHTDB.found_matplotlib:
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm
else:
  print('matplotlib is needed for contour plots.'
      + 'You should be able to find installation instructions at http://matplotlib.sourceforge.net')

if pyJHTDB.found_h5py:
  import h5py
  import pyJHTDB.cutout
else:
  print('h5py is needed for working with cutouts.')
  
lTDB = pyJHTDB.libJHTDB()
lTDB.initialize()

def get_velocity(step=0, spatialInterp=6, temporalInterp=0):
  time = 0.0065 * step
  result_vel = lTDB.getData(time, surface,
          sinterp = spatialInterp, tinterp = temporalInterp,
          data_set = 'channel', # for a non-zero Reynolds Stress
          getFunction = 'getVelocity')

  return result_vel

def get_velocitygradient(batch_size=2, spatialInterp=6, temporalInterp=0, \
                spacing=math.pi*2.**(-9), \
                FD4Lag4=40):
  time = 0.0065 * np.random.randint(4000)
  points = np.empty((batch_size,3),dtype = 'float32')
  y_pos  = np.empty((batch_size),dtype= 'float32') 
  points[:,0] = 8 * math.pi *np.random.random_sample(size = (batch_size))[:]     
  points[:,1] = 2.0 * np.random.random_sample(size = (batch_size))[:] -1 # [-1,1]
  y_pos[:]    = points[:,1]
  for p in range(batch_size):
    if y_pos[p] > 0:
      y_pos[p] = 1 - y_pos[p]
    else:
      y_pos[p] = y_pos[p]+1
  print(y_pos)
  points[:,2] = 3 * math.pi *np.random.random_sample(size = (batch_size))[:]
  startTime = 0.002 * np.random.randint(1024)
  endTime = startTime + 0.012
  lag_dt = 0.0004

  result_gra = lTDB.getData(time, points,
          sinterp = FD4Lag4, tinterp = temporalInterp,
          data_set = 'channel', # for a non-zero Reynolds Stress 
          getFunction = 'getVelocityGradient') 
  result_vel = lTDB.getData(time, points,
          sinterp = spatialInterp, tinterp = temporalInterp,
          data_set = 'channel', # for a non-zero Reynolds Stress 
          getFunction = 'getVelocity')
  tran   = result_gra.reshape((batch_size, 3, 3))
  Strain = np.empty([batch_size,3,3])
  Omega  = np.empty([batch_size,3,3])
  for p in range(batch_size):
    Strain[p,:,:] = 0.5*(tran[p,:,:]+tran[p,:,:].T)
    Omega[p,:,:]  = 0.5*(tran[p,:,:]-tran[p,:,:].T)
    """
    print('{0}: '.format(p) +
              'S_11 = {0:+e}, S_12 = {1:+e}, S_13 = {2:+e}\n   '.format(Strain[p,0,0], Strain[p,0,1], Strain[p,0,2]) +
              'S_21 = {0:+e}, S_22 = {1:+e}, S_23 = {2:+e}\n   '.format(Strain[p,1,0], Strain[p,1,1], Strain[p,1,2]) +
              'S_31 = {0:+e}, S_32 = {1:+e}, S_33 = {2:+e}\n   '.format(Strain[p,2,0], Strain[p,2,1], Strain[p,2,2]))  
    print('{0}: '.format(p) +
              'W_11 = {0:+e}, W_12 = {1:+e}, W_13 = {2:+e}\n   '.format(Omega[p,0,0], Omega[p,0,1], Omega[p,0,2]) +
              'W_21 = {0:+e}, W_22 = {1:+e}, W_23 = {2:+e}\n   '.format(Omega[p,1,0], Omega[p,1,1], Omega[p,1,2]) +
              'W_31 = {0:+e}, W_32 = {1:+e}, W_33 = {2:+e}\n   '.format(Omega[p,2,0], Omega[p,2,1], Omega[p,2,2]))
    """
  return result_vel,\
          result_gra, \
          Strain.reshape((batch_size, 9)), \
          Omega.reshape((batch_size, 9)), \
          y_pos

def get_lambdas_tensors(S,R):
  lam = np.empty([Image_size])
  TM  = np.empty([NUM_T,3,3])
  #T   = np.empty([NUM_T,NUM_b])
  lam[0] = np.trace(S.dot(S))
  lam[1] = np.trace(R.dot(R))
  lam[2] = np.trace(S.dot(S).dot(S))
  lam[3] = np.trace(R.dot(R).dot(S))
  lam[4] = np.trace(R.dot(R).dot(S).dot(S))
  TM[0,:,:] = S
  TM[1,:,:] = S.dot(R) - R.dot(S)
  TM[2,:,:] = S.dot(S) - 1/3*np.eye(3)*lam[0]
  TM[3,:,:] = R.dot(R) - 1/3*np.eye(3)*lam[1]
  TM[4,:,:] = R.dot(S).dot(S) - S.dot(S).dot(R)
  TM[5,:,:] = R.dot(R).dot(S) + S.dot(R).dot(R) - 2/3*np.eye(3)*np.trace(S.dot(R).dot(R))
  TM[6,:,:] = R.dot(S).dot(R).dot(R) - R.dot(R).dot(S).dot(R)
  TM[7,:,:] = S.dot(R).dot(S).dot(S) - S.dot(S).dot(R).dot(S)
  TM[8,:,:] = R.dot(R).dot(S).dot(S) + S.dot(S).dot(R).dot(R) -\
                                          2/3*np.eye(3)*np.trace(S.dot(S).dot(R).dot(R))
  TM[9,:,:] = R.dot(S).dot(S).dot(R).dot(R) - R.dot(R).dot(S).dot(S).dot(R)
  #print(TM)
  #T = np.reshape(TM,(-1,NUM_b))
  return lam, np.reshape(TM,(-1,NUM_b))

class Reynolds_Stress(object):
  def __init__(self):
    nu     = 5e-5
    u_tau  = 4.99e-2
    Re_tau = 9.9935e2 
    with open('profiles.csv',encoding='utf-8') as f:
      reader = csv.DictReader(f)
      self._y_plus  = np.array([float(row['y+']) for row in reader])
    with open('profiles.csv',encoding='utf-8') as f:
      reader = csv.DictReader(f)
      self._U_plus  = np.array([float(row['U+']) for row in reader])
    with open('profiles.csv',encoding='utf-8') as f:
      reader = csv.DictReader(f)
      self._uv      = np.array([float(row['uv+']) for row in reader])
    with open('profiles.csv',encoding='utf-8') as f:
      reader = csv.DictReader(f)
      self._uu = np.array([float(row['uu+']) for row in reader])
    with open('profiles.csv',encoding='utf-8') as f:
      reader = csv.DictReader(f)
      self._vv = np.array([float(row['vv+']) for row in reader])
    with open('profiles.csv',encoding='utf-8') as f:
      reader = csv.DictReader(f)
      self._ww = np.array([float(row['ww+']) for row in reader])
    self._size = self._y_plus.size
    self._y  =  self._y_plus[:] * nu / u_tau # from y^plus to y
    self._U  =  self._U_plus[:] * u_tau
    self._k  =  0.5*(self._uu[:] + self._vv[:] + self._ww[:])
    self._b  =  np.zeros([self._size,NUM_b])
    #print(self._ww)
    self._b[:,0]  = 0.5*self._uu[:]/self._k[:] - 1/3
    self._b[:,1]  = 0.5*self._uv[:]/self._k[:]
    self._b[:,3]  = 0.5*self._uv[:]/self._k[:]
    self._b[:,4]  = 0.5*self._vv[:]/self._k[:] - 1/3
    self._b[:,8]  = 0.5*self._ww[:]/self._k[:] - 1/3
    # interpolate
    self._U_interpolate  = itp.splrep(self._y,self._U)
    self._uv_interpolate = itp.splrep(self._y,self._uv)
    self._uu_interpolate = itp.splrep(self._y,self._uu)
    self._vv_interpolate = itp.splrep(self._y,self._vv)
    self._ww_interpolate = itp.splrep(self._y,self._ww)
    # interpolate
    self._b_0 = itp.splrep(self._y,self._b[:,0])
    self._b_1 = itp.splrep(self._y,self._b[:,1])
    self._b_4 = itp.splrep(self._y,self._b[:,4])
    self._b_8 = itp.splrep(self._y,self._b[:,8])

  @property
  def y(self):
    return self._y

  @property
  def uv(self):
    return self._uv

  @property
  def uu(self):
    return self._uu

  @property
  def vv(self):
    return self._vv

  @property
  def ww(self):
    return self._ww

  @property
  def k(self):
    return self._k

  @property
  def b(self):
    return self._b

  def get_U(self,points):
    U_tensor = itp.splev(points,self._U_interpolate)
    return U_tensor

  def get_b(self,points,batch_size):
    b_tensor = np.zeros([batch_size,NUM_b])
    b_tensor[:,0] = itp.splev(points,self._b_0)
    b_tensor[:,1] = itp.splev(points,self._b_1)
    b_tensor[:,3] = itp.splev(points,self._b_1)
    b_tensor[:,4] = itp.splev(points,self._b_4)
    b_tensor[:,8] = itp.splev(points,self._b_8)
    return b_tensor

class DataVel(object):
  def __init__(self,container_size=400):
    Vel = np.empty([3*N*N, container_size])
    f = h5py.File(filename)
    try:        
      f.keys()
      Vel    = f["Vel"][:,:]
      f.close()
      print("Load from saved JHTDB")
    except:
      f.close()
      reynolds_stress = Reynolds_Stress()
      y_pos = np.zeros([N,N])
      for i in range(N):
        for j in range(N):
          if surface[i, j, 1] > 0:
            y_pos[i,j] = 1 - surface[i, j, 1]
          else:
            y_pos[i,j] = surface[i, j, 1]+1
      get_U = reynolds_stress.get_U(y_pos[:, :].reshape((N*N)))
      U_average = get_U.reshape((N,N))
      for i in range(container_size):
        velocity = get_velocity(step=i)
        velocity[:,:,0] = velocity[:,:,0] - U_average[:,:]
        energy = .5*(np.sqrt(velocity[:,:,0]**2 + velocity[:,:,1]**2 + velocity[:,:,2]**2))
        fig = plt.figure(figsize=(6.,6.))
        ax = fig.add_axes([.0, .0, 1., 1.])
        contour = ax.contour(z, y, energy, 30)
        ax.set_xlabel('z')
        ax.set_ylabel('y')
        plt.clabel(contour, inline=1, fontsize=10)
        plt.title('Energy contours, t = {0}, x = {1:.3}'.format(i, xoff))
        fig.savefig('./images/'+figname + str(i)+'.eps', format = 'eps', bbox_inches = 'tight') 
        Vel[:, i] = velocity.reshape((N*N*3))
      print("Download from JHTDB server")
      f = h5py.File(filename, 'w')
      f.create_dataset('Vel'   , data=Vel   ,dtype='f')
      f.close()
      print("Save the JHTDB data")
    print(Vel.shape)  
    self._Vel    = Vel

  @property
  def Vel(self):
    return self._Vel

class Data(object):
  def __init__(self,container_size=1000,batch_size=1, fake_data=False):
    if fake_data:
      fake_Aij = [1] * Image_size
      fake_stress = [0.5] * NUM_b
      self._Aij = [fake_Aij for _ in xrange(batch_size)]
      self._stress = [fake_stress for _ in xrange(batch_size)]      
    else:
      reynolds_stress    = Reynolds_Stress()
      Vel     = np.empty([container_size*batch_size,3])
      Aij     = np.empty([container_size*batch_size,Image_size])
      stress  = np.zeros([container_size*batch_size,NUM_b])
      Strain  = np.empty([container_size*batch_size,Image_size])
      Omega   = np.empty([container_size * batch_size,Image_size])
      f = h5py.File(filename, 'r')
      try:        
        f.keys()
        Vel    = f["Vel"][:,:]
        Aij    = f["Aij"][:,:]
        Strain = f["Strain"][:,:]
        Omega  = f["Omega"][:,:]
        stress = f["stress"][:,:]
        f.close()
        print("Load from saved JHTDB")
      except:
        f.close()
        for i in range(container_size):
          velocity = get_velocity(step=i)
          stress[i*batch_size:(i+1)*batch_size,:] = reynolds_stress.get_b(y_pos,batch_size)
          Vel[i*batch_size:(i+1)*batch_size,:]    = velocity
          Aij[i*batch_size:(i+1)*batch_size,:]    = velocitygradient
          Strain[i*batch_size:(i+1)*batch_size,:] = St
          Omega[i*batch_size:(i+1)*batch_size,:]  = Om
        print("Download from JHTDB server")
        f = h5py.File(filename, 'w')
        f.create_dataset('Vel'   , data=Vel   ,dtype='f')
        f.create_dataset('Aij'   , data=Aij   ,dtype='f')      
        f.create_dataset('Strain', data=Strain,dtype='f')
        f.create_dataset('Omega' , data=Omega, dtype='f')
        f.create_dataset('stress', data=stress, dtype='f')
        f.close()
        print("Save the JHTDB data")
      print(Vel.shape)  
      print(Aij.shape)
      print(stress.shape)
      print(Strain.shape)
      print(Omega.shape)
      self._Vel    = Vel
      self._Aij    = Aij
      self._stress = stress
      self._Strain = Strain
      self._Omega  = Omega

  @property
  def Vel(self):
    return self._Vel

  @property
  def Aij(self):
    return self._Aij

  @property
  def stress(self):
    return self._stress

  @property
  def Strain(self):
    return self._Strain

  @property
  def Omega(self):
    return self._Omega


