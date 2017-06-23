"""smp_sys.systems_bhasim.py

Mathias Schmerling's Python port of the CoR-Lab's Matlab model of the Bionic Handling Assistant (BHA)

.. moduleauthor:: Mathias Schmerling (BCCN Berlin, Adaptive Systems Group, HU Berlin), Oswald Berthold 2017

.. note::
    Original Matlab code::

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%          Copyright (c) 2013 R. F. Reinhart, K. Neumann, A. Lemme CoR-Lab          %%%
    %%%          Bielefeld University, Germany, http://cor-lab.de                         %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# from explauto.environment.environment import Environment
# from explauto.utils.utils import bounds_min_max

import time
import os
import sys
import numpy as np
from numpy import cos, sin, arctan2, pi


import matplotlib.pyplot as plt
#from matplotlib.patches import Circle, PathPatch
# from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.art3d as art3d

#from goalBabbling import *
#from forwardModel import *
#from inverseModel import *


from smp_sys.systems import SMPSys

def bounds_min_max(v, mins, maxs):
    res = np.minimum(v, maxs)
    res = np.maximum(res, mins)
    return res


def plot_taskspace_positions(positions, style='ob'):
    fig = plt.gcf()
    if positions.shape[1] == 3:
        ax = fig.gca(projection='3d')
        ax.plot(positions[:,0], positions[:,1], positions[:,2], style)
    elif positions.shape[1] == 2:
        ax = fig.gca()
        ax.plot(positions[:,0], positions[:,1], style)
    else:
        pass 


def func_MakeSeqFromGrid(X,Y,Z):
    x1, x2, x3 = np.shape(X)
    seq = []
    for i in range(x1):
        for j in range(x2):
            for k in range(x3):
                new_column = [X[i,j,k],Y[i,j,k],Z[i,j,k]]
                seq.append(new_column)
    return np.array(seq)


def func_FromSpericalCoords2CartesianCoords(pc):
    rad = pc[:,0]
    phi = pc[:,1]
    theta = pc[:,2]
    cc = np.zeros_like(pc)
    cc[:,0] = rad * sin(theta) * cos(phi)
    cc[:,1] = rad *sin(theta) * sin(phi)
    cc[:,2] = rad *cos(theta)
    return cc

"""
def plotCircle3D(ax, center, radius, normal=None):
    circle = Circle((center[0],center[1]), radius, fill=False)
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=center[2], zdir='z')
    return ax
"""

def null(A):
    u, s, v = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    return v[rank:].T.copy()

def plotCircle3D(ax, center, normal , radius):
    theta = np.arange(0,2*np.pi,0.1)
    nPoints = np.shape(theta)[0]
    v=null(np.array([normal]))
    center = center.reshape((-1,1))
    points=np.tile(center,nPoints) + radius * ( np.outer(v[:,0],cos(theta)) + np.outer(v[:,1],sin(theta)) )
    ax.plot(points[0,:],points[1,:],points[2,:],'k-')
    return ax

"""
function Z = func_MakeGridFromSeq(seq,sX,sY):
    
    Z = zeros(sX,sY);
        
    count = 1;
    for i = 1:sX
        for j = 1:sY
            Z(i,j) = seq(1,count);
            count = count + 1;
        end
    end

end
"""
"""
function phi = func_FromCartesianCoords2SpericalCoords(x):
    [sX,sY] = size(x);
    x1 = [1;0;0]; x2 = [0;1;0]; 
    phi = zeros(sX,sY);
    for i = 1:sX
        vec = x(i,:);
        phi(i,:) = [sign(vec(3))*norm(vec), acos((vec*x1)/(norm(vec))), acos((vec*x2)/(norm(vec)))];
    end
end

"""

# FIXME: explauto wrapper for smpsys?
# class BhaSimulated(Environment):
#     def __init__(self, m_mins, m_maxs, s_mins, s_maxs, numSegs, SegRadii):
#         Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
#         self.Segments = [];
#         self.numSegs = numSegs
#         self.minmaxLens = np.tile(np.array([[0.1, 0.3]]),(3*numSegs,1))
#         for i in range(numSegs):
#             self.Segments.append(Segment(SegRadii[i]))


###################################### new class definition #########################################
class BhaSimulatedSys(SMPSys):
    """BhaSimulated
    """
    # SegRadii = [0.1;0.093;0.079]; for real BHA robot

    useprocess = False

    defaults = {
        'sysdim': 1,
        'x0': np.random.uniform(-0.3, 0.3, (3, 1)),
        'statedim': 3,
        'dt': 1e-1,
        'mass': 1.0,
        "force_max":  1.0,
        "force_min": -1.0,
        "friction": 0.001,
        "sysnoise": 1e-2,
        # real
        'dim_s_motor': 9,
        'dim_s_extero': 3,
        'numsegs': 3,
        'segradii': np.array([0.1,0.093,0.079]),
        'm_mins': [0.05] * 9,
        'm_maxs': [0.4]  * 9,
        's_mins': [-1] * 9,
        's_maxs': [1]  * 9,
        }
    
    def __init__(self, conf = {}):
        SMPSys.__init__(self, conf)

        self.Segments = [];
        self.numSegs = self.numsegs
        self.SegRadii = self.segradii
        self.minmaxLens = np.tile(np.array([[0.1, 0.3]]),(3*self.numSegs,1))
        for i in range(self.numSegs):
            self.Segments.append(Segment(self.SegRadii[i]))
        

    def step(self, x):
        """update the robot, pointmass"""
        # print "%s.step x = %s" % (self.__class__.__name__, x)
        # print "x", x.shape
        # self.m = self.compute_motor_command(self.m + x)# .reshape((self.dim_s_motor, 1))
        self.lens = self.compute_motor_command(x.T)# .reshape((self.dim_s_motor, 1))
        
        # print "m", m
        # self.apply_force(x)
        return {"s_proprio": self.lens, # self.compute_sensors_proprio(),
                "s_extero":  self.compute_sensors_extero(),
                's_all':     self.compute_sensors(),
                }
    
    def compute_motor_command(self, lens):
        return bounds_min_max(lens, self.m_mins, self.m_maxs)

    def compute_sensors_extero(self):
        self.lens
        return self.compute_sensori_effect(self.lens)

    def compute_sensors_proprio(self):
        # hand_pos += 
        return self.lens + self.sysnoise * np.random.randn(*self.lens.shape)
    
    def compute_sensors(self):
        """compute the proprio and extero sensor values from state"""
        # return np.vstack((self.m, self.compute_sensors_extero()))
        prop = self.compute_sensors_proprio()
        exte = self.compute_sensors_extero()
        # print "prop", prop.shape, "exte", exte.shape
        ret = np.hstack((prop, exte)).T
        # print "ret", ret.shape
        return ret
        # return self.x    
    
    def compute_sensori_effect(self,lens):
        lens = np.real(lens)
        if lens.ndim == 2 and np.any(lens.shape == 1):
           lens = lens.flatten()

        if lens.ndim==1:
            eepos = np.zeros(3)
            for j in range(3*self.numSegs):
                lens[j] = np.clip(lens[j], self.minmaxLens[j,0], self.minmaxLens[j,1])
            eepos = self.fwdkinematicsUntil(np.zeros(3),lens,self.numSegs-1)   
         
        elif lens.ndim==2:
            p = np.shape(lens)[0]
            eepos = np.zeros((p,3))
            for i in range(p):
                for j in range(3*self.numSegs):
                    lens[:,j] = np.clip(lens[:,j], self.minmaxLens[j,0], self.minmaxLens[j,1])
                eepos[i,:] = self.fwdkinematicsUntil(np.zeros(3),lens[i,:],self.numSegs-1)

        return eepos #,lens
         

    def fwdkinematicsUntil(self,pos,lens,num):
        lens = np.reshape(lens,(3,self.numSegs), order='F') 
        for i in range(num,-1,-1):
            pos = self.Segments[i].calculateCoordTrafo(pos,lens[:,i])
        return pos
    
    def visSegStripes(self,lens,seg):
        numSamples = 10
        lens = np.reshape(lens,(3,self.numSegs), order='F')
        dashes = np.zeros((self.numSegs,numSamples))
        dashes[0,:] = np.linspace(lens[0,seg],0,numSamples)
        dashes[1,:] = np.linspace(lens[1,seg],0,numSamples)
        dashes[2,:] = np.linspace(lens[2,seg],0,numSamples)
        #dashes = [np.linspace(lens[0,seg],0,numSamples) , linspace(lens[1,seg],0,numSamples) , linspace(lens[2,seg],0,numSamples)]
        pos1 = np.zeros((self.numSegs,numSamples))
        pos2 = np.zeros((self.numSegs,numSamples))
        pos3 = np.zeros((self.numSegs,numSamples))
        for i in range(numSamples):
            lensT = lens.copy()
            lensT[:,seg] = dashes[:,i]
            pos1[:,i] = self.fwdkinematicsUntil(self.Segments[seg].b * np.array([-1,0,0]), lensT.flatten(order='F'), seg)
            pos2[:,i] = self.fwdkinematicsUntil(self.Segments[seg].b * np.array([-cos(2*pi/3),sin(2*pi/3),0]), lensT.flatten(order='F'), seg)
            pos3[:,i] = self.fwdkinematicsUntil(self.Segments[seg].b * np.array([-cos(2*2*pi/3),sin(2*2*pi/3),0]), lensT.flatten(order='F'), seg)
        return pos1, pos2, pos3
      
    def visualize(self, ax, lens_ , **kwargs_plot):
        """Visualize BHA"""
        lens = self.compute_motor_command(lens_)

        if lens.ndim == 1:
            stripes31,stripes32,stripes33 = self.visSegStripes(lens,2)
            stripes21,stripes22,stripes23 = self.visSegStripes(lens,1)
            stripes11,stripes12,stripes13 = self.visSegStripes(lens,0)

            dir1 = self.fwdkinematicsUntil(np.array([0,0,1]),lens,0)
            dir2 = self.fwdkinematicsUntil(np.array([0,0,1]),lens,1)
            dir3 = self.fwdkinematicsUntil(np.array([0,0,1]),lens,2)
            pos1 = self.fwdkinematicsUntil(np.array([0,0,0]),lens,0)
            pos2 = self.fwdkinematicsUntil(np.array([0,0,0]),lens,1)
            pos3 = self.fwdkinematicsUntil(np.array([0,0,0]),lens,2)
            # ax = fig.gca(projection='3d')
            source = np.array([0,0,0])
            ax = plotCircle3D(ax, source, np.array([0,0,1]), self.Segments[0].b)

            ax = plotCircle3D(ax, pos1, dir1 - pos1, self.Segments[0].b)   #, normal = dir1.T-pos1.T);
            ax = plotCircle3D(ax, pos1, dir1 - pos1, self.Segments[1].b) # pos1,dir1.T-pos1.T,
            ax = plotCircle3D(ax, pos2, dir2 - pos2, self.Segments[1].b) # pos2,dir2.T-pos2.T,
            ax = plotCircle3D(ax, pos2, dir2 - pos2, self.Segments[2].b) # pos2,dir2.T-pos2.T,
            ax = plotCircle3D(ax, pos3, dir3 - pos3, self.Segments[2].b) # pos3,dir3.T-pos3.T,
            
            ax.plot(stripes31[0,:],stripes31[1,:],stripes31[2,:],'k')
            ax.plot(stripes32[0,:],stripes32[1,:],stripes32[2,:],'k')
            ax.plot(stripes33[0,:],stripes33[1,:],stripes33[2,:],'k')
            ax.plot(stripes21[0,:],stripes21[1,:],stripes21[2,:],'k')
            ax.plot(stripes22[0,:],stripes22[1,:],stripes22[2,:],'k')
            ax.plot(stripes23[0,:],stripes23[1,:],stripes23[2,:],'k')
            ax.plot(stripes11[0,:],stripes11[1,:],stripes11[2,:],'k')
            ax.plot(stripes12[0,:],stripes12[1,:],stripes12[2,:],'k')
            ax.plot(stripes13[0,:],stripes13[1,:],stripes13[2,:],'k')
            ax.axis([self.s_mins[0], self.s_maxs[0], self.s_mins[1], self.s_maxs[1]])
            source = np.array([source.copy()])
            pos1 = np.array([pos1.copy()])
            pos2 = np.array([pos2.copy()])
            pos3 = np.array([pos3.copy()])
            ax.plot(source[:,0],source[:,1],source[:,2],'ok')
            ax.plot(pos1[:,0],pos1[:,1],pos1[:,2],'ok')
            ax.plot(pos2[:,0],pos2[:,1],pos2[:,2],'ok')
            ax.plot(pos3[:,0],pos3[:,1],pos3[:,2],'or')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        elif lens.ndim==2:
            lens_copy = lens.copy()
            plt.ion()
            for i in range(lens_copy.shape[0]):
                lens = lens_copy[i,:]
                stripes31,stripes32,stripes33 = self.visSegStripes(lens,2)
                stripes21,stripes22,stripes23 = self.visSegStripes(lens,1)
                stripes11,stripes12,stripes13 = self.visSegStripes(lens,0)

                dir1 = self.fwdkinematicsUntil(np.array([0,0,1]),lens,0)
                dir2 = self.fwdkinematicsUntil(np.array([0,0,1]),lens,1)
                dir3 = self.fwdkinematicsUntil(np.array([0,0,1]),lens,2)
                pos1 = self.fwdkinematicsUntil(np.array([0,0,0]),lens,0)
                pos2 = self.fwdkinematicsUntil(np.array([0,0,0]),lens,1)
                pos3 = self.fwdkinematicsUntil(np.array([0,0,0]),lens,2)
                # ax = fig.gca(projection='3d')
                source = np.array([0,0,0])
                ax.cla()
                ax = plotCircle3D(ax, source, np.array([0,0,1]), self.Segments[0].b)

                ax = plotCircle3D(ax, pos1, dir1 - pos1, self.Segments[0].b)   #, normal = dir1.T-pos1.T);
                ax = plotCircle3D(ax, pos1, dir1 - pos1, self.Segments[1].b) # pos1,dir1.T-pos1.T,
                ax = plotCircle3D(ax, pos2, dir2 - pos2, self.Segments[1].b) # pos2,dir2.T-pos2.T,
                ax = plotCircle3D(ax, pos2, dir2 - pos2, self.Segments[2].b) # pos2,dir2.T-pos2.T,
                ax = plotCircle3D(ax, pos3, dir3 - pos3, self.Segments[2].b) # pos3,dir3.T-pos3.T,
                
                ax.plot(stripes31[0,:],stripes31[1,:],stripes31[2,:],'k')
                ax.plot(stripes32[0,:],stripes32[1,:],stripes32[2,:],'k')
                ax.plot(stripes33[0,:],stripes33[1,:],stripes33[2,:],'k')
                ax.plot(stripes21[0,:],stripes21[1,:],stripes21[2,:],'k')
                ax.plot(stripes22[0,:],stripes22[1,:],stripes22[2,:],'k')
                ax.plot(stripes23[0,:],stripes23[1,:],stripes23[2,:],'k')
                ax.plot(stripes11[0,:],stripes11[1,:],stripes11[2,:],'k')
                ax.plot(stripes12[0,:],stripes12[1,:],stripes12[2,:],'k')
                ax.plot(stripes13[0,:],stripes13[1,:],stripes13[2,:],'k')
                ax.axis([self.s_mins[0], self.s_maxs[0], self.s_mins[1], self.s_maxs[1]])
                source = np.array([source.copy()])
                pos1 = np.array([pos1.copy()])
                pos2 = np.array([pos2.copy()])
                pos3 = np.array([pos3.copy()])
                ax.plot(source[:,0],source[:,1],source[:,2],'ok')
                ax.plot(pos1[:,0],pos1[:,1],pos1[:,2],'ok')
                ax.plot(pos2[:,0],pos2[:,1],pos2[:,2],'ok')
                ax.plot(pos3[:,0],pos3[:,1],pos3[:,2],'or')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.draw()
                time.sleep(0.5)
            plt.ioff()

###################################### end of class definition ######################################




###################################### new class definition #########################################
class Segment(object):
    """Segment
    """
    def __init__(self,b):
        self.b = b


    def calculateGeoParams(self, lens):
        g = np.sqrt(np.sum(lens**2) - (lens[0]*lens[1]+lens[0]*lens[2]+lens[1]*lens[2]) ) # what does np.sum do here? axiswise?
        theta = (2*g)/(3*self.b)
        phi = arctan2(np.real(np.sqrt(3)*(lens[2]-lens[1])),np.real(lens[1]+lens[2]-2*lens[0]))
        return theta, phi
        
    def calculateCoordTrafo(self,p1,lens):
        theta, phi = self.calculateGeoParams(lens)
        lhat = np.sum(lens)/3
        p0 = np.dot(self.Rot(phi,theta),p1) + np.array([lhat*self.sinX(theta/2)*cos((pi-theta)/2)*sin(phi-(pi/2)),
                                                       -lhat*self.sinX(theta/2)*cos((pi-theta)/2)*cos(phi-(pi/2)),
                                                        lhat*self.sinX(theta/2)*sin((pi-theta)/2)])
        return p0
        

    def Rot(self,phi,theta):
        a11 = cos(phi)*cos(theta)*cos(phi) + sin(phi)*sin(phi)
        a12 = cos(phi)*cos(theta)*sin(phi) - sin(phi)*cos(phi)
        a13 = -cos(phi)*sin(theta)
        a21 = sin(phi)*cos(theta)*cos(phi) - cos(phi)*sin(phi)
        a22 = sin(phi)*cos(theta)*sin(phi) + cos(phi)*cos(phi)
        a23 = - sin(phi)*sin(theta)
        a31 = sin(theta)*cos(phi)
        a32 = sin(theta)*sin(phi)
        a33 = cos(theta)
        return np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
        
        
    def sinX(self,alpha):
            if sin(alpha) == 0 and np.abs(alpha) < pi:
                return 1
            else:
                return sin(alpha)/alpha
###################################### end of class definition ######################################





	
def main():

    # def __init__(self, m_mins, m_maxs, s_mins, s_maxs, numSegs, SegRadii):
    # bha = Environment.from_configuration('bhasimulated', 'low_dimensional')    
    n = 3;
    m = 9;
    np.random.seed(87123)
    maxIterations = 54

    # numSegs = 3
    # self.SegRadii = np.array([0.1,0.093,0.079])
    # bha = BhaSimulated(numSegs,SegRadii)
    # bha = BhaSimulated([-1] * 9, [1] * 9, [-1] * 9, [1] * 9, numSegs, SegRadii)
    bha = BhaSimulatedSys(conf = BhaSimulatedSys.defaults)
    

    # first home posture 
    qhome = 0.2*np.ones(m)
    # xhome,qhome = bha.execute(qhome)


    plt.ion()
    fig = plt.figure()
    fig.show()
    ax = fig.gca(projection='3d')
    
    for i in range(10):
        lens_ = np.random.uniform(0.1, 1, (1, 9))
        lens = bha.compute_motor_command(lens_)
        eepos = bha.compute_sensori_effect(lens)
        print "bha[%d] m = %s, eepos = %s" % (i, lens, eepos)

        bha.visualize(ax, lens)
        plt.draw()
        plt.pause(1e-6)

    plt.show()
    
    # print 'homeposition'
    # print 'xhome:'
    # print xhome
    # print 'qhome:'
    # print qhome

    # inverseModel = LLM(n, m, qhome, xhome)
    # inverseModel.lrate = 0.05
    # inverseModel.radius = 0.1
    # X,Y,Z = np.meshgrid(np.arange(0.735,0.8250+0.01,0.045),  
    #                     np.arange(-pi,pi+0.01,pi/7), 
    #                     np.arange((5.14*2*pi/360),(36*2*pi/360)+0.0001,(5.14*2*pi/360))   )
    # taskspace = func_MakeSeqFromGrid(X,Y,Z)
    # taskspace = func_FromSpericalCoords2CartesianCoords(taskspace)
    

    # gb = goalBabbling_RolfSteil(inverseModel, bha, qhome, taskspace, numSteps=10)
    # gb.sigma = 0.05 / np.sqrt(m-1) # simple heuristic to adjust noise for varying m
    # gb.sigmaDelta = 0.1 * gb.sigma

    # for i in range(maxIterations):
    #     print '#################### iteration ' + str(i) + ' ###################'
    #     gb.iterate()
    #     #time.sleep(5)
    # #gb.visualize()
    # print taskspace.shape
    # fig = plt.figure()
    # #ax = fig.gca(projection='3d')
    # #ax.plot(taskspace[:,0], taskspace[:,1], taskspace[:,2], 'ob')
    # plot_taskspace_positions(positions=taskspace, style='ob')
    # motor_estimates = gb.inverseModel.predict(taskspace)
    # target_estimates, motor_estimates = gb.forwardModel.execute(motor_estimates)
    # #print gb.inverseModel.prototypes.shape
    # #print target_estimates.shape
    # #print target_estimates
    # #print motor_estimates
    # #plot_taskspace_positions(positions=target_estimates, style= 'xr')
    # #plt.show()


if __name__ == "__main__":
    main()



