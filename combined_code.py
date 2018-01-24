############################################
#### N-BODY GRAVITATIONAL SIMULATIONS   ####
#### By: Jeremy Bullock,                ####
####     Robert Burnet,                 ####
####     Cliff Vong                     ####
####                                    ####
#### TABLE OF CONTENTS:                 ####
####                                    ####
####     I) MODULE IMPORTS              ####
####                                    ####
####     1) ODE SOLVER METHODS          ####
####                                    ####
####     2) TEST CASES                  ####
####         2.1) PR0JECTILE MOTION     ####
####         2.2) MOON'S ORBIT          ####
####                                    ####
####     3) SIMULATIONS                 ####
####         3.1) GALAXY COLLISION      ####
####         3.2) LAGRANGE POINTS       ####
####         3.3) 3-BODY SCATTERING     ####
####                                    ####
############################################


############################################


############################################
### I) MODULE IMPORTS: #####################
############################################


import numpy as np
import scipy as sp
import scipy.constants as con
import scipy.integrate as spint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3


############################################


############################################
### 1) ODE SOLVER METHODS: #################
############################################


# The following functions are the ODE solvers that can be used. All are tested
# during test cases. A decision is made to use the RK4 method ODE solver for
# our simulations.

def euler(f, p0, t):

    '''
    Carries out Euler method of solving ODEs.

    Args:
        f  : function for the derivatives of the phase space coordinates
        p0 : initial phase space coordinates
        t  : array of time steps

    Returns:
        array which represents phase space coordinates calculated after each
        time step in t.
    '''
    
    # initialize positions and velocities at their input values
    # Now with both stored in a single array!
    p = np.array(p0)
    nd = len(p)
    
    t0 = t[0]
    nt = len(t)
    
    # Derivatives given by a function now.
    dpdt = f(p,t0)

    # We now save all outputs in a 2D array
    # one row for each time

    psave = np.empty((nt,nd))
    
    # Fill first row with initial conditions
    psave[0,:] = p

    # Now loop over remaining times
    for i in range(1,nt):

        dt = t[i]-t0
        t0 = t[i]
        
        # Update phase-space coordinates
        p += dpdt*dt
 
        # Update derivatives
        dpdt = f(p,t0)

        # Save row for this timestep
        psave[i,:] = p
    
    return (psave)


def euler_cromer(f, p0, t):
    
    '''
    Function to solve ODE using Euler-Cromer method to calculate trajectories
    of particles and galaxies.

    Args:
        f  : function which calculates and returns time derivates of each phase
            space coordinate
        p0 : initial phase space coordinates
        t  : time array

    Returns:
        array which represents phase space coordinates calculated after each
        time step in t.
    '''

    # Intitialize output array
    p = np.zeros((len(t), len(p0)))
    p[0] = p0
    mid = int(len(p0)/2)

    # Calculate phase space coordinates after each time step and save to
    # output array
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]
        p[i+1][mid:] = p[i][mid:] + f(p[i], t[i])[mid:]*dt
        p[i+1][:mid] = p[i][:mid] + p[i+1][mid:]*dt

    # Output phase space array
    return p
                      

def midpoint(f, p0, t):
    
    '''
    Function to solve ODE using Euler-Cromer method to calculate trajectories
    of particles and galaxies.

    Args:
        f  : function which calculates and returns time derivates of each phase
            space coordinate
        p0 : initial phase space coordinates
        t  : time array

    Returns:
        array which represents phase space coordinates calculated after each
        time step in t.
    '''

    # Intitialize output array
    p = np.zeros((len(t), len(p0)))
    p[0] = p0

    # Calculate phase space coordinates after each time step and save to
    # output array
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]
        p[i+1] = p[i] + f(p[i] + 0.5*dt*f(p[i], t[i]), t[i] + 0.5*dt)*dt

    # Output phase space array
    return p


def verlet(f, p0, t):
    
    '''
    Function to solve ODE using Verlet method to calculate trajectories of
    particles and galaxies.

    Args:
        f  : function which calculates and returns time derivates of each phase
            space coordinate
        p0 : initial phase space coordinates
        t  : time array

    Returns:
        array which represents phase space coordinates calculated after each
        time step in t.
    '''

    # Intitialize output array
    p = np.zeros((len(t), len(p0)))
    p[0] = p0
    mid = int(len(p0)/2)
    dt0 = t[1]-t[0]
    p[1][:mid] = p[0][:mid] + p[0][mid:]*dt0 + 0.5*f(p[0],t[0])[mid:]*dt0**2.0
    
    # Calculate phase space coordinates after each time step and save to
    # output array
    for i in range(1,len(t)-1):
        dt = t[i+1]-t[i]
        p[i+1][:mid] = 2*p[i][:mid] - p[i-1][:mid] + f(p[i],t[i])[mid:]*dt**2.0
        
    # Output phase space array
    return p


def rk4(f, p0, t):
    
    '''
    Carries out RK4 method of solving ODEs.

    Args:
        f  : function for the derivatives of the phase space coordinates
        p0 : initial phase space coordinates
        t  : array of time steps

    Returns:
        array which represents phase space coordinates calculated after each
        time step in t.
    '''
    
    # initialize output array and input initial conditions
    p = np.zeros((len(t),len(p0)))
    p[0] = p0
    
    # fill in output array
    for i in range(len(t)-1):
        
        # define step
        dt = t[i+1] - t[i]
        
        # calculate K values
        K1 = dt*f(p[i],t[i])
        K2 = dt*f(p[i]+0.5*K1,t[i]+0.5*dt)
        K3 = dt*f(p[i]+0.5*K2,t[i]+0.5*dt)
        K4 = dt*f(p[i]+K3,t[i]+dt)
        
        # compute next point using RK4 method
        p[i+1] = p[i] + (1/6.)*(K1 + 2*K2 + 2*K3 + K4)

    return p


############################################


############################################
### 2) TEST CASES:              ############
###     2.1) PROJECTILE MOTION  ############
###     2.2) MOON'S ORBIT       ############
############################################


############################################
## 2.1) PROJECTILE MOTION: #################
############################################

## Projectile motion test parameters:

# Mass of Earth
M_earth = 5.972e24

# Radius of Earth
R_earth = 6.371e6

# Initial phase space coordinates of projectile
p0_projmot = np.array([0.0,0.0,2.0,2.0])

# Initial phase space coordinates of projectile with escape velocity
p0_projmot_escape = np.array([0.0,0.0,2.0,np.sqrt(2*con.G*M_earth/R_earth)])

# Time array
t_projmot = np.arange(0,50,0.0001)


## ProjectileMotion class and example performance of plotting difference
## between calculated heights from each integration method and exact solution:

class ProjectileMotion(object):

    '''
    Class that will test various integration methods for solving projectile
    motion and plot the difference between each method's output in height and
    the exact solution.

    Args:
        M              : Mass of Earth
        R              : Radius of Earth
        coordinates    : Initial phase space coordinates of projectile
        t              : time array

    Methods:
        dpdt           : Function to calculate the derivatives of phase space
                         coordinates.
        exact_solution : Function that determines the exact solution to
                         projectile motion.
        plot           : Function to carry out calculation of trajectory
                         under various integration methods and plot the
                         difference between the calculated heights of each
                         method and the exact solution for height of
                         projectile.
    '''

    def __init__(self, M, R, coordinates, t):

        '''
        Initialize ProjectileMotion class

        Args:
            M : Mass of Earth
            R : Radius of Earth
            coordinates : Initial phase space coordinates of projectile
            t : time array
        '''

        self.M = M
        self.R = R
        self.p0 = coordinates
        self.t = t

    def dpdt(self, p, t):

        '''
        Function to calculate the derivatives of phase space coordinates.
        
        Args:
            p : phase space coordinates
            t : time array
        '''
        return np.array([p[2],p[3],0,-con.G*self.M/(self.R+p[1])**2.0])

    def exact_soln(self, p0, t):

        '''
        Function that determines the exact solution to projectile motion.

        Args:
            p0 : initial phase space coordinates
            t : time array
            
        Returns:
            array which represents phase space coordinates calculated after
            each time step in t.
        '''
        
        # Intitialize output array
        p = np.zeros((len(t), len(p0)))
        p[0] = p0

        # Calculate phase space coordinates after each time step and save to
        # output array
        for i in range(len(t)-1):
            dt = t[i+1]-t[i]
            p[i+1][2] = p[i][2]
            p[i+1][3] = p[i][3] + (-con.G*self.M/(self.R+p[i][1])**2.0)*dt
            p[i+1][0] = p[i][0] + p[i][2]*dt
            p[i+1][1] = p[i][1] + p[i][3]*dt + 0.5*(-con.G*self.M/ \
                                                    (self.R + \
                                                     p[i][1])**2.0)*dt**2.
            

        # Output phase space array
        return p

    def plot(self):

        '''
        Function to carry out calculation of trajectory under various
        integration methods and plot the difference between the calculated
        heights of each method and the exact solution for height of projectile.
        '''

        # Retrieve midpoint method and exact solution outputs
        midpoint_out = midpoint(self.dpdt, self.p0, self.t)
        euler_cromer_out = euler_cromer(self.dpdt, self.p0, self.t)
        euler_out = euler(self.dpdt, self.p0, self.t)
        verlet_out = verlet(self.dpdt, self.p0, self.t)
        rk4_out = rk4(self.dpdt, self.p0, self.t)
        odeint_out = spint.odeint(self.dpdt, self.p0, self.t)
        exact = self.exact_soln(self.p0, self.t)


        # Pull out x and difference in y coordinates of each solution to
        # exact solution
        x = []
        y_midpoint = []
        y_euler_cromer = []
        y_euler = []
        y_verlet = []
        y_rk4 = []
        y_odeint = []

        for i in range(len(exact)):
            if exact[i][1] >= 0.0:
                x.append(exact[i][0])
                y_midpoint.append(midpoint_out[i][1] - exact[i][1])
                y_euler_cromer.append(euler_cromer_out[i][1] - exact[i][1])
                y_euler.append(euler_out[i][1] - exact[i][1])
                y_verlet.append(verlet_out[i][1] - exact[i][1])
                y_rk4.append(rk4_out[i][1] - exact[i][1])
                y_odeint.append(odeint_out[i][1] - exact[i][1])
        
        # Plot projectile motion all methods
        plt.plot(x,y_euler, label='Euler method')
        plt.plot(x,y_euler_cromer, label='Euler-Cromer method')
        plt.plot(x,y_midpoint, label='Midpoint method')
        plt.plot(x,y_verlet, label='Verlet method')
        plt.plot(x,y_rk4, label='rk4 method')
        plt.plot(x,y_odeint, label='scipy ODEINT method')
        plt.legend()
        plt.xlabel('x (m)')
        plt.ylabel('Difference in height from exact solution (m)')
        plt.tight_layout()
        plt.title('Projectile motion from Earth\'s surface, V$_{0,x}$ = V$_{0,y}$ = 2.0m/s')
        plt.savefig('projectile_motion_difference_from_exact_all_methods.png')
        plt.close()

        # Plot projectile motion good methods
        plt.plot(x,y_midpoint, label='Midpoint method')
        plt.plot(x,y_verlet, label='Verlet method')
        plt.plot(x,y_rk4, label='rk4 method')
        plt.plot(x,y_odeint, label='scipy ODEINT method')
        plt.legend()
        plt.xlabel('x (m)')
        plt.ylabel('Difference in height from exact solution (m)')
        plt.tight_layout()
        plt.title('Projectile motion from Earth\'s surface, V$_{0,x}$ = V$_{0,y}$ = 2.0m/s')
        plt.savefig('projectile_motion_difference_from_exact_good_methods.png')
        plt.close()

# Generate projectile motion plots comparing integration methods
testcase = ProjectileMotion(M_earth,R_earth,p0_projmot,t_projmot)
testcase.plot()


############################################
## 2.2) MOON'S ORBIT: ######################
############################################


## Moon's orbit parameters:

# Mass of Earth
M_earth = 5.972e24

# Initial Moon phase space coordinates
p0_moon_animate = [384400000,0.,0.,0.,1018.2723,0.]

# Time array (from 0 to Moon's orbital period in seconds)
t_moon_animate = np.arange(0,27.323*24*60*60,10000)

# Time array for energy conservation plots
t_moon_energy = np.arange(0,1e8,1000)


## MoonAnimate class and example performance of Moon's orbit animation as well
## as an example of performing energy plots for comparing integration methods:

class MoonAnimate(object):

    '''
    Class that will use Midpoint method to animate one orbit of the Moon as well
    as generate plots of energy for various integration methods of the Moon's
    orbit to compare each method.

    Args:
        M            : Mass of Earth
        coordinates  : Initial phase space coordinates of moon
        t            : time array

    Methods:
        dpdt         : Function that calculates derivatives in phase space.
        update_point : Function to update points at each interval during
                       animation.
        animate      : Function that will animate Moon's orbit and save to mp4
                       file.
        plot         : Function to carry out calculation of trajectory under
                      various integration methods and plot the resultant
                      energies
    '''

    def __init__(self, M, coordinates, t):

        '''
        Initialize MoonAnimate class.

        Args:
            M           : Mass of Earth
            coordinates : Initial phase space coordinates of moon
            t           : time array
        '''

        self.M = M
        self.p0 = coordinates
        self.t = t

    def dpdt(self, p, t):
        
        '''
        Function that calculates derivatives in phase space.
            
        Args:
            p : phase space coordinates
            t : time array
            
        Returns:
            array of shape p which represents the derivatives of p element by
            element.
        '''
        
        r = (p[0]**2. + p[1]**2. + p[2]**2.)**0.5

        return np.array([p[3], p[4], p[5], -con.G*self.M*p[0]/(r**3.), \
                         -con.G*self.M*p[1]/(r**3.), -con.G*self.M*p[2]/(r**3.)])

    def update_point(self, num, data, line):
        
        '''
        Function to update points at each interval during animation.

        Args:
            num         : number of timesteps to run animation over
            dataLines1  : array of coordinates at each time step to animate
                          for galaxy 1
            dataLines2  : array of coordinates at each time step to animate
                          for galaxy 2
            lines1      : actual points being plotted for galaxy 1
            lines2      : actual points being plotted for galaxy 2

        Returns:
            plots to be animated.
        '''
        line.set_data(data[0:2, num])
        line.set_3d_properties(data[2, num])
        return line

    def animate(self):
        
        '''
        Function that will animate Moon's orbit and save to mp4 file.

        Returns:
            mp4 file of one orbit of the moon performed.
        '''
        
        #energy conservation with RK4 method:
        output = rk4(self.dpdt, self.p0, self.t)

        # Animate moon
        animated_output = np.array([output[:,0],output[:,1],output[:,2]])

        data = animated_output/1e8

        fig1 = plt.figure()
        ax = p3.Axes3D(fig1)

        l, = ax.plot(data[0, 0:1],data[1, 0:1],data[2, 0:1], 'o')

        ax.set_xlim3d([-5., 5.])
        ax.set_xlabel('X')

        ax.set_ylim3d([-5., 5.])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-5., 5.])
        ax.set_zlabel('Z')

        line_ani = animation.FuncAnimation(fig1, self.update_point, \
                                           len(data[0]), fargs=(data, l), \
                                           interval=1, blit=False)
        line_ani.save('moon.mp4', fps=60, bitrate=7200)
        plt.close()

    def plot(self):

        '''
        Function to carry out calculation of trajectories under various
        integration methods and plot the resultant energies.

        Returns:
            plots as png's of the resultant energies from each integration
            method tested.
        '''
        

        # Carry out various methods and save their outputs
        midpoint_out = midpoint(self.dpdt, self.p0, self.t)
        euler_cromer_out = euler_cromer(self.dpdt, self.p0, self.t)
        euler_out = euler(self.dpdt, self.p0, self.t)
        verlet_out = verlet(self.dpdt, self.p0, self.t)
        rk4_out = rk4(self.dpdt, self.p0, self.t)
        odeint_out = spint.odeint(self.dpdt, self.p0, self.t)

        # Initialize kinetic and potential energy arrays
        kin_mid = np.zeros(len(midpoint_out))
        pot_mid = np.zeros(len(midpoint_out))
        kin_euler_cromer = np.zeros(len(midpoint_out))
        pot_euler_cromer = np.zeros(len(midpoint_out))
        kin_euler = np.zeros(len(midpoint_out))
        pot_euler = np.zeros(len(midpoint_out))
        kin_verlet = np.zeros(len(midpoint_out))
        pot_verlet = np.zeros(len(midpoint_out))
        kin_rk4 = np.zeros(len(midpoint_out))
        pot_rk4 = np.zeros(len(midpoint_out))
        kin_odeint_out = np.zeros(len(midpoint_out))
        pot_odeint_out = np.zeros(len(midpoint_out))

        # Calculate kinetic and potential energies and save them to
        # corresponding kinetic and potential energy arrays
        for i in range(len(midpoint_out)):
                kin_mid[i] += 0.5*(midpoint_out[i][3]**2.0 + \
                                   midpoint_out[i][4]**2.0 + \
                                   midpoint_out[i][5]**2.0)
                pot_mid[i] += -con.G*self.M/(midpoint_out[i][0]**2.0 + \
                                             midpoint_out[i][1]**2.0 + \
                                             midpoint_out[i][2]**2.0)**0.5
                kin_euler_cromer[i] += 0.5*(euler_cromer_out[i][3]**2.0 + \
                                            euler_cromer_out[i][4]**2.0 + \
                                            euler_cromer_out[i][5]**2.0)
                pot_euler_cromer[i] += -con.G*self.M/ \
                                       (euler_cromer_out[i][0]**2.0 + \
                                        euler_cromer_out[i][1]**2.0 + \
                                        euler_cromer_out[i][2]**2.0)**0.5
                kin_euler[i] += 0.5*(euler_out[i][3]**2.0 + \
                                     euler_out[i][4]**2.0 + \
                                     euler_out[i][5]**2.0)
                pot_euler[i] += -con.G*self.M/(euler_out[i][0]**2.0 + \
                                               euler_out[i][1]**2.0 + \
                                               euler_out[i][2]**2.0)**0.5
                kin_verlet[i] += 0.5*(verlet_out[i][3]**2.0 + \
                                      verlet_out[i][4]**2.0 + \
                                      verlet_out[i][5]**2.0)
                pot_verlet[i] += -con.G*self.M/(verlet_out[i][0]**2.0 + \
                                                verlet_out[i][1]**2.0 + \
                                                verlet_out[i][2]**2.0)**0.5
                kin_rk4[i] += 0.5*(rk4_out[i][3]**2.0 + \
                                   rk4_out[i][4]**2.0 + \
                                   rk4_out[i][5]**2.0)
                pot_rk4[i] += -con.G*self.M/(rk4_out[i][0]**2.0 + \
                                             rk4_out[i][1]**2.0 + \
                                             rk4_out[i][2]**2.0)**0.5
                kin_odeint_out[i] += 0.5*(odeint_out[i][3]**2.0 + \
                                          odeint_out[i][4]**2.0 + \
                                          odeint_out[i][5]**2.0)
                pot_odeint_out[i] += -con.G*self.M/(odeint_out[i][0]**2.0 + \
                                                    odeint_out[i][1]**2.0 + \
                                                    odeint_out[i][2]**2.0)**0.5

        # Calculate corresponding total energy arrays for each method
        Energy_init = kin_mid[0] + pot_mid[0]
        Energy_mid = kin_mid + pot_mid - Energy_init
        Energy_euler_cromer = kin_euler_cromer + pot_euler_cromer - Energy_init
        Energy_euler = kin_euler + pot_euler - Energy_init
        Energy_verlet = kin_verlet + pot_verlet - Energy_init
        Energy_rk4 = kin_rk4 + pot_rk4 - Energy_init
        Energy_odeint = kin_odeint_out + pot_odeint_out - Energy_init

        # rescale time array in units of days
        t_days = self.t/(60*60*24)

        # Plot energies of all methods
        plt.plot(t_days,Energy_euler, label='Euler method')
        plt.plot(t_days,Energy_euler_cromer, label='Euler-Cromer method')
        plt.plot(t_days,Energy_mid, label='Midpoint method')
        plt.plot(t_days,Energy_verlet, label='Verlet method')
        plt.plot(t_days,Energy_rk4, label='rk4 method')
        plt.plot(t_days,Energy_odeint, label='scipy ODEINT method')
        plt.title('Deviation from Energy of Moon\'s orbit')
        plt.xlabel('Time (days)')
        plt.ylabel('Energy (J/kg)')
        plt.legend()
        plt.savefig('Energy_conservation_all_methods.png')
        plt.close()

        # Plot energies of good methods
        plt.plot(t_days,Energy_euler_cromer, label='Euler-Cromer method')
        plt.plot(t_days,Energy_mid, label='Midpoint method')
        plt.plot(t_days,Energy_rk4, label='rk4 method')
        plt.plot(t_days,Energy_odeint, label='scipy ODEINT method')
        plt.title('Deviation from Energy of Moon\'s orbit')
        plt.xlabel('Time (days)')
        plt.ylabel('Energy (J/kg)')
        plt.legend()
        plt.savefig('Energy_conservation_good_methods.png')
        plt.close()

        # Plot energies of good methods, without ODEINT
        plt.plot(t_days,Energy_euler_cromer, label='Euler-Cromer method')
        plt.plot(t_days,Energy_mid, label='Midpoint method')
        plt.plot(t_days,Energy_rk4, label='rk4 method')
        plt.title('Deviation from Energy of Moon\'s orbit')
        plt.xlabel('Time (days)')
        plt.ylabel('Energy (J/kg)')
        plt.legend()
        plt.savefig('Energy_conservation_good_methods_no_ODEINT.png')
        plt.close()


# Carry out Moon's orbit animation
moon_animate = MoonAnimate(M_earth, p0_moon_animate, t_moon_animate)
moon_animate.animate()

# Generate Moon's orbit energy plots comparing integration methods
moon_energy = MoonAnimate(M_earth, p0_moon_animate, t_moon_energy)
moon_energy.plot()


############################################


############################################
#### 3) SIMULATIONS:            ############
####    3.1) GALAXY COLLISION   ############
####    3.2) LAGRANGE POINTS    ############
####    3.3) 3-BODY SCATTERING  ############
############################################


############################################
### 3.1) GALAXY COLLISION: #################
############################################


## Galaxy collision simulation parameters:

# Gravitational constant (kpc * solar_mass^(-1) * (kpc/Myr)^2)
G_galcol = 4.302e-6 * (60*60*24*365.25*1e6)**2.0  / 3.086e16**2.0

# Galaxy 1 parameters:
# Mass of galaxy 1 (solar masses)
M_gal1 = 1e11
# Radius of galaxy 1 (kpc)
R_gal1 = 30
# Initial phase space coordinates of galaxy 1
gal1_traj = [0.0,0.0,0.0,0.0,0.0,0.0]

# Galaxy 2 parameters:
# Mass of galaxy 2 (solar masses)
M_gal2 = 1e10
# Radius of galaxy 2 (kpc)
R_gal2 = 15
# Initial phase space coordinates of galaxy 2
gal2_traj = [70.71,70.71,0.0,-(2*G_galcol*M_gal1/100)**0.5*np.cos(np.pi/6),\
             -(2*G_galcol*M_gal1/100)**0.5*np.sin(np.pi/6),0.0]

# Time range (myr)
t_galcol = np.arange(0,5000,3)

# Number of stars to simulate in each galaxy. For best looking results, set
# N_stars to 10000, this is the number of stars I used to generate movies,
# although note that setting it as such will use up to 5 GB of memory and may
# take over 8 hours to run with an i5 4690k, 8GB of RAM, and GTX 970 (if t =
# np.arange(0,5000,3); if t is bigger, it will take longer).
# Currently set to 500 for memory and computational time saving purposes.
N_stars_galcol = 500


## GalaxyCollision class and example performance of galaxy collision animation:

class GalaxyCollision(object):
    
    '''
    Class that intitializes the positions and velocities of the two galaxies
    (and their stars) to be collided, as well as animate a collision when
    corresponding method is called.

    Args:
        N_stars         : number of stars to simulate around each galaxy
        mass1           : mass of galaxy 1
        coordinates1    : initial coordinates and velocities of center of
                          galaxy 1
        radius1         : radius of galaxy 1
        mass2           : mass of galaxy 2
        coordinates2    : initial coordinates and velocities of center of
                          galaxy 2
        radius2         : radius of galaxy 2
        t               : time array for simulation
        direction       : direction of orbits of stars around each galaxy,
                          either 'clockwise' or 'counterclockwise'

    Methods:
        initvelocities  : Calculate the initial velocity of an orbit at
                          positions (r,theta) that will give circular orbits
                          for specified galaxy mass and position.
        initorbits      : Calculate initial orbits of N_stars around each
                          galaxy with velocities and energies such that it is
                          orbitting in a circular orbit around each galactic
                          center.
        dpdt            : Function that calculates derivatives in phase space.
        update_pont     : Function to update points at each interval during
                          animation.
        animate         : Function to animate collision of galaxies and save
                          to mp4file.
    '''

    def __init__(self, G, N_stars, mass1, coordinates1, radius1, mass2, \
                 coordinates2, radius2, t, direction='counterclockwise'):

        '''
        Intialize GalaxyCollision class.

        Args:
            G               : gravitational constant in specific units
            N_stars         : number of stars to simulate around each galaxy
            mass1           : mass of galaxy 1
            coordinates1    : initial coordinates and velocities of center of
                              galaxy 1
            radius1         : radius of galaxy 1
            mass2           : mass of galaxy 2
            coordinates2    : initial coordinates and velocities of center of
                              galaxy 2
            radius2         : radius of galaxy 2
            t               : time array for simulation
            direction       : direction of orbits of stars around each galaxy,
                              either 'clockwise' or 'counterclockwise'
        '''

        self.G = G
        self.N_stars = N_stars
        self.mass1 = mass1
        self.p1 = coordinates1
        self.radius1 = radius1
        self.mass2 = mass2
        self.p2 = coordinates2
        self.radius2 = radius2
        self.t = t
        if direction == 'clockwise' or direction == 'counterclockwise':
            self.direction = direction
        else:
            print('direction must be "clockwise" or "counterclockwise" ',\
                  'defaulting to "counterclockwise"')
            self.direction = 'counterclockwise'
        
        # Initialize orbits of stars around each galaxy
        self.orbits = self.initorbits()
        self.neworbits = []
        
    def initvelocities(self, r, theta, mass, p):

        '''
        Calculate the initial velocity of an orbit at positions (r,theta) that
        will give circular orbits for specified galaxy mass and position.

        Args:
            r       : radial separations of particles and galaxy of interest
            theta   : angular positions of particles
            mass    : mass of galaxy of interest
            p       : initial phase space coordinates of galaxy

        Returns:
            two arrays which represent x and y velocities of each particle with
            positions (r, theta) such that the x and y velocities result in
            circular orbits about the corresponding galaxy they're orbiting.
        '''

        # Total circular velocities (kpc/Myr)
        vel = np.sqrt(self.G * mass / r)

        # x, y components of velocities plus the velocity of the galaxy
        if self.direction == 'clockwise':
            velx = -np.sin(theta)*vel
            vely = np.cos(theta)*vel
        else:
            velx = -np.sin(theta)*vel + p[3]
            vely = np.cos(theta)*vel + p[4]

        # Output x, y velocities of particles
        return (velx, vely)

    def initorbits(self):

        '''
        Calculate initial orbits of N_stars around each galaxy with velocities
        and energies such that it is orbitting in a circular orbit around each
        galactic center.

        Returns:
            array which represents the initial phase space coordinates of
            particles in both galaxies as well as the galaxies themselves.
        '''

        # Calculate random radial separations out to the radius of the galaxy
        r_orbits1 = np.random.random(self.N_stars)*self.radius1
        r_orbits2 = np.random.random(self.N_stars)*self.radius2

        # Calculate random azimuthal angle from 0 to 2pi
        theta1 = np.random.random(self.N_stars)*2*np.pi
        theta2 = np.random.random(self.N_stars)*2*np.pi

        # Get random x, y positions of stars from radii and angles and
        # position of the galaxy
        x1 = np.cos(theta1)*r_orbits1 + self.p1[0]
        y1 = np.sin(theta1)*r_orbits1 + self.p1[1]
        x2 = np.cos(theta2)*r_orbits2 + self.p2[0]
        y2 = np.sin(theta2)*r_orbits2 + self.p2[1]

        # Retrieve velocities
        velx1, vely1 = self.initvelocities(r_orbits1, theta1, self.mass1, \
                                           self.p1)
        velx2, vely2 = self.initvelocities(r_orbits2, theta2, self.mass2, \
                                           self.p2)
        if self.direction == 'clockwise':
            velx1 = -velx1 + self.p1[3]
            vely1 = -vely1 + self.p1[4]
            velx2 = -velx2 + self.p2[3]
            vely2 = -vely2 + self.p2[4]

        # Output phase space coordinates of particles and galaxies
        return np.array([x1, y1, self.p1[2]*np.ones(self.N_stars), \
                         x2, y2, self.p2[2]*np.ones(self.N_stars), \
                         self.p1[0], self.p1[1], self.p1[2], \
                         self.p2[0], self.p2[1], self.p2[2], \
                         velx1, vely1, self.p1[5]*np.ones(self.N_stars), \
                         velx2, vely2, self.p2[5]*np.ones(self.N_stars), \
                         self.p1[3], self.p1[4], self.p1[5], \
                         self.p2[3], self.p2[4], self.p2[5]])

    def dpdt(self, p, t):
    
        '''
        Function that calculates derivatives in phase space.
        
        Args:
            p : phase space coordinates : (x,y,z) coordinates of two particles
                (one in galaxy 1 and another in galaxy 2) and the two galaxies
                along with x,y,z velocities of each
            t : time array
            
        Returns:
            array of shape p which represents the derivatives of p element by
            element.
        '''

        # Calculate radial separation between particle in galaxy 1 and galaxy 1
        # center
        r1 = ((p[0]-p[6])**2. + (p[1]-p[7])**2. + (p[2]-p[8])**2.)**0.5

        # Calculate radial separation between particle in galaxy 1 and galaxy 2
        # center
        r2 = ((p[0]-p[9])**2. + (p[1]-p[10])**2. + (p[2]-p[11])**2.)**0.5

        # Calculate radial separation between particle in galaxy 2 and galaxy 1
        # center
        r3 = ((p[3]-p[6])**2. + (p[4]-p[7])**2. + (p[5]-p[8])**2.)**0.5

        # Calculate radial separation between particle in galaxy 2 and galaxy 2
        # center
        r4 = ((p[3]-p[9])**2. + (p[4]-p[10])**2. + (p[5]-p[11])**2.)**0.5

        # Calculate radial separation between the two galaxies
        rgal = ((p[6]-p[9])**2.0 + (p[7] - p[10])**2.0 + (p[8]-p[11])**2.0)**0.5

        # Return derivatives of phase space coordinates
        return np.array( [p[12], p[13], p[14], p[15], p[16], p[17], \
                          p[18], p[19], p[20], p[21], p[22], p[23], \
                          -self.G*(self.mass1*(p[0]-p[6])/(r1**3.) + \
                                   self.mass2*(p[0]-p[9])/(r2**3.)), \
                          -self.G*(self.mass1*(p[1]-p[7])/(r1**3.) + \
                                   self.mass2*(p[1]-p[10])/(r2**3.)), \
                          -self.G*(self.mass1*(p[2]-p[8])/(r1**3.) + \
                                   self.mass2*(p[2]-p[11])/(r2**3.)), \
                          -self.G*(self.mass1*(p[3]-p[6])/(r3**3.) + \
                                   self.mass2*(p[3]-p[9])/(r4**3.)), \
                          -self.G*(self.mass1*(p[4]-p[7])/(r3**3.) + \
                                   self.mass2*(p[4]-p[10])/(r4**3.)), \
                          -self.G*(self.mass1*(p[5]-p[8])/(r3**3.) + \
                                   self.mass2*(p[5]-p[11])/(r4**3.)), \
                          -self.G*self.mass2*(p[6]-p[9])/rgal**3.0, \
                          -self.G*self.mass2*(p[7]-p[10])/rgal**3.0, \
                          -self.G*self.mass2*(p[8]-p[11])/rgal**3.0, \
                          -self.G*self.mass1*(p[9]-p[6])/rgal**3.0, \
                          -self.G*self.mass1*(p[10]-p[7])/rgal**3.0, \
                          -self.G*self.mass1*(p[11]-p[8])/rgal**3.0] )

    def update_point(self, num, dataLines1, dataLines2, lines1, lines2):
            
        '''
        Function to update points at each interval during animation.

        Args:
            num         : number of timesteps to run animation over
            dataLines1  : array of coordinates at each time step to animate for
                          galaxy 1
            dataLines2  : array of coordinates at each time step to animate for
                          galaxy 2
            lines1      : actual points being plotted for galaxy 1
            lines2      : actual points being plotted for galaxy 2

        Returns:
            plots to be animated.
        '''
        
        for line, data in zip(lines1, dataLines1):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(0.2)
            
        for line, data in zip(lines2, dataLines2):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(0.2)
            
        return lines1, lines2

    def animate(self, filename):

        '''
        Function to animate collision of galaxies and save to mp4 file.

        Args:
            filename : string to be used as filename for .mp4 file
            
        Returns:
            mp4 file of galaxy collision animation performed.
        '''
        
        # Output array of particle trajectories
        output = np.zeros([self.N_stars,len(self.t),len(self.orbits)])

        # Calculate particle trajectories using ODE solver
        for i in range(self.N_stars):
            output[i] += rk4(self.dpdt, [self.orbits[0][i], \
                                         self.orbits[1][i], \
                                         self.orbits[2][i], \
                                         self.orbits[3][i], \
                                         self.orbits[4][i], \
                                         self.orbits[5][i], \
                                         self.orbits[6], self.orbits[7], \
                                         self.orbits[8], self.orbits[9], \
                                         self.orbits[10], self.orbits[11], \
                                         self.orbits[12][i], \
                                         self.orbits[13][i], \
                                         self.orbits[14][i], \
                                         self.orbits[15][i], \
                                         self.orbits[16][i], \
                                         self.orbits[17][i], \
                                         self.orbits[18], self.orbits[19], \
                                         self.orbits[20], self.orbits[21], \
                                         self.orbits[22], self.orbits[23]],\
                             self.t)

        # Reshape arrays for animation
        animated_output = []
        animated_output2 = []

        for i in range(len(output)):
            animated_output.append(np.array([output[i][:,0],output[i][:,1],\
                                             output[i][:,2]]))
            animated_output2.append(np.array([output[i][:,3],output[i][:,4],\
                                              output[i][:,5]]))

        animated_output = np.array(animated_output)
        animated_output2 = np.array(animated_output2)

        # Intitialize animation
        fig1 = plt.figure()
        ax = p3.Axes3D(fig1)

        l = [ax.plot(dat[0, 0:1],dat[1, 0:1],dat[2, 0:1], 'o')[0] \
             for dat in animated_output]
        l2 = [ax.plot(dat[0, 0:1],dat[1, 0:1],dat[2, 0:1], 'o')[0] \
              for dat in animated_output2]

        # Set x limits from -100 to 100 kpc
        ax.set_xlim3d([-100., 100.])
        ax.set_xlabel('X')

        # Set y limits from -100 to 100 kpc
        ax.set_ylim3d([-100., 100.])
        ax.set_ylabel('Y')

        # Set z limits from -100 to 100 kpc
        ax.set_zlim3d([-100., 100.])
        ax.set_zlabel('Z')

        # Run animation and save
        line_ani = animation.FuncAnimation(fig1, self.update_point, \
                                           len(animated_output[0][0]), \
                                           fargs=(animated_output, \
                                                  animated_output2, l, l2),\
                                           interval=1, blit=False)
        line_ani.save(filename+'.mp4', fps = 60, bitrate=7200)
        plt.close()


# Initialize GalaxyCollision object with galaxy 1 and galaxy 2 parameters
galcol = GalaxyCollision(G_galcol, N_stars_galcol, M_gal1, gal1_traj, R_gal1, \
                         M_gal2, gal2_traj, R_gal2, t_galcol, \
                         direction = 'counterclockwise')
# Animate collision and save to file
galcol.animate('rk4_parabolic_orbit_10000_particles_counterclockwise')


############################################
### 3.2) LAGRANGE POINTS: ##################
############################################


# Gravitational constant (AU^3 * (solar mass)^(-1) * (days)^(-2))
G_lp = 2.959e-4
# Distance between M1 and M2
# Distance in AU
R = 1.

# Parameters of M1 (Sun)
# Mass in solar masses
M1_lp = 1.

# Parameters of M2 (Earth)
# Mass in solar mass
M2_lp = 3.003467e-6

# Initial positions and velocities of M1 and M2 in phase space
M1_traj = [-M2/(M1+M2), 0.0, 0.0, 0.0]
M2_traj = [M1/(M1+M2), 0.0, 0.0, 0.0]

# Initial positions and velocities at Lagrange points
L1_traj = [1.-1.01*((M2/(M1+M2))/3.)**(1./3), 0.0, 0.0, 0.0]
L2_traj = [1.+((M2/(M1+M2))/3.)**(1./3), 0.0, 0.0, 0.0]
L3_traj = [-1.*(1+(5/12)*(M2/(M1+M2))), 0.0, 0.0, 0.0]
L4_traj = [0.5*(M1-M2)/(M1+M2), np.sqrt(3)/2, 0.0, 0.0]
L5_traj = [0.5*(M1-M2)/(M1+M2), -np.sqrt(3)/2, 0.0, 0.0]

# Time range (days)
t_lagpts = np.arange(0, 356*5, 0.5)

class LagrangePoints(object):
    '''
    Class that intitializes the positions and velocities of the Sun, Earth and Lagrange points

    Args:
        mass1           : mass of Sun
        coordinates1    : initial coordinates and velocity of Sun
        mass2           : mass of Earth
        coordinates2    : initial coordinates and velocity of Earth
        coordinates3    : initial coordinates and velocity of L1
        coordinates4    : initial coordinates and velocity of L2
        coordinates5    : initial coordinates and velocity of L3
        coordinates6    : initial coordinates and velocity of L4
        coordinates7    : initial coordinates and velocity of L5
        time            : time array for simulation
    '''

    def __init__(self, mass1, coordinates1, mass2, coordinates2, coordinates3, coordinates4, coordinates5, coordinates6, coordinates7, time):
        '''
        intialize LagrangePoints class

        Args:
            mass1           : mass of Sun
            coordinates1    : initial coordinates and velocities of Sun
            mass2           : mass of Earth
            coordinates2    : initial coordinates and velocities of Earth
            coordinates3    : initial coordinates and velocities of L1
            coordinates4    : initial coordinates and velocities of L2
            coordinates5    : initial coordinates and velocities of L3
            coordinates6    : initial coordinates and velocities of L4
            coordinates7    : initial coordinates and velocities of L5
            time            : time array for simulation
        '''

        self.mass1 = mass1
        self.p1    = coordinates1
        self.mass2 = mass2
        self.p2    = coordinates2
        self.p3    = coordinates3
        self.p4    = coordinates4
        self.p5    = coordinates5
        self.p6    = coordinates6
        self.p7    = coordinates7
        self.t     = time

        # Initialize orbits
        self.orbits = self.initorbits()

    def initvelocities(self, r, theta, mass, p):
        '''
        Calculate the initial velocity of an orbit at positions (r,theta) that will give
        circular orbits

        Args:
            r       : radial separations of objects
            theta   : angular positions of objects
            mass    : mass of body of interest
            p       : initial phase space coordinates of object

        Returns:
            two arrays which represent x and y velocities of each particle with positions
            (r, theta) such that the x and y velocities result in circular orbits about
            the corresponding body they're orbiting.
        '''

        # Total circular velocities (kpc/Myr)
        vel = np.sqrt(G_lp * mass / r)

        # x, y components of velocities
        velx = -np.sin(theta)*vel
        vely = np.cos(theta)*vel

        # Output x, y velocities of object
        return (velx, vely)

    def initorbits(self):
        '''
        Calculate initial orbits of the bodies about the centre of mass of the system

        Returns:
            array which represents the initial phase space coordinates of all the bodies
            in the system
        '''
        # Coordinates of objects in orbit
        x1    = self.p1[0]
        y1    = self.p1[1]
        x2    = self.p2[0]
        y2    = self.p2[1]
        x3    = self.p3[0]
        y3    = self.p3[1]
        x4    = self.p4[0]
        y4    = self.p4[1]
        x5    = self.p5[0]
        y5    = self.p5[1]
        x6    = self.p6[0]
        y6    = self.p6[1]
        x7    = self.p7[0]
        y7    = self.p7[1]
        r1    = (x1**2 + y1**2)**0.5
        r2    = (x2**2 + y2**2)**0.5
        r3    = (x3**2 + y3**2)**0.5
        r4    = (x4**2 + y4**2)**0.5
        r5    = (x5**2 + y5**2)**0.5
        r6    = (x6**2 + y6**2)**0.5
        r7    = (x7**2 + y7**2)**0.5

        # Calculates angle of objects with respect to centre of mass  
        theta1 = np.arctan2(y1, x1)
        theta2 = np.arctan2(y2, x2)
        theta3 = np.arctan2(y3, x3)
        theta4 = np.arctan2(y4, x4)
        theta5 = np.arctan2(y5, x5)
        theta6 = np.arctan2(y6, x6)
        theta7 = np.arctan2(y7, x7)

        # Calculate velocities throughout orbit
        velx1, vely1 = self.initvelocities(r1+r2, theta1, self.mass2, self.p1)
        velx2, vely2 = self.initvelocities(r2+r1, theta2, self.mass1, self.p2)
        velx3, vely3 = -2*np.pi*r3/(365)*np.sin(theta3)+self.p3[2], 2*np.pi*r3/(365)*np.cos(theta3)+self.p3[3]
        velx4, vely4 = -2*np.pi*r4/(365)*np.sin(theta4)+self.p4[2], 2*np.pi*r4/(365)*np.cos(theta4)+self.p4[3]
        velx5, vely5 = -2*np.pi*r5/(365)*np.sin(theta5)+self.p5[2], 2*np.pi*r5/(365)*np.cos(theta5)+self.p5[3]
        velx6, vely6 = -2*np.pi*r6/(365)*np.sin(theta6)+self.p6[2], 2*np.pi*r6/(365)*np.cos(theta6)+self.p6[3]
        velx7, vely7 = -2*np.pi*r7/(365)*np.sin(theta7)+self.p7[2], 2*np.pi*r7/(365)*np.cos(theta7)+self.p7[3]

        # Output phase space coordinates of orbiting bodies
        return np.array([x1, y1, 0.0, x2, y2, 0.0, x3, y3, 0.0, x4, y4, 0.0, x5, y5, 0.0, x6, y6, 0.0, x7, y7, 0.0, \
                        velx1, vely1, 0.0, velx2, vely2, 0.0, velx3, vely3, 0.0, velx4, vely4, 0.0, velx5, vely5, 0.0, velx6, vely6, 0.0, velx7, vely7, 0.0])

    def dpdt(self, p, t):
        '''
        Function that calculates derivatives in phase space
        
        Args:
            p : phase space coordinates : (x,y) coordinates of all bodies in the system
                along with x,y velocities of each
            t : time array
        Returns:
            array of shape p which represents the derivatives of p element by element
        '''
        # Radial separation between Sun and Earth
        r1  = ((p[0]-p[3])**2 + (p[1]-p[4])**2)**0.5
        
        # Radial separation between Earth and Sun
        r2  = ((p[3]-p[0])**2 + (p[4]-p[1])**2)**0.5

        # Radial separation between L1 and Sun
        r3S = ((p[6]-p[0])**2 + (p[7]-p[1])**2)**0.5

        # Radial separation between L1 and Earth
        r3E = ((p[6]-p[3])**2 + (p[7]-p[4])**2)**0.5

        # Radial separation between L2 and Sun
        r4S = ((p[9]-p[0])**2 + (p[10]-p[1])**2)**0.5

        # Radial separation between L2 and Earth
        r4E = ((p[9]-p[3])**2 + (p[10]-p[4])**2)**0.5

        # Radial separation between L3 and Sun
        r5S = ((p[12]-p[0])**2 + (p[13]-p[1])**2)**0.5

        # Radial separation between L3 and Earth
        r5E = ((p[12]-p[3])**2 + (p[13]-p[4])**2)**0.5

        # Radial separation between L4 and Sun
        r6S = ((p[15]-p[0])**2 + (p[16]-p[1])**2)**0.5

        # Radial separation between L4 and Earth
        r6E = ((p[15]-p[3])**2 + (p[16]-p[4])**2)**0.5

        # Radial separation between L5 and Sun
        r7S = ((p[18]-p[0])**2 + (p[19]-p[1])**2)**0.5

        # Radial separation between L5 and Earth
        r7E = ((p[18]-p[3])**2 + (p[19]-p[4])**2)**0.5

        return np.array([p[21], p[22], 0.0,  p[24], p[25], 0.0, p[27], p[28], 0.0, p[30], p[31], 0.0, p[33], p[34], 0.0, p[36], p[37], 0.0, p[39], p[40], 0.0, \
                        -G_lp*(self.mass2*(p[0]-p[3])/(r1**3)), -G_lp*(self.mass2*(p[1]-p[4])/(r1**3)), 0.0, \
                        -G_lp*(self.mass1*(p[3]-p[0])/(r2**3)), -G_lp*(self.mass1*(p[4]-p[1])/(r2**3)), 0.0, \
                        -G_lp*(self.mass1*(p[6]-p[0])/(r3S**3)) - G_lp*(self.mass2*(p[6]-p[3])/(r3E**3)), \
                        -G_lp*(self.mass1*(p[7]-p[1])/(r3S**3)) - G_lp*(self.mass2*(p[7]-p[4])/(r3E**3)), 0.0,\
                        -G_lp*(self.mass1*(p[9]-p[0])/(r4S**3)) - G_lp*(self.mass2*(p[9]-p[3])/(r4E**3)), \
                        -G_lp*(self.mass1*(p[10]-p[1])/(r4S**3)) - G_lp*(self.mass2*(p[10]-p[4])/(r4E**3)), 0.0, \
                        -G_lp*(self.mass1*(p[12]-p[0])/(r5S**3)) - G_lp*(self.mass2*(p[12]-p[3])/(r5E**3)), \
                        -G_lp*(self.mass1*(p[13]-p[1])/(r5S**3)) - G_lp*(self.mass2*(p[13]-p[4])/(r5E**3)), 0.0, \
                        -G_lp*(self.mass1*(p[15]-p[0])/(r6S**3)) - G_lp*(self.mass2*(p[15]-p[3])/(r6E**3)), \
                        -G_lp*(self.mass1*(p[16]-p[1])/(r6S**3)) - G_lp*(self.mass2*(p[16]-p[4])/(r6E**3)), 0.0, \
                        -G_lp*(self.mass1*(p[18]-p[0])/(r7S**3)) - G_lp*(self.mass2*(p[18]-p[3])/(r7E**3)), \
                        -G_lp*(self.mass1*(p[19]-p[1])/(r7S**3)) - G_lp*(self.mass2*(p[19]-p[4])/(r7E**3)), 0.0])

    def update_point(self, num, dataLines1, dataLines2, dataLines3, dataLines4, dataLines5, dataLines6, dataLines7, lines1, lines2, lines3, lines4, lines5, lines6, lines7):
        '''
        Function to update points at each interval during animation

        Args:
            num         : number of timesteps to run animation over
            dataLines1  : array of coordinates at each time step to animate for Sun
            dataLines2  : array of coordinates at each time step to animate for Earth
            dataLines3  : array of coordinates at each time step to animate for L1
            dataLines4  : array of coordinates at each time step to animate for L2
            dataLines5  : array of coordinates at each time step to animate for L3
            dataLines6  : array of coordinates at each time step to animate for L4
            dataLines7  : array of coordinates at each time step to animate for L5
            lines1      : actual points being plotted for Sun
            lines2      : actual points being plotted for Earth
            lines3      : actual points being plotted for L1
            lines4      : actual points being plotted for L2
            lines5      : actual points being plotted for L3
            lines6      : actual points being plotted for L4
            lines7      : actual points being plotted for L5

        Returns:
            plots to be animated
        '''
        
        for line, data in zip(lines1, dataLines1):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(25)
            line.set_markerfacecolor('y')
            line.set_markeredgecolor('y')
            
        for line, data in zip(lines2, dataLines2):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(10)
            line.set_markerfacecolor('b')
            line.set_markeredgecolor('b')

        for line, data in zip(lines3, dataLines3):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(5)
            line.set_markerfacecolor('r')
            line.set_markeredgecolor('r')

        for line, data in zip(lines4, dataLines4):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(5)
            line.set_markerfacecolor('g')
            line.set_markeredgecolor('g')

        for line, data in zip(lines5, dataLines5):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(5)
            line.set_markerfacecolor('c')
            line.set_markeredgecolor('c')

        for line, data in zip(lines6, dataLines6):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(5)
            line.set_markerfacecolor('m')
            line.set_markeredgecolor('m')

        for line, data in zip(lines7, dataLines7):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(5)
            line.set_markerfacecolor('orange')
            line.set_markeredgecolor('orange')

        return lines1, lines2, lines3, lines4, lines5, lines6, lines7

    def animate(self, filename):
        '''
        Function to animate orbit of Earth-Sun system and save to mp4 file.

        Args:
            filename : string to be used as filename for .mp4 file.
        '''
        
        # Output array of particle trajectories
        # Calculate particle trajectories using ODE solver

        output = rk4(self.dpdt, [self.orbits[0], self.orbits[1], self.orbits[2], \
                                        self.orbits[3], self.orbits[4], self.orbits[5], \
                                        self.orbits[6], self.orbits[7], self.orbits[8], \
                                        self.orbits[9], self.orbits[10], self.orbits[11], \
                                        self.orbits[12], self.orbits[13], self.orbits[14], \
                                        self.orbits[15], self.orbits[16], self.orbits[17], \
                                        self.orbits[18], self.orbits[19], self.orbits[20], \
                                        self.orbits[21], self.orbits[22], self.orbits[23], \
                                        self.orbits[24], self.orbits[25], self.orbits[26], \
                                        self.orbits[27], self.orbits[28], self.orbits[29], \
                                        self.orbits[30], self.orbits[31], self.orbits[32], \
                                        self.orbits[33], self.orbits[34], self.orbits[35], \
                                        self.orbits[36], self.orbits[37], self.orbits[38], \
                                        self.orbits[39], self.orbits[40], self.orbits[41]], self.t)

        # Reshape arrays for animation
        animated_output  = []
        animated_output2 = []
        animated_output3 = []
        animated_output4 = []
        animated_output5 = []
        animated_output6 = []
        animated_output7 = []

        animated_output.append(np.array([output[:,0],output[:,1],output[:,2]]))
        animated_output2.append(np.array([output[:,3],output[:,4],output[:,5]]))
        animated_output3.append(np.array([output[:,6],output[:,7],output[:,8]]))
        animated_output4.append(np.array([output[:,9],output[:,10],output[:,11]]))
        animated_output5.append(np.array([output[:,12],output[:,13],output[:,14]]))
        animated_output6.append(np.array([output[:,15],output[:,16],output[:,17]]))
        animated_output7.append(np.array([output[:,18],output[:,19],output[:,20]]))

        animated_output  = np.array(animated_output)
        animated_output2 = np.array(animated_output2)
        animated_output3 = np.array(animated_output3)
        animated_output4 = np.array(animated_output4)
        animated_output5 = np.array(animated_output5)
        animated_output6 = np.array(animated_output6)
        animated_output7 = np.array(animated_output7)

        data  = animated_output
        data2 = animated_output2
        data3 = animated_output3
        data4 = animated_output4
        data5 = animated_output5
        data6 = animated_output6
        data7 = animated_output7

        # Intitialize animation
        fig1 = plt.figure()
        ax   = p3.Axes3D(fig1)

        l  = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'o')[0] for dat in data]
        l2 = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'o')[0] for dat in data2]
        l3 = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'o')[0] for dat in data3]
        l4 = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'o')[0] for dat in data4]
        l5 = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'o')[0] for dat in data5]
        l6 = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'o')[0] for dat in data6]
        l7 = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], 'o')[0] for dat in data7]

        # Set x limits from -2 to 2 AU
        ax.set_xlim3d([-2., 2.])
        ax.set_xlabel('X')

        # Set y limits from -2 to 2 AU
        ax.set_ylim3d([-2., 2.])
        ax.set_ylabel('Y')

        # Set z limits from -2 to 2 AU
        ax.set_zlim3d([-2., 2.])
        ax.set_zlabel('Z')

        # Run animation and save
        line_ani = animation.FuncAnimation(fig1, self.update_point, len(data[0][0]), \
                                           fargs=(data, data2, data3, data4, data5, data6, data7, l, l2, l3, l4, l5, l6, l7), interval=1, blit=False)
        line_ani.save(filename+'.mp4', fps = 60, bitrate=7200)        

# Initialize LagrangePoints object with Sun and Earth parameters
lagpts = LagrangePoints(M1_lp, M1_traj, M2_lp, M2_traj, L1_traj, L2_traj, L3_traj, L4_traj, L5_traj, t_lagpts)
# Animate orbits and save to file
lagpts.animate('lagrange_points_no_perturpation_5_years')


############################################
### 3.3) 3-BODY SCATTERING: ################
############################################

#units of km^3/solarmass/year^2
G = (6.67408e-11)*(1.99e30)*(3.154e7*3.154e7)/((1000)**3)

#Mass 1 and 2 of binary star system (in solar masses)
M1 = 4.
M2 = 4.
#Mass 3 (The incoming star)
M3 = 8

#Radius of the stars (km) (the Sun)
R1 = 695700.
R2 = 695700.
R3 = 695700.

#Initialize initial phase space coordinates for the 3 stars (Elements [2],[3],[4] corresponds to (x,y,z))
star1 = [M1,R1,5.*R1,1.,1.]
star2 = [M2,R2,-5.*R2,1.,1.]
star3 = [M3,R3,30.*R3,0.*R3,45*R3]

#Time array, units of years
t = np.arange(0,0.05,0.00001)

class ThreeBody(object):
    '''
    Class that intitializes the positions and velocities of the binary system and the third star to be simulated, as well as animate a collision when
    corresponding method is called.

    Args:
        star1           : mass of star 1, radius of star 1, 3 initial phase space coordinates
        star2           : mass of star 2, radius of star 2, 3 initial phase space coordinates
        star3           : mass of star 3, radius of star 3, 3 initial phase space coordinates
        G               : gravitational constant in the units of km^3/solarmass/year^2
        t               : time array for simulation
        
    Methods:
        initvelocities  : Calculate the initial velocity of an orbit at
                          positions (r,theta) that will give circular orbits
                          for the two binary stars
        initorbits      : Calculates the initial conditions of the 3 stars
        dpdt            : Function that calculates derivatives in phase space.
        update_pont     : Function to update points at each interval during
                          animation.
        animate         : Function to animate the 3 body scattering simulation and save
                          to mp4file.
    '''

    def __init__(self,star1,star2,star3,G,t):
        '''
        Intialize ThreeBody class.

        Args:
            G               : gravitational constant in specific units
            mass1           : mass of star 1
            mass2           : mass of star 2
            mass3           : mass of star 3
            r1              : radius of star 1
            r2              : radius of star 2
            r3              : radius of star 3
            p1              : Initial phase space coordinates of star 1
            p2              : Initial phase space coordinates of star 2
            p3              : Initial phase space coordinates of star 3
            t               : time array for simulation
            p_space         : 18 element array containing all initial conditions (x,y,z) and (v_x,v_y,v_z) of the 3 stars
        '''
        self.G = G
        self.mass1 = star1[0]
        self.mass2 = star2[0]
        self.mass3 = star3[0]
        self.r1 = star1[1]
        self.r2 = star2[1]
        self.r3 = star3[1]
        self.p1 = [star1[2],star1[3],star1[4]]
        self.p2 = [star2[2],star2[3],star2[4]]
        self.p3 = [star3[2],star3[3],star3[4]]
        self.t = t
        self.p_space = self.initorbits()

    def initvelocities(self, r, theta, mass, p, d):
        '''
        Calculate the initial velocity of a binary system at positions (r,theta) that will give
        stable orbits

        Args:
            r       : radial separations of the binary stars
            theta   : angular positions of objects
            mass    : mass of body of interest
            p       : initial phase space coordinates of object
            d       : distance from body of interest to the origin

        Returns:
            two arrays which represent x and y velocities of each star with positions
            (r, theta) such that the x and y velocities result in stable orbits in the binary system
        '''

        # Total circular velocities (kpc/Myr)
        vel = np.sqrt(G * mass* d / (r*r))

        # x, y components of velocities
        velx = np.sin(theta)*vel 
        vely = np.cos(theta)*vel
    
        # Output x, y velocities of object
        return (velx, vely)
  
    def initorbits(self):
        '''
        Calculate initial orbits of the 3 stars

        Returns:
            array which represents the initial phase space coordinates of all 3 stars
            in the system
        '''
        # Coordinates of objects in orbit
        x1    = self.p1[0]
        y1    = self.p1[1]
        x2    = self.p2[0]
        y2    = self.p2[1]
        x3    = self.p3[0]
        y3    = self.p3[1]
        z3    = self.p3[2]

        r1    = (x1**2 + y1**2)**0.5
        r2    = (x2**2 + y2**2)**0.5
        r3    = (x3**2 + y3**2)**0.5

        # Calculates angle of objects with respect to centre of mass  
        theta1 = np.arctan2(y1, x1)
        theta2 = np.arctan2(y2, x2)
        theta3 = np.arctan2(y3, x3)

        # Calculate velocities throughout orbit
        velx1, vely1 = self.initvelocities(r1+r2, theta1, self.mass2, self.p1, r1)
        velx2, vely2 = self.initvelocities(r2+r1, theta2, self.mass1, self.p2, r2)
        # Using the initial velocities of the binary system as a refernce of initial velocity for the third incoming star
        velx3, vely3 = -0.*abs(vely1), 0.
        velz3 = -1*abs(vely1)

        # Output phase space coordinates of orbiting bodies
        return np.array([x1, y1, 0.0, x2, y2, 0.0, x3, y3, z3, \
                        velx1, vely1, 0.0, velx2, vely2, 0.0, velx3, vely3, velz3])

    def dpdt(self,p,t):
        '''
        
        Function that calculates derivatives in phase space
        
        Args:
            p : phase space coordinates : (x,y,z) coordinates of each star, then the velocities (v_x,v_y,v_z)
            t : time array
        Returns:
            array of shape p which represents the derivatives of p element by element
        
        '''
        r12 = ((p[0]-p[3])**2 + (p[1]-p[4])**2 + (p[2]-p[5])**2)**0.5
        r13 = ((p[0]-p[6])**2 + (p[1]-p[7])**2 + (p[2]-p[8])**2)**0.5
        r23 = ((p[3]-p[6])**2 + (p[4]-p[7])**2 + (p[5]-p[8])**2)**0.5

        #Phase space acceleration of the 3 stars
        p_accel_star1x = -self.G*(self.mass2*(p[0]-p[3])/(r12**3.) + self.mass3*(p[0]-p[6])/(r13**3.))
        p_accel_star1y = -self.G*(self.mass2*(p[1]-p[4])/(r12**3.) + self.mass3*(p[1]-p[7])/(r13**3.))
        p_accel_star1z = -self.G*(self.mass2*(p[2]-p[5])/(r12**3.) + self.mass3*(p[2]-p[8])/(r13**3.))
        
        p_accel_star2x = -self.G*(self.mass1*(p[3]-p[0])/(r12**3.) + self.mass3*(p[3]-p[6])/(r23**3.))
        p_accel_star2y = -self.G*(self.mass1*(p[4]-p[1])/(r12**3.) + self.mass3*(p[4]-p[7])/(r23**3.))
        p_accel_star2z = -self.G*(self.mass1*(p[5]-p[2])/(r12**3.) + self.mass3*(p[5]-p[8])/(r23**3.))
        
        p_accel_star3x = -self.G*(self.mass1*(p[6]-p[0])/(r13**3.) + self.mass2*(p[6]-p[3])/(r23**3.))
        p_accel_star3y = -self.G*(self.mass1*(p[7]-p[1])/(r13**3.) + self.mass2*(p[7]-p[4])/(r23**3.))
        p_accel_star3z = -self.G*(self.mass1*(p[8]-p[2])/(r13**3.) + self.mass2*(p[8]-p[5])/(r23**3.))

        #Returning array of phase space coordinates of length 18, velocity followed by acceleration
        return np.array([
                         p[9],p[10],p[11],\
                         p[12],p[13],p[14],\
                         p[15],p[16],p[17],\
                         p_accel_star1x, p_accel_star1y, p_accel_star1z,\
                         p_accel_star2x, p_accel_star2y, p_accel_star2z,\
                         p_accel_star3x, p_accel_star3y, p_accel_star3z
                         ])

    def update_point(self, num, dataLines1, dataLines2, dataLines3, lines1, lines2, lines3):
            
        '''
        Function to update points at each interval during animation
    
        Args:
            num         : number of timesteps to run animation over
            dataLines1  : array of coordinates at each time step to animate for star 1
            dataLines2  : array of coordinates at each time step to animate for star 2
            dataLines3  : array of coordinates at each time step to animate for star 3
            lines1      : actual points being plotted for star 1
            lines2      : actual points being plotted for star 2
            lines3      : actual points being plotted for star 3
    
        Returns:
            plots to be animated
        '''
        
        for line, data in zip(lines1, dataLines1):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(5)
            
        for line, data in zip(lines2, dataLines2):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(5)
           
        for line, data in zip(lines3, dataLines3):
            line.set_data(data[0:2, num])
            line.set_3d_properties(data[2, num])
            line.set_markersize(8)
            
        return lines1, lines2, lines3

    def animate(self, filename):

        '''
        Function to animate 3 body scattering simulation and save to mp4 file.

        Args:
            filename : string to be used as filename for .mp4 file.
        '''
        

        # Calculate particle trajectories using Runge-Kutta 4

        output = rk4(self.dpdt, [self.p_space[0], self.p_space[1], self.p_space[2], \
                                          self.p_space[3], self.p_space[4], self.p_space[5], \
                                          self.p_space[6], self.p_space[7], self.p_space[8], \
                                          self.p_space[9], self.p_space[10], self.p_space[11], \
                                          self.p_space[12], self.p_space[13], self.p_space[14], \
                                          self.p_space[15], self.p_space[16], self.p_space[17]], self.t)

        # Reshape arrays for animation
        animated_output = []
        animated_output2 = []
        animated_output3 = []
        
        animated_output.append(np.array([output[:,0],output[:,1],output[:,2]]))
        animated_output2.append(np.array([output[:,3],output[:,4],output[:,5]]))
        animated_output3.append(np.array([output[:,6],output[:,7],output[:,8]]))

        animated_output = np.array(animated_output)
        animated_output2 = np.array(animated_output2)
        animated_output3 = np.array(animated_output3)
        
        data = animated_output
        data2 = animated_output2
        data3 = animated_output3

        # Intitialize animation
        fig1 = plt.figure()
        ax = p3.Axes3D(fig1)

        l = [ax.plot(dat[0, 0:1],dat[1, 0:1],dat[2, 0:1], 'o')[0] for dat in data]
        l2 = [ax.plot(dat[0, 0:1],dat[1, 0:1],dat[2, 0:1], 'o')[0] for dat in data2]
        l3 = [ax.plot(dat[0, 0:1],dat[1, 0:1],dat[2, 0:1], 'o')[0] for dat in data3]

        # Set x limits from -#*Radius of binary stars to +#*Radius of binary stars
        ax.set_xlim3d([-75*695700, 75*695700])
        ax.set_xlabel('X')

        # Set y limits from -50*Radius of binary stars to +#*Radius of binary stars
        ax.set_ylim3d([-75*695700, 75*695700])
        ax.set_ylabel('Y')

        # Set z limits from -#*Radius of binary stars to +#*Radius of binary stars
        ax.set_zlim3d([-75*695700, 75*695700])
        ax.set_zlabel('Z')

        # Run animation and save
        line_ani = animation.FuncAnimation(fig1, self.update_point, len(data[0][0]), \
                                           fargs=(data, data2, data3, l, l2, l3), interval=1, blit=False)
        line_ani.save(filename+'.mp4', fps = 60, bitrate=7200)       

three = ThreeBody(star1,star2,star3,G,t)

three.animate('ThreeBody')