# Libraries #
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
############

#### GEODESIC INTEGRATOR ####
def F_schwarzschild(x, Y,M):
    # Y = (u, u') where u = 1/r and u' = du/dphi
    u, uprime = Y
    du = uprime
    ddu = -u+3*M*u**2 # Binet equation
    return [du, ddu]
def integratore_schwarzschild(b,r0,M):
    u0=1.0/r0 
    du0=np.sqrt(1.0/b**2-u0**2+2*M*u0**3) 
    
    Y0=[u0,du0] # initial conditions

    # Event 1: Stop if the photon reaches infinity (u = 0)
    def evento_fuga(phi, y,M):
        return y[0] 
    evento_fuga.terminal = True # terminate integration when this event occurs
    evento_fuga.direction = -1.0 # moving towards infinity (u decreasing)

    # Event 2: Stop if the photon falls into the horizon (r <= 2M)
    def evento_orizzonte(phi, y,M):
        return 1.0 / (2*M) - y[0]
    evento_orizzonte.terminal = True # terminate integration when this event occurs
    evento_orizzonte.direction = -1.0 # moving towards the horizon (u increasing 1/2M-u decreasing)
        
    phi_span = (0, 10*np.pi)  # span of integration in phi, 5 orbits max       

    # integrator
    sol = solve_ivp(F_schwarzschild, 
        phi_span, 
        Y0, 
        args=[M],
        events=[evento_fuga, evento_orizzonte],
        rtol=1e-13, 
        atol=1e-13,
        method='DOP853') # grat accuracy, high order method, ideal for smooth problems like this one

    catturato = sol.t_events[1].size > 0 # its True if event 1 (falling into the horizon) occurred
    if catturato:
        return sol, True, np.nan
    
    phi_out = sol.t_events[0][0] # angle of deflection

    return sol, False, phi_out

# Auxiliary functions for Kerr metric
def Sigma(r, theta,M,a):
    return r**2+a**2*(np.cos(theta))**2
def Delta(r,M,a):
    return r**2-2*M*r+a**2
def R(r,b,Q,M,a):
    return (r**2+a**2-a*b)**2-Delta(r,M,a)*(Q+(b-a)**2)
def Th(th,b,Q,M,a):
    return Q+(np.cos(th))**2*(a**2-b**2/(np.sin(th))**2)
def Rprime(r,b,Q,M,a):
    return 4*r*(r**2+a**2-a*b)-2*(Q+(b-a)**2)*(r-M)
def Thprime(th,b,M,a):
    return 2*(-a**2*np.sin(th)*np.cos(th)+b**2*np.cos(th)/(np.sin(th))**3)
def A(r,th,M,a):
    return 1-2*M*r/Sigma(r,th,M,a)
def B(r,th,M,a):
    return 2*M*r*a*(np.sin(th))**2/Sigma(r,th,M,a)

def F_kerr(tau, Y, b, Q,M,a):
    # Y = (r, phi, th, r', th')
    r, phi, th, dotr, dotth = Y
    dr=dotr
    dphi=Sigma(r,th,M,a)*(A(r,th,M,a)*b+B(r,th,M,a))/(Delta(r,M,a)*(np.sin(th))**2)
    dth=dotth
    ddr=Rprime(r,b,Q,M,a)/2
    ddth=Thprime(th,b,M,a)/2
    return [dr, dphi, dth, ddr, ddth]
def integratore_kerr(b,Q,r0,th0,signdotth0,M,a):
    # initial conditions
    Y0=[r0,0,th0,-np.sqrt(R(r0,b,Q,M,a)),signdotth0*np.sqrt(max(Th(th0,b,Q,M,a),0.0))] # max is used to avoid negative values due to numerical errors
    
    # Event 1: Stop if the photon escapes to infinity (r = 1e5)
    def evento_fuga(tau, y,b,Q,M,a):
        return y[0]- 1e5
    evento_fuga.terminal = True
    evento_fuga.direction = 1 # moving towards infinity (r increasing)

    # Event 2: Stop if the photon falls into the horizon (r <= r_+)
    def evento_orizzonte(tau, y,b,Q,M,a):
        return  M+np.sqrt(M**2-a**2) - y[0]+ 1e-6 # small offset to avoid numerical issues
    evento_orizzonte.terminal = True
    evento_orizzonte.direction = 1 # moving towards the horizon (r decreasing r_+ - r increasing)

    # integration span in affine parameter tau, time of Mino
    tau_span = (0, 10000)                     

    # integrator
    sol = solve_ivp(F_kerr, 
        tau_span, 
        Y0, 
        args=(b,Q,M,a),
        events=[evento_fuga, evento_orizzonte],
        rtol=1e-10, 
        atol=1e-10,
        method='DOP853')

    catturato = sol.t_events[1].size > 0 # is True if event 1 (falling into the horizon) occurred
    if catturato:
        return sol, True, np.nan,np.nan
    # angles of deflection
    phi_out = sol.y_events[0][0][1] 
    th_out=sol.y_events[0][0][2]
    return sol, False, phi_out, th_out
#################################

### SCHWARZSCHILD ###
def schwarzshild(M,r0,FOV,N):
    # input:  Mass of the black hole, distance from the black hole, field of view, number of pixels per side of the image
    # output: deflection table for each pixel, black hole shadow mask

    b_c=3*np.sqrt(3)*M      # critical impact parameter

    #           SCREEN of the CAMERA            #
    foc=(N/2)/np.tan(FOV/2)                     # focal lenght
    x=np.linspace(-(N-1)/2, (N-1)/2, N)         # coordinate of the screen
    y=np.linspace(-(N-1)/2, (N-1)/2, N)
    X, Y=np.meshgrid(-x,-y)                     # cartesian grid 
    d_pixel=np.sqrt(X**2+Y**2)                  # map of distances in pixels
    b_pixel=r0*d_pixel/np.sqrt((d_pixel**2+foc**2)*(1-2.0*M/r0)) # map of impact parameters in pixels
    #############################################

    #                           INTERPOLATION                       #
    b_max=b_pixel.max()
    chi_pixel=np.log(np.maximum(b_pixel/b_c - 1.0, 1e-12))          # map of chi(b)=log(b/b_c-1) change of variable
    chi_vec = np.linspace(np.log(1e-9), np.log(b_max/b_c - 1), 300) # list of Chi
    b_vec = b_c*(1 + np.exp(chi_vec))                               # list of b
    phi_vec = [] 
    for b in b_vec:
        _, _, phi_out = integratore_schwarzschild(b, r0,M)
        phi_vec.append(phi_out)
    phi_vec = np.array(phi_vec) # list of phi(b) for interpolation
    phi_pixel= np.interp(np.clip(chi_pixel, chi_vec[0], chi_vec[-1]), chi_vec, phi_vec) # interpolation of phi(b) for each pixel
    #########################################

    #        CALCULATION OF b_c    #
    def catturato(b):
        _, cat, _ = integratore_schwarzschild(b, r0,M)
        return cat
    lo, hi = 4.0*M, 6.0*M          # initial interval for b_c
    for _ in range(30):            # binary search to find b_c 
        mid = 0.5*(lo + hi)
        if catturato(mid): 
            lo = mid
        else:        
            hi = mid
    b_c_numerico=lo/M
    print(f"b_c numerico = {b_c_numerico:.10f} M,  teorico = {b_c/M:.10f} M") # validazione
    #################################
    black=np.zeros((N, N), dtype=bool) # boolean mask for the black hole shadow
    black[b_pixel<b_c_numerico]=1
    return phi_pixel, black
#####################

### KERR ###
def kerr(M,a,r0,th0,FOV,N):
    ### SCREEN ###
    foc=(N/2)/np.tan(FOV/2)   #focal length
    x=np.linspace(-(N-1)/2, (N-1)/2, N)# coordinate of the screen
    y=np.linspace(-(N-1)/2, (N-1)/2, N)
    X, Y=np.meshgrid(x,y) # cartesian grid 
    alpha_pixel=np.arctan(np.sqrt(X**2+Y**2)/foc) # map of angles alpha
    psi_pixel=np.arctan2(Y,X) # map of angles psi
    ###############
    ### b and Q ###
    # auxiliary quantities
    A= (r0**2 + a**2)**2 - a**2*Delta(r0,M,a)*np.sin(th0)**2
    om  = 2*M*a*r0/A
    varpi=np.sqrt(A/Sigma(r0,th0,M,a))*np.sin(th0)
    en  = np.sqrt(Sigma(r0,th0,M,a)*Delta(r0,M,a)/A)
    K=varpi*np.sin(alpha_pixel)*np.cos(psi_pixel)
    D = en + om*K

    b_pixel = K/D
    Q_pixel = Sigma(r0,th0,M,a)*(np.sin(alpha_pixel)*np.sin(psi_pixel)/D)**2 \
              - np.cos(th0)**2*(a**2 - b_pixel**2/np.sin(th0)**2)
    #############
    phi_pixel=np.zeros((N,N), dtype=float)
    th_pixel= np.zeros((N,N), dtype=float)
    black=np.zeros((N, N), dtype=bool)
    for i in range(N): # loop over pixels
        for j in range(N):
            _,black[i][j], phi_pixel[i][j],th_pixel[i][j]=integratore_kerr(b_pixel[i][j],Q_pixel[i][j],r0,th0,Y[i][j]/abs(Y[i][j]),M,a) 
            # integration of the geodesic for each pixel, Y[i][j]/abs(Y[i][j]) gives the sign of dotth0
    return phi_pixel,th_pixel,black
#############

### SKY ###
def cielo(sky, nx,ny,nz):
    # input:  sky image, direction vectors nx, ny, nz
    # output: sky image destorted by the black hole
    lat = np.arccos(np.clip(nz, -1, 1))   # latitude
    long  = np.arctan2(ny, -nx)           # longitude
    long = np.nan_to_num(long, nan=0.0)   # replace NaN values with 0.0
    lat  = np.nan_to_num(lat,  nan=0.0)
    H, W, _ = sky.shape #height and width
    # equirectangular projection of the sky
    col = ((long + np.pi) / (2*np.pi) * W).astype(int) % W # astype(int) to convert to integer indices, %W to wrap around the image
    row = np.clip((lat / np.pi * H).astype(int), 0, H - 1) # clip to ensure indices are within bounds
    return sky[row, col]

def immagine(name_file,phi_pixel,black,FOV,N, th_pixel=None):
    # input:  name of the sky image, deflection table for each pixel, black hole shadow mask, field of view, number of pixels per side of the image, theta angle for Kerr case (optional)
    # output: two images, one with the black hole and one without
    foc=(N/2)/np.tan(FOV/2) 
    x=np.linspace(-(N-1)/2, (N-1)/2, N)         # coordinate of the screen
    y=np.linspace(-(N-1)/2, (N-1)/2, N)
    X, Y=np.meshgrid(-x,-y)  
    d_pixel=np.sqrt(X**2+Y**2)                  # map of distances in pixels
    alpha_pixel=np.arctan(d_pixel/foc)          # map of the alpha angle
    phi_vuoto_pixel=np.pi-alpha_pixel           # map of the deflection in the void

    # vector field of asymptotic directions in the void (nx,ny,nz)
    nx_vuoto_pixel = np.cos(phi_vuoto_pixel)
    ny_vuoto_pixel = np.sin(phi_vuoto_pixel) * X/d_pixel
    nz_vuoto_pixel = np.sin(phi_vuoto_pixel) * Y/d_pixel

    if(th_pixel is None): # Schwarzschild case
        # vector field of asymptotic directions (nx,ny,nz)
        nx_pixel = np.cos(phi_pixel)
        ny_pixel = np.sin(phi_pixel) * X/d_pixel
        nz_pixel = np.sin(phi_pixel) * Y/d_pixel
    else: # Kerr case
        # vector field of asymptotic directions (nx,ny,nz)
        nx_pixel = np.cos(phi_pixel)*np.sin(th_pixel)
        ny_pixel = np.sin(phi_pixel)*np.sin(th_pixel)
        nz_pixel = np.cos(th_pixel)

    sky = plt.imread(name_file) # read the sky image
    sky= sky[:, :, :3]          # remove alpha channel if present
    img1 = cielo(sky, nx_vuoto_pixel, ny_vuoto_pixel, nz_vuoto_pixel)   # sky without black hole
    img2 = cielo(sky, nx_pixel, ny_pixel, nz_pixel)                     # sky with black hole
    img2[black] = 0                                                     # black hole shadow

    # plotting the two images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, titolo in zip(axes, [img1, img2], ["Senza buco nero", "Con buco nero"]):
        ax.imshow(img, origin='lower')
        ax.set_title(titolo)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # return the two images 
    return img1 , img2
################