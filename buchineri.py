# Librerie #
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
############

#### INTEGRATORE GEODETICHE ####
def F_schwarzschild(x, Y,M):
    y, z = Y
    dydx = z
    dzdx = -y+3*M*y**2 # equazione di binet
    return [dydx, dzdx]
def integratore_schwarzschild(b,r0,M):
    u0=1.0/r0 
    du0=np.sqrt(1.0/b**2-u0**2+2*M*u0**3) 
    
    Y0=[u0,du0] # condizioni iniziali

    # Evento 1: Stop se il fotone scappa all'infinito (u = 0)
    def evento_fuga(phi, y,M):
        return y[0] 
    evento_fuga.terminal = True
    evento_fuga.direction = -1 # nella fase di allontanamento

    # Evento 2: Stop se il fotone cade nell'orizzonte (r <= 2M)
    def evento_orizzonte(phi, y,M):
        return 1.0 / (2*M) - y[0]
    evento_orizzonte.terminal = True
    evento_orizzonte.direction = -1.0
        
    phi_span = (0, 10*np.pi)  # intervallo di integrazione         

    #integratore
    sol = solve_ivp(F_schwarzschild, 
        phi_span, 
        Y0, 
        args=[M],
        events=[evento_fuga, evento_orizzonte],
        rtol=1e-13, 
        atol=1e-13,
        method='DOP853')

    catturato = sol.t_events[1].size > 0 # è True se l'evento 1 (caduta nell'orizzonte) è avvenuto
    if catturato:
        return sol, True, np.nan
    
    phi_out = sol.t_events[0][0] #angolo di deflessione

    return sol, False, phi_out

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
    # Y = (r, phi, th, r', th'), derivate rispetto al tempo di Mino
    r, phi, th, dotr, dotth = Y
    dr=dotr
    dphi=Sigma(r,th,M,a)*(A(r,th,M,a)*b+B(r,th,M,a))/(Delta(r,M,a)*(np.sin(th))**2)
    dth=dotth
    ddr=Rprime(r,b,Q,M,a)/2
    ddth=Thprime(th,b,M,a)/2

    return [dr, dphi, dth, ddr, ddth]
def integratore_kerr(b,Q,r0,th0,signdotth0,M,a):

    # condizioni iniziali
    Y0=[r0,0,th0,-np.sqrt(R(r0,b,Q,M,a)),signdotth0*np.sqrt(max(Th(th0,b,Q,M,a),0.0))] # max: evita sqrt(-1e-31) per errori di arrotondamento
    
    # Condizione 1: Stop se il fotone scappa all'infinito (r = 1e5)
    def evento_fuga(tau, y,b,Q,M,a):
        return y[0]- 1e5
    evento_fuga.terminal = True
    evento_fuga.direction = 1 # nella fase di allontanamento

    # Condizione 2: Stop se il fotone cade nell'orizzonte (r <= r_+)
    def evento_orizzonte(tau, y,b,Q,M,a):
        return  M+np.sqrt(M**2-a**2) - y[0]+ 1e-6
    evento_orizzonte.terminal = True
    evento_orizzonte.direction = 1

    # intervallo di integrazione 
    tau_span = (0, 10000)                     

    #integratore
    sol = solve_ivp(F_kerr, 
        tau_span, 
        Y0, 
        args=(b,Q,M,a),
        events=[evento_fuga, evento_orizzonte],
        rtol=1e-10, 
        atol=1e-10,
        method='DOP853')

    catturato = sol.t_events[1].size > 0 #è True se l'evento 1 (caduta nell'orizzonte) è avvenuto
    if catturato:
        return sol, True, np.nan,np.nan
    #angoli di deflessione
    phi_out = sol.y_events[0][0][1] 
    th_out=sol.y_events[0][0][2]
    return sol, False, phi_out, th_out
#################################

### SCHWARZSCHILD ###
def schwarzshild(M,r0,FOV,N):
    # input:  Massa buco nero, distanza dal buco nero, field of view,numero di pixel per lato dell'immagine
    # output: tabella di deflessione per ogni pixel, b_c numerico
         
    b_c=3*np.sqrt(3)*M      # parametro d'impatto critico

    #           SCHERMO FOTOCAMERA              #
    foc=(N/2)/np.tan(FOV/2)                     # focale
    x=np.linspace(-(N-1)/2, (N-1)/2, N)         # coordinate dello schermo
    y=np.linspace(-(N-1)/2, (N-1)/2, N)
    X, Y=np.meshgrid(-x,-y)                     # griglia cartesiana 
    d_pixel=np.sqrt(X**2+Y**2)                  # mappa distanze in pixel
    b_pixel=r0*d_pixel/np.sqrt((d_pixel**2+foc**2)*(1-2.0*M/r0)) # mappa di b 
    #############################################

    # VETTORE phi_out(b) per INTERPOLAZIONE                         #
    b_max=b_pixel.max()
    chi_pixel=np.log(np.maximum(b_pixel/b_c - 1.0, 1e-12))          # mappa di chi(b)=log(b/b_c-1) cambio variabile
    chi_vec = np.linspace(np.log(1e-9), np.log(b_max/b_c - 1), 300) # lista di Chi
    b_vec = b_c*(1 + np.exp(chi_vec))                               # lista di b
    phi_vec = [] 
    for b in b_vec:
        _, _, phi_out = integratore_schwarzschild(b, r0,M)
        phi_vec.append(phi_out)
    phi_vec = np.array(phi_vec)
    phi_pixel= np.interp(np.clip(chi_pixel, chi_vec[0], chi_vec[-1]), chi_vec, phi_vec) #interpolazione
    #########################################

    #        CALCOLO DI b_c        #
    def catturato(b):
        _, cat, _ = integratore_schwarzschild(b, r0,M)
        return cat
    lo, hi = 4.0*M, 6.0*M          # lo: catturato, hi: sfugge
    for _ in range(30):            # bisezione per calcolare b_c  
        mid = 0.5*(lo + hi)
        if catturato(mid): 
            lo = mid
        else:        
            hi = mid
    b_c_numerico=lo/M
    print(f"b_c numerico = {b_c_numerico:.10f} M,  teorico = {b_c/M:.10f} M") # validazione
    #################################
    black=np.zeros((N, N), dtype=bool)
    black[b_pixel<b_c_numerico]=1
    return phi_pixel, black
#####################

### KERR ###
def kerr(M,a,r0,th0,FOV,N):
    ### SCHERMO ###
    foc=(N/2)/np.tan(FOV/2)   #focale
    x=np.linspace(-(N-1)/2, (N-1)/2, N)# coordinate dello schermo
    y=np.linspace(-(N-1)/2, (N-1)/2, N)
    X, Y=np.meshgrid(x,y) # griglia cartesiana 
    ###############
    ### b e Q ###
    Sig = r0**2 + a**2*np.cos(th0)**2
    Del = r0**2 - 2*M*r0 + a**2
    A   = (r0**2 + a**2)**2 - a**2*Del*np.sin(th0)**2
    om  = 2*M*a*r0/A
    al  = np.sqrt(Sig*Del/A)

    D       = np.sqrt(X**2 + Y**2 + foc**2)
    K       = A*np.sin(th0)/(Sig*np.sqrt(Del)) * X/D
    b_pixel = -K/(1 - om*K)
    pt      = (1 - om*b_pixel)/al
    pth     = np.sqrt(Sig)*pt*Y/D          # serve anche per il segno di theta-dot
    Q_pixel = pth**2 - a**2*np.cos(th0)**2 + b_pixel**2/np.tan(th0)**2
    #############
    phi_pixel=np.zeros((N,N), dtype=float)
    th_pixel= np.zeros((N,N), dtype=float)
    black=np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            _,black[i][j], phi_pixel[i][j],th_pixel[i][j]=integratore_kerr(b_pixel[i][j],Q_pixel[i][j],r0,th0,Y[i][j]/abs(Y[i][j]),M,a)
    return phi_pixel,th_pixel,black
#############

### IMMAGINI ###
def cielo(sky, nx,ny,nz):
    #a una direzione n restituisce il pixel nel cielo
    lat = np.arccos(np.clip(nz, -1, 1))   # colatitudine
    long  = np.arctan2(ny, -nx)          # longitudine
    long = np.nan_to_num(long, nan=0.0) # se ci sono dei nan vengono posti a 0
    lat  = np.nan_to_num(lat,  nan=0.0)
    H, W, _ = sky.shape #height and width
    col = ((long + np.pi) / (2*np.pi) * W).astype(int) % W
    row = np.clip((lat / np.pi * H).astype(int), 0, H - 1)
    return sky[row, col]
def immagine(name_file,phi_pixel,black,M,r0,FOV,N, th_pixel=None):
    foc=(N/2)/np.tan(FOV/2) 
    x=np.linspace(-(N-1)/2, (N-1)/2, N)         # coordinate dello schermo
    y=np.linspace(-(N-1)/2, (N-1)/2, N)
    X, Y=np.meshgrid(-x,-y)  
    d_pixel=np.sqrt(X**2+Y**2)                  # mappa distanze in pixel
    alpha_pixel=np.arctan(d_pixel/foc)          # mappa dell'angolo alpha
    phi_vuoto_pixel=np.pi-alpha_pixel           # mappa della deflessione nel vuoto

    # campo vettoriale di direzioni asintotiche nel vuoto (nx,ny,nz)

    nx_vuoto_pixel = np.cos(phi_vuoto_pixel)
    ny_vuoto_pixel = np.sin(phi_vuoto_pixel) * X/d_pixel
    nz_vuoto_pixel = np.sin(phi_vuoto_pixel) * Y/d_pixel

    if(th_pixel is None):
        # campo vettoriale di direzioni asintotiche (nx,ny,nz)
        nx_pixel = np.cos(phi_pixel)
        ny_pixel = np.sin(phi_pixel) * X/d_pixel
        nz_pixel = np.sin(phi_pixel) * Y/d_pixel
    else:
        nx_pixel = np.cos(phi_pixel)*np.sin(th_pixel)
        ny_pixel = np.sin(phi_pixel)*np.sin(th_pixel)
        nz_pixel = np.cos(th_pixel)

    sky = plt.imread(name_file)
    sky= sky[:, :, :3]          # scarta l'eventuale canale alpha
    img1 = cielo(sky, nx_vuoto_pixel, ny_vuoto_pixel, nz_vuoto_pixel)   # cielo senza buco nero
    img2 = cielo(sky, nx_pixel, ny_pixel, nz_pixel)                     # cielo deflesso
    img2[black] = 0                                                # ombra

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, titolo in zip(axes, [img1, img2], ["Senza buco nero", "Con buco nero"]):
        ax.imshow(img, origin='lower')
        ax.set_title(titolo)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return img1 , img2
################