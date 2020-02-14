import numpy as np
import math

""" Conversion factor for [:math:`^\\circ` C] to [:math:`^\\circ` K] """
CtoK = 273.16  # 273.15
""" von Karman's constant """
kappa = 0.4  # NOTE: 0.41

def charnock_C35(wind,u10n,usr,seastate,waveage,wcp,sigH,lat):
    g=gc(lat,None)
    a1, a2=0.0017, -0.0050    
    charnC=np.where(u10n>19,a1*19+a2,a1*u10n+a2)
    A, B=0.114, 0.622  #wave-age dependent coefficients
    Ad, Bd=0.091, 2.0  #Sea-state/wave-age dependent coefficients
    charnW=A*(usr/wcp)**B
    zoS=sigH*Ad*(usr/wcp)**Bd
    charnS=(zoS*g)/usr**2    
    charn=np.where(wind>10,0.011+(wind-10)/(18-10)*(0.018-0.011),np.where(wind>18,0.018,0.011*np.ones(np.shape(wind))))
    if waveage:
        if seastate:
            charn=charnS
        else:
            charn=charnW
    else:
        charn=charnC
    ac = np.zeros((len(wind),3))
    ac[:,0] = charn
    ac[:,1] = charnS
    ac[:,2] = charnW
    return ac
   
def cd_C35(u10n,wind,usr,charn,monob,Ta,hh_in,lat):
    g=gc(lat,None)
    zo=charn*usr**2/g+0.11*visc_air(Ta)/usr # surface roughness
    rr=zo*usr/visc_air(Ta)
    zoq=np.where(5.8e-5/rr**0.72>1.6e-4,1.6e-4,5.8e-5/rr**0.72) # These thermal roughness lengths give Stanton and
    zot=zoq                                                     # Dalton numbers that closely approximate COARE 3.0
    cdhf=kappa/(np.log(hh_in[0]/zo)-psiu_26(hh_in[0]/monob))    
    cthf=kappa/(np.log(hh_in[1]/zot)-psit_26(hh_in[1]/monob))
    cqhf=kappa/(np.log(hh_in[2]/zoq)-psit_26(hh_in[2]/monob))
    return zo,cdhf, cthf, cqhf
    
def cdn_calc(u10n,Ta,Tp,method="Smith80"):
    if (method == "Smith80"):
        cdn = np.where(u10n <=3,(0.61+0.567/u10n)*0.001,(0.61+0.063*u10n)*0.001)
    elif (method == "LP82"):
        cdn = np.where((u10n < 11) & (u10n>=4),1.2*0.001,np.where((u10n<=25) & (u10n>=11),(0.49+0.065*u10n)*0.001,1.14*0.001))            
    elif (method == "Smith88" or method == "COARE3.0" or method == "COARE4.0"):
        cdn = cdn_from_roughness(u10n,Ta,None, method)
    elif (method == "HEXOS"):
        cdn = (0.5+0.091*u10n)*0.001 #Smith et al. 1991 #(0.27 + 0.116*u10n)*0.001  Smith et al. 1992        
    elif (method == "HEXOSwave"):
        cdn=cdn_from_roughness(u10n,Ta,Tp, method)
    elif (method == "YT96"):  
        cdn = np.where((u10n<6) & (u10n>=3),(0.29 + 3.1/u10n + 7.7/u10n**2)*0.001,np.where((u10n<=26) & (u10n>=6),(0.60 + 0.070*u10n)*0.001,(0.61+0.567/u10n)*0.001)) # for u<3 same as Smith80
    elif (method == "LY04"):
        cdn = np.where(u10n>=0.5,(0.142 + (2.7/u10n) + (u10n/13.09))*0.001,np.nan)
    else:
        print("unknown method cdn: "+method)

    return cdn
 
#---------------------------------------------------------------------

def cdn_from_roughness(u10n,Ta,Tp,method="Smith88"):  
    g,tol = 9.812, 0.000001
    cdn,ustar = np.zeros(np.asarray(u10n).shape),np.zeros(np.asarray(u10n).shape)
    cdnn = (0.61 + 0.063 * u10n) * 0.001   
    zo,zc,zs=np.zeros(np.asarray(u10n).shape),np.zeros(np.asarray(u10n).shape),np.zeros(np.asarray(u10n).shape)
    for it in range(5):
        cdn = np.copy(cdnn)
        ustar = np.sqrt(cdn*u10n**2)
        if (method == "Smith88"):
            zc = 0.011*ustar**2/g #.....Charnock roughness length (equn 4 in Smith 88)
            zs = 0.11*visc_air(Ta)/ustar #.....smooth surface roughness length (equn 6 in Smith 88)
            zo = zc + zs #.....equns 7 & 8 in Smith 88 to calculate new CDN
        elif (method == "COARE3.0"):
            zc = 0.011 + (u10n-10)/(18-10)*(0.018-0.011)
            zc=np.where(u10n<10,0.011,np.where(u10n>18,0.018,zc))
            zs = 0.11 * visc_air(Ta)/ ustar
            zo = zc*ustar*ustar/g + zs
        elif (method == "HEXOSwave"):
            if (np.all(Tp==None) or np.nansum(Tp)==0):
                Tp = 0.729*u10n # Taylor and Yelland 2001            
            cp_wave = g*Tp/2/np.pi # use input wave period
            zo = 0.48*ustar**3/g/cp_wave # Smith et al. 1992
        else:
            print("unknown method for cdn_from_roughness "+method)    
        cdnn = (kappa/np.log(10/zo))**2
    cdn=np.where(np.abs(cdnn-cdn) < tol, cdnn,np.nan)
       
    return cdnn
#---------------------------------------------------------------------

def ctcqn_calc(zol, cdn, u10n, zo,Ta, method="Smith80"):
    l = np.shape(u10n)
    if (method == "Smith80" or method == "Smith88" or method == "YT96"):
        cqn = np.ones(l)*1.20*0.001 # from Smith88, no value given by S80 or YT80
        ctn = np.ones(l)*1.00*0.001
    elif (method == "LP82"):
        cqn = np.where((zol <= 0) & (u10n>4) & (u10n<14),1.15*0.001,np.nan)                     
        ctn = np.where((zol <= 0) & (u10n>4) & (u10n<25), 1.13*0.001, 0.66*0.001) 
    elif (method == "HEXOS" or method == "HEXOSwave"):
        cqn = np.where((u10n<=23) & (u10n>=3),1.1*0.001,np.nan)
        ctn = np.where((u10n<=18) & (u10n>=3),1.1*0.001,np.nan)
    elif (method == "COARE3.0" or method == "COARE4.0"):
        ustar = (cdn * u10n**2)**0.5
        rr = zo*ustar/visc_air(Ta)
        zoq = 5.5e-5/rr**0.6
        zoq[zoq>1.15e-4]  = 1.15e-4
        zot = zoq
        cqn = kappa**2/np.log(10/zo)/np.log(10/zoq)
        ctn = kappa**2/np.log(10/zo)/np.log(10/zot)
    elif (method == "LY04"):
        cqn = 34.6*0.001*cdn**0.5
        ctn = np.where(zol <= 0, 32.7*0.001*cdn**0.5, 18*0.001*cdn**0.5)
    else:
        print("unknown method ctcqn: "+method)
        
    return ctn, cqn
#---------------------------------------------------------------------
 
def cd_calc(cdn, height, ref_ht, psim):
    cd = cdn*np.power(1+np.sqrt(cdn)*np.power(kappa,-1)*(np.log(height/ref_ht)-psim),-2)
    return cd
#---------------------------------------------------------------------
 
def ctcq_calc(cdn, cd, ctn, cqn, h_t, h_q, ref_ht, psit, psiq):
    ct = ctn*(cd/cdn)**0.5/(1+ctn*((np.log(h_t/ref_ht)-psit)/(kappa*cdn**0.5)))
    cq = cqn*(cd/cdn)**0.5/(1+cqn*((np.log(h_q/ref_ht)-psiq)/(kappa*cdn**0.5)))
    return ct, cq
#---------------------------------------------------------------------
 
def psim_calc(zol, method="Smith80"): 
    coeffs = get_stabco(method)
    alpha, beta, gamma = coeffs[0], coeffs[1], coeffs[2]
    if (method == "COARE3.0" or method == "COARE4.0"):
        psim = np.where(zol<0,psim_conv_coare3(zol,alpha,beta,gamma),psim_stab_coare3(zol,alpha,beta,gamma))
    else:
        psim = np.where(zol<0,psim_conv(zol,alpha,beta,gamma),psim_stab(zol,alpha,beta,gamma))
    
    return psim
#---------------------------------------------------------------------
 
def psit_calc(zol, method="Smith80"):
    coeffs = get_stabco(method)
    alpha, beta, gamma = coeffs[0], coeffs[1], coeffs[2]
    if (method == "COARE3.0" or method == "COARE4.0"):
        psit = np.where(zol<0,psi_conv_coare3(zol,alpha,beta,gamma),psi_stab_coare3(zol,alpha,beta,gamma))
    else:
        psit = np.where(zol<0,psi_conv(zol,alpha,beta,gamma),psi_stab(zol,alpha,beta,gamma))
        
    return psit
#---------------------------------------------------------------------
def get_stabco(method="Smith80"):
    if (method == "Smith80" or method == "Smith88" or method == "LY04"):
        # Smith 1980, from Dyer (1974)
        alpha, beta, gamma = 16, 0.25, 5
    elif (method == "LP82"):
        alpha, beta, gamma = 16, 0.25, 7 
    elif (method == "HEXOS" or method == "HEXOSwave"):
        alpha, beta, gamma = 16, 0.25, 8
    elif (method == "YT96"):
        alpha, beta, gamma = 20, 0.25, 5
    elif (method == "COARE3.0" or method == "COARE4.0"):
        # use separate subroutine
        alpha, beta, gamma = 15, 1/3, 5   # not sure about gamma=34.15
        #alpha <- NA  #beta <- NA
    else:
        print("unknown method stabco: "+method)        
    coeffs = np.zeros(3)
    coeffs[0] = alpha
    coeffs[1] = beta
    coeffs[2] = gamma
          
    return coeffs
#---------------------------------------------------------------------
#======================================================================
def psi_conv_coare3(zol,alpha,beta,gamma):
    x = (1-alpha*zol)**0.5      # Kansas unstable
    psik = 2*np.log((1+x)/2.)
    y = (1-34.15*zol)**beta
    psic = 1.5*np.log((1+y+y*y)/3.)-(3)**0.5*np.arctan((1+2*y)/
                         (3)**0.5)+4*np.arctan(1)/(3)**0.5
    f = zol*zol/(1.+zol*zol)
    psit = (1-f)*psik+f*psic    
    return psit
#======================================================================
def psi_stab_coare3(zol,alpha,beta,gamma):          #Stable
    c = np.where(0.35*zol > 50, 50, 0.35*zol)       #Stable
    psit = -((1+2*zol/3)**1.5+0.6667*(zol-14.28)/np.exp(c)+8.525)    
    return psit
#======================================================================
def psi_conv(zol,alpha,beta,gamma):
    xtmp = (1 - alpha*zol)**beta
    psit = 2*np.log((1+xtmp**2)*0.5)    
    return psit
#======================================================================
def psi_stab(zol,alpha,beta,gamma):
    psit = -gamma*zol           
    return psit
#======================================================================
def psim_conv_coare3(zol,alpha,beta,gamma):
    x = (1-15*zol)**0.25        #Kansas unstable
    psik = 2*np.log((1+x)/2)+np.log((1+x*x)/2)-2*np.arctan(x)+2*np.arctan(1)
    y = (1-10.15*zol)**0.3333                   #Convective
    psic = 1.5*np.log((1+y+y*y)/3.)-np.sqrt(3)*np.arctan((1+2*y)/np.sqrt(3))+4.*np.arctan(1)/np.sqrt(3)
    f = zol*zol/(1+zol*zol)
    psim = (1-f)*psik+f*psic
    return psim
#======================================================================
def psim_stab_coare3(zol,alpha,beta,gamma):
    c = np.where(0.35*zol > 50, 50, 0.35*zol)       #Stable
    psim = -((1+1*zol)**1.0+0.6667*(zol-14.28)/np.exp(-c)+8.525)
    return psim
#======================================================================
def psim_conv(zol,alpha,beta,gamma):
    xtmp = (1 - alpha*zol)**beta
    psim = 2*np.log((1+xtmp)*0.5)+np.log((1+xtmp**2)*0.5)-2*np.arctan(xtmp)+np.pi/2                    
    return psim
#======================================================================
def psim_stab(zol,alpha,beta,gamma):
  psim = -gamma*zol
  return psim
#======================================================================
#---------------------------------------------------------------------
def get_skin(sst,qsea,rho,jcool,Rl,Rs,Rnl,cp,lv,usr,tsr,qsr,lat):
  # coded following Saunders (1967) with lambda = 6
  g=gc(lat,None)
  if ( np.nanmin(sst) > 200 ): # if Ta in Kelvin convert to Celsius
        sst = sst-273.16
  #************  cool skin constants  *******
  rhow, cpw, visw, tcw = 1022, 4000, 1e-6, 0.6  # density of water, specific heat capacity of water, water viscosity, thermal conductivity of water
  Al = 2.1e-5*(sst+3.2)**0.79
  be = 0.026
  bigc = 16*g*cpw*(rhow*visw)**3/(tcw*tcw*rho*rho)
  wetc = 0.622*lv*qsea/(287.1*(sst+273.16)**2)
  Rns = 0.945*Rs # albedo correction
  hsb=-rho*cp*usr*tsr
  hlb=-rho*lv*usr*qsr
  qout=Rnl+hsb+hlb
  tkt = 0.001*np.ones(np.shape(sst))
  dels=Rns*(0.065+11*tkt-6.6e-5/tkt*(1-np.exp(-tkt/8.0e-4)))
  qcol=qout-dels
  alq=Al*qcol+be*hlb*cpw/lv
  xlamx=np.where(alq>0,6/(1+(bigc*alq/usr**4)**0.75)**0.333,6)
  tkt=xlamx*visw/(np.sqrt(rho/rhow)*usr) #np.nanmin(0.01, xlamx*visw/(np.sqrt(rhoa/rhow)*usr))
  tkt=np.where(alq>0,np.where(tkt > 0.01, 0.01,tkt),tkt)
  dter=qcol*tkt/tcw
  dqer=wetc*dter
  return dter, dqer
#---------------------------------------------------------------------
def get_gust(Ta,usr,tsrv,zi,lat): 
   if (np.max(Ta)<200): # convert to K if in Celsius
       Ta=Ta+273.16         
   if np.isnan(zi):
      zi = 600  
   g=gc(lat,None)
   Bf=-g/Ta*usr*tsrv
   ug=np.ones(np.shape(Ta))*0.2
   ug=np.where(Bf>0,1.2*(Bf*zi)**0.333,0.2) 
   return ug
#---------------------------------------------------------------------
def get_heights(h):
    hh=np.zeros(3)
    if type(h) == float or type(h) == int:
       hh[0], hh[1], hh[2] = h, h, h
    elif len(h)==2:
       hh[0], hh[1], hh[2] = h[0], h[1], h[1]
    else:
       hh[0], hh[1], hh[2] = h[0], h[1], h[2]       
    return hh
#---------------------------------------------------------------------
def svp_calc(T):
# t is in Kelvin
# svp in mb, pure water
  if ( np.nanmin(T) < 200 ): # if T in Celsius convert to Kelvin
        T = T+273.16  
  svp = np.where(np.isnan(T), np.nan,2.1718e08*np.exp(-4157/(T-33.91-0.16)))
  return svp
#---------------------------------------------------------------------
 
def qsea_calc(sst,pres):
# sst in Kelvin
# pres in mb
# qsea in kg/kg
   if ( np.nanmin(sst) < 200 ): # if sst in Celsius convert to Kelvin
        sst = sst+273.16
        
   ed = svp_calc(sst)
   e = 0.98*ed 
   qsea = (0.622*e)/(pres-0.378*e)
   qsea = np.where(~np.isnan(sst+pres),qsea,np.nan)
   return qsea
#---------------------------------------------------------------------

def rh_calc(Ta,qair,pres):
  if ( np.nanmin(Ta) < 200 ): # if sst in Celsius convert to Kelvin
     Ta = Ta+273.16  
  e = np.where(np.isnan(Ta+qair+pres),np.nan,(qair*pres)/(0.62197+qair*0.378))
  ed = np.where(np.isnan(e),np.nan,svp_calc(Ta))
  rh = np.where(np.isnan(ed),np.nan,e/ed*100)
  return rh
#---------------------------------------------------------------------
 
def q_calc(Ta,rh,pres):
# rh in %
# air in K, if not it will be converted to K
# pres in mb
# qair in kg/kg
  if ( np.nanmin(Ta) < 200 ): # if sst in Celsius convert to Kelvin
     Ta = Ta+273.15
  
  e = np.where(np.isnan(Ta+rh+pres),np.nan,svp_calc(Ta)*rh*0.01)
  qair = np.where(np.isnan(e),np.nan,((0.62197*e)/(pres-0.378*e))) # Haltiner and Martin p.24
  return qair
#---------------------------------------------------------------------
def gc(lat,lon=None):
    """
    computes gravity relative to latitude
    inputs:
        lat : latitudes in deg
        lon : longitudes (optional)
    output:
        gc: gravity constant
    """
    gamma = 9.7803267715
    c1 = 0.0052790414
    c2 = 0.0000232718
    c3 = 0.0000001262
    c4 = 0.0000000007
    if lon is not None:
        lon_m,lat_m = np.meshgrid(lon,lat)
    else:
        lat_m = lat
    phi = lat_m*np.pi/180.
    xx = np.sin(phi)
    gc = gamma*(1+c1*np.power(xx,2)+c2*np.power(xx,4)+c3*np.power(xx,6)+c4*np.power(xx,8))    
    return gc
#---------------------------------------------------------------------
def PtoDepth(p,Lat):
    """
    computes depth in m from pressure in db following 
    Saunders and Fofonoff (1976). Deep sea res.
    """
    x=math.sin(math.radians(Lat))
    x=x*x
#    gravity variation with latitude (Anon, 1970)
    gr = 9.780318 * (1 + (5.2788E-3 + 2.36E-5 * x) * x) + 1.092E-6 * p
    d = (((-1.82E-15 * p + 2.279E-10) * p - 2.2512E-5) * p + 9.72659) * p;
    depth = d/gr
    
    return depth
#---------------------------------------------------------------------
def visc_air(Ta):
    """
    Computes the kinematic viscosity of dry air as a function of air temperature
    following Andreas (1989), CRREL Report 89-11.
    input:
        Ta : air temperature [Celsius]
    output
    visa : kinematic viscosity [m^2/s]
    """
    Ta = np.asarray(Ta)
    if ( np.nanmin(Ta) > 200 ): # if Ta in Kelvin convert to Celsius
        Ta = Ta-273.16
    visa = 1.326e-5 * (1 + 6.542e-3*Ta + 8.301e-6*Ta**2 - 4.84e-9*Ta**3)
    return visa

######
# functions from coare35vn.mat 
######
#------------------------------------------------------------------------------
def psit_26(zet):
    """
    computes temperature structure function
    """
    dzet= np.where(0.35*zet > 50, 50, 0.35*zet)  # stable
    psi=-((1+0.6667*zet)**1.5+0.6667*(zet-14.28)*np.exp(-dzet)+8.525)
    k=np.where(zet<0) # unstable
    x=(1-15*zet[k])**0.5
    psik=2*np.log((1+x)/2)
    x=(1-34.15*zet[k])**0.3333
    psic=1.5*np.log((1+x+x**2)/3)-np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3))+4*np.arctan(1)/np.sqrt(3)
    f=zet[k]**2/(1+zet[k]**2)
    psi[k]=(1-f)*psik+f*psic
    return psi
#------------------------------------------------------------------------------
def psiu_26(zet):
    """
    computes velocity structure function
    """
    dzet=np.where(0.35*zet > 50, 50, 0.35*zet) # stable
    a, b, c, d= 0.7, 3/4, 5, 0.35
    psi=-(a*zet+b*(zet-c/d)*np.exp(-dzet)+b*c/d)
    k=np.where(zet<0) # unstable
    x=(1-15*zet[k])**0.25
    psik=2*np.log((1+x)/2)+np.log((1+x**2)/2)-2*np.arctan(x)+2*np.arctan(1)
    x=(1-10.15*zet[k])**0.3333
    psic=1.5*np.log((1+x+x**2)/3)-np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3))+4*np.arctan(1)/np.sqrt(3)
    f=zet[k]**2/(1+zet[k]**2)
    psi[k]=(1-f)*psik+f*psic
    return psi
#------------------------------------------------------------------------------
def psiu_40(zet):
    """
    computes velocity structure function
    """
    dzet=np.where(0.35*zet > 50, 50, 0.35*zet) # stable
    a, b, c, d= 1, 3/4, 5, 0.35
    psi=-(a*zet+b*(zet-c/d)*np.exp(-dzet)+b*c/d)
    k=np.where(zet<0) # unstable
    x=(1-18*zet[k])**0.25
    psik=2*np.log((1+x)/2)+np.log((1+x**2)/2)-2*np.arctan(x)+2*np.arctan(1)
    x=(1-10*zet[k])**0.3333
    psic=1.5*np.log((1+x+x**2)/3)-np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3))+4*np.arctan(1)/np.sqrt(3)
    f=zet[k]**2/(1+zet[k]**2)
    psi[k]=(1-f)*psik+f*psic
    return psi
#------------------------------------------------------------------------------
def  bucksat(T,P):
    """
    computes saturation vapor pressure [mb]
    given T [degC] and P [mb]
    """
    T = np.asarray(T)
    if ( np.nanmin(T) > 200 ): # if Ta in Kelvin convert to Celsius
        T = T-CtoK
    exx=6.1121*np.exp(17.502*T/(T+240.97))*(1.0007+3.46e-6*P)
    return exx
#------------------------------------------------------------------------------
def qsat26sea(T,P):
    """
    computes surface saturation specific humidity [g/kg]
    given T [degC] and P [mb]
    """
    T = np.asarray(T)
    if ( np.nanmin(T) > 200 ): # if Ta in Kelvin convert to Celsius
        T = T-CtoK
    ex=bucksat(T,P)
    es=0.98*ex # reduction at sea surface
    qs=622*es/(P-0.378*es)
    return qs
#------------------------------------------------------------------------------
def qsat26air(T,P,rh):
    """
    computes saturation specific humidity [g/kg]
    given T [degC] and P [mb]
    """
    T = np.asarray(T)
    if ( np.nanmin(T) > 200 ): # if Ta in Kelvin convert to Celsius
        T = T-CtoK
    es=bucksat(T,P)
    em=0.01*rh*es
    q=622*em/(P-0.378*em)
    return q,em
#------------------------------------------------------------------------------
def RHcalc(T,P,Q):
    """
    computes relative humidity given T,P, & Q
    """
    es=6.1121*np.exp(17.502*T/(T+240.97))*(1.0007+3.46e-6*P)
    em=Q*P/(0.378*Q+0.622)
    RHrf=100*em/es
    return RHrf