import os
import numpy as np
from numpy import pi, sqrt, arctan, sin, cos, roots, size, real, imag, sqrt, exp
import const


def STNRmodal(f,Susp,ifo):
    T=Susp.Temp #Temperature (K)
    Tmirror=Susp.Stage[0].Temp
    Trm=Susp.Stage[1].Temp
    Tmario=Susp.Stage[2].Temp
    WireMat0=Susp.Stage[0].WireMaterial
    WireMat1=Susp.Stage[1].WireMaterial
    WireMat2=Susp.Stage[2].WireMaterial


    g=const.g
    kb=const.kB
    w=2*pi*f
    Lharm=ifo.Infrastructure.Length

    Mmirror=Susp.Stage[0].Mass   #Mirror mass
    nw=Susp.Stage[0].NWires
    Lmirror=Susp.Stage[0].Length
    dwmirror=Susp.Stage[0].WireDiameter
    
    
    alphaFS=Susp[WireMat0].Alpha # FS expansion coefficient [1/K]
    betaFS=Susp[WireMat0].dlnEdT # FS 1/E dE/dT [1/K]
    CFS=Susp[WireMat0].C #FS Thermal coefficient [J/(kg K)]
    KFS=Susp[WireMat0].K #FS thermal conduction [W/(m K)]
    Ysilica=Susp[WireMat0].Y
    rhoFS=Susp[WireMat0].Rho
    phi0=Susp[WireMat0].Phi # FS bulk loss angle
    ds=Susp[WireMat0].Dissdepth #Surface depth
    Qvhmirror=Susp.Stage[0].Qvh
    rwmirror=dwmirror/2
    I2mirror=pi/4*rwmirror**4
    Amirror=pi*rwmirror**2
    Lambda=Mmirror*g/nw
    omega2g=(g/Lmirror)**0.5
    k2v=nw*Amirror*Ysilica/Lmirror
    omega02v=(k2v/Mmirror)**0.5
    thetaHV=Susp.VHCoupling.theta
    
    phithT=np.zeros(len(f),dtype = "complex_")
    phiT=np.zeros(len(f),dtype = "complex_")
    phiP=np.zeros(len(f),dtype = "complex_")  #Mirror Pendulum Loss Angle
    
    DFmirror=(1/Lmirror)*(nw*Ysilica*I2mirror/Mmirror/g)**0.5    #Dilution factor
    mu=4/rwmirror
    #phie=(muw*hbs*Sw/Vw)*phisurf;  #eccess losses
    phie=mu*ds*phi0
    #Thermoelastic noise
    Delta=(Ysilica*Tmirror/rhoFS/CFS)*(alphaFS-betaFS*(Lambda/Amirror/Ysilica))**2
    tau=((CFS*(2*rwmirror)**2)/2.16/2/pi/KFS)*rhoFS
    phiST=phie+phi0
    for i in range(len(w)):
        phithT[i]=Delta*w[i]*tau/(1+w[i]**2*tau**2)
    
    phiT=phithT+phiST
    phiP=DFmirror*phiT
    
    #########################Reaction Mass#################
    Mrm=Susp.Stage[1].Mass   #Mirror mass
    nwrm=Susp.Stage[1].NWires
    Lrm=Susp.Stage[1].Length
    dwrm=Susp.Stage[1].WireDiameter
    
    alphac85=Susp[WireMat1].Alpha # FS expansion coefficient [1/K]
    betac85=Susp[WireMat1].dlnEdT# FS 1/E dE/dT [1/K]
    Cc85=Susp[WireMat1].C# FS Thermal coefficient [J/(kg K)]
    Kc85=Susp[WireMat1].K# FS thermal conduction [W/(m K)]
    Yc85=Susp[WireMat1].Y#
    rhoc85=Susp[WireMat1].Rho#
    phic85=Susp[WireMat1].Phi # FS bulk loss angle
    dsc85=Susp[WireMat1].Dissdepth# Surface depth
    Qvhrm=Susp.Stage[1].Qvh#
    Qvvrm=Susp.Stage[1].Qvv#
    
    rwrm=dwrm/2
    I2rm=pi/4*rwrm**4
    Arm=pi*rwrm**2
    Lambdarm=Mrm*g/nwrm
    omega3g=(g/Lrm)**0.5
    k3v=nwrm*Arm*Yc85/Lrm
    omega03v=(k3v/Mrm)**0.5
    
    phithTrm=np.zeros(len(f),dtype = "complex_")
    phiTrm=np.zeros(len(f),dtype = "complex_")
    phirm0=np.zeros(len(f),dtype = "complex_") #Reaction Mass Loss Angle
    
    DFrm=(1/Lrm)*(nwrm*Yc85*I2rm/Mrm/g)**0.5    #Dilution factor
    murm=4/rwrm
    #phie=(muw*hbs*Sw/Vw)*phisurf;  %eccess losses
    phierm=murm*dsc85*phic85
    #Thermoelastic noise
    Deltarm=(Yc85*Trm/rhoc85/Cc85)*(alphac85-betac85*(Lambdarm/Arm/Yc85))**2
    taurm=((Cc85*(2*rwrm)**2)/2.16/2/pi/Kc85)*rhoc85
    phiSTrm=phierm+phic85
    for i in range(len(w)):
        phithTrm[i]=Deltarm*w[i]*taurm/(1+w[i]**2*taurm**2)
    phiTrm=phithTrm+phiSTrm
    phirm0=DFrm*phiTrm
    
    ################Marionette#########################
    Mmario=Susp.Stage[2].Mass   #Mirror mass
    nwmario=Susp.Stage[2].NWires
    Lmario=Susp.Stage[2].Length
    dwmario=Susp.Stage[2].WireDiameter
    alphamaraging=Susp[WireMat2].Alpha # FS expansion coefficient [1/K]
    betamaraging=Susp[WireMat2].dlnEdT# FS 1/E dE/dT [1/K]
    Cmaraging=Susp[WireMat2].C# FS Thermal coefficient [J/(kg K)]
    Kmaraging=Susp[WireMat2].K# FS thermal conduction [W/(m K)]
    Ymaraging=Susp[WireMat2].Y
    rhomaraging=Susp[WireMat2].Rho
    phimaraging=Susp[WireMat2].Phi # FS bulk loss angle
    dsmaraging=Susp[WireMat2].Dissdepth# Surface depth
    Qvhmario=Susp.Stage[2].Qvh#
    Qvvmario=Susp.Stage[2].Qvv#
    rwmario=dwmario/2
    I2mario=pi/4*rwmario**4
    Amario=pi*rwmario**2
    Lambdamario=Mmario*g/nwmario
    omega1g=(g/Lmario)**0.5
    #k1v=Amario*Ymaraging/Lmario
    nu01vmis=0.4
    omega01vmis=2*pi*nu01vmis
    k1v=(Mmirror+Mrm+Mmario)*omega01vmis**2
    omega01v=(k1v/Mmario)**0.5
    phithTmario=np.zeros(len(f),dtype = "complex_")
    phiTmario=np.zeros(len(f),dtype = "complex_")
    phimario0=np.zeros(len(f),dtype = "complex_")
    #Reaction Mass Loss Angle
    DFmario=(1/Lmario)*(nwmario*Ymaraging*I2mario/(Mmario+Mmirror+Mrm)/g)**0.5
    #Dilution factor
    mumario=4/rwmario
    #phie=(muw*hbs*Sw/Vw)*phisurf;  %eccess losses
    phiemario=mumario*dsmaraging*phimaraging
    #Thermoelastic noise
    Deltamario=(Ymaraging*Tmario/rhomaraging/Cmaraging)*(alphamaraging-betamaraging*(Lambdamario/Amario/Ymaraging))**2
    taumario=((Cmaraging*(2*rwmario)**2)/2.16/2/pi/Kmaraging)*rhomaraging
    phiSTmario=phiemario+phimaraging
    for i in range(len(w)):
        phithTmario[i]=Deltamario*w[i]*taumario/(1+w[i]**2*taumario**2)
    
    phiTmario=phithTmario+phiSTmario
    phimario0=DFmario*phiTmario
    #Uncoupled oscillators---------------------------------------
    M01=Mmario
    M02=Mmirror
    M03=Mrm
    Mtot=Mmirror+Mmario+Mrm
    Ts1=Tmario
    Ts2=Tmirror
    Ts3=Trm
    Q01=np.zeros(len(f),dtype = "complex_")
    Q02=np.zeros(len(f),dtype = "complex_")
    Q03=np.zeros(len(f),dtype = "complex_")
    Q01v=np.zeros(len(f),dtype = "complex_")
    Q02v=np.zeros(len(f),dtype = "complex_")
    Q03v=np.zeros(len(f),dtype = "complex_")
    tau01=np.zeros(len(f),dtype = "complex_")
    tau02=np.zeros(len(f),dtype = "complex_")
    tau03=np.zeros(len(f),dtype = "complex_")
    tau01v=np.zeros(len(f),dtype = "complex_")
    tau02v=np.zeros(len(f),dtype = "complex_")
    tau03v=np.zeros(len(f),dtype = "complex_")
    
    mut=M01/Mtot
    omega01=(omega1g**2/mut+omega1g**2*DFmario/mut)**0.5
    omega02=omega2g*(1+DFmirror)**0.5
    omega03=omega3g*(1+DFrm)**0.5

    nu01=omega01/2/pi
    nu02=omega02/2/pi
    nu03=omega03/2/pi

    mu21=M02/M01
    mu31=M03/M01
    
    Qp=np.zeros(len(f),dtype = "complex_")
    
    for i in range(len(w)):
        Qp[i]=(phiP[i]*omega02/w[i])**(-1)
        Q01[i]=mut**0.5*(phimario0[i]*omega01/w[i]+Qvhmario**(-1))**(-1)
        Q02[i]=(phiP[i]*omega02/w[i])**(-1)
        Q03[i]=(phirm0[i]*omega03/w[i]+Qvhrm**(-1))**(-1)
        tau01[i]=Q01[i]/omega01
        tau02[i]=Q02[i]/omega02
        tau03[i]=Q03[i]/omega03
        Q01v[i]=(phimaraging*omega01v/w[i]+Qvvmario**(-1))**(-1)
        Q02v[i]=(phiST*omega02v/w[i])**(-1)
        Q03v[i]=(phic85*omega03v/w[i]+Qvvrm**(-1))**(-1)
        tau01v[i]=Q01v[i]/omega01v
        tau02v[i]=Q02v[i]/omega02v
        tau03v[i]=Q03v[i]/omega03v
    
    #Hor Normal Modes Frequencies
    B3=1
    B2=-(omega01**2+(1+mu21)*omega02**2+(1+mu31)*omega03**2)
    B1=((1+mu21+mu31)*omega03**2*omega02**2+(omega02**2+omega03**2)*omega01**2)
    B0=-(omega01*omega02*omega03)**2

    den2=-2*B2**3+9*B1*B2*B3-27*B0*B3**2
    den3=-B2**2+3*B1*B3

    F1=(4*den3**3+den2**2)**0.5
    F2=(den2+F1)**(1/3)

    omegap=(-B2/3-2**(1/3)*B1/F2+2**(1/3)*B2**2/F2/3+F2/3/2**(1/3))**0.5
    omega0=(-B2/3+B1/F2/2**(2/3)-1j*3**0.5*B1/F2/2**(2/3)-B2**2/F2/2**(2/3)/3+1j*B2**2/F2/2**(2/3)/3**0.5-F2/6/2**(1/3)-1j*F2/2/2**(1/3)/3**0.5)**0.5
    omegam=(-B2/3/B3+B1/F2/2**(2/3)+1j*3**0.5*B1/F2/2**(2/3)-B2**2/F2/2**(2/3)/3-1j*B2**2/F2/2**(2/3)/3**0.5-F2/6/2**(1/3)+1j*F2/2/2**(1/3)/3**0.5)**0.5
    
    omega1=omegam
    omega2=omega0
    omega3=omegap
    #Ver Normal Modes Frequencies
    B3v=1
    B2v=-(omega01v**2+(1+mu21)*omega02v**2+(1+mu31)*omega03v**2)
    B1v=((1+mu21+mu31)*omega03v**2*omega02v**2+(omega02v**2+omega03v**2)*omega01v**2)
    B0v=-(omega01v*omega02v*omega03v)**2

    den2v=-2*B2v**3+9*B1v*B2v*B3v-27*B0v*B3v**2
    den3v=-B2v**2+3*B1v*B3v

    F1v=(4*den3v**3+den2v**2)**0.5
    F2v=(den2v+F1v)**(1/3)
    
    omegapv=(-B2v/3-2**(1/3)*B1v/F2v+2**(1/3)*B2v**2/F2v/3+F2v/3/2**(1/3))**0.5
    omega0v=(-B2v/3+B1v/F2v/2**(2/3)-1j*3**0.5*B1v/F2v/2**(2/3)-B2v**2/F2v/2**(2/3)/3+1j*B2v**2/F2v/2**(2/3)/3**0.5-F2v/6/2**(1/3)-1j*F2v/2/2**(1/3)/3**0.5)**0.5
    omegamv=(-B2v/3/B3v+B1v/F2v/2**(2/3)+1j*3**0.5*B1v/F2v/2**(2/3)-B2v**2/F2v/2**(2/3)/3-1j*B2v**2/F2v/2**(2/3)/3**0.5-F2v/6/2**(1/3)+1j*F2v/2/2**(1/3)/3**0.5)**0.5

    omega1v=omegamv
    omega2v=omega0v
    omega3v=omegapv
    
    #Matrix of the coordinate transformation (Hor)
    lambda11=1
    lambda13=1
    lambda32=1
    lambda12=(omega03**2-omega2**2)/omega03**2
    lambda21=omega02**2/(omega02**2-omega1**2)
    lambda22=((omega01**2+mu21*omega02**2+mu31*omega03**2-omega2**2)/omega02**2/mu21)*(1-(omega2/omega03)**2)-(M03/M02)*(omega03/omega02)**2
    lambda23=omega02**2/(omega02**2-omega3**2)
    lambda31=omega03**2/(omega03**2-omega1**2)
    lambda33=omega03**2/(omega02**2-omega3**2)
    detLambda=lambda13*lambda32*lambda21-lambda11*lambda32*lambda23-lambda13*lambda22*lambda31+lambda12*lambda23*lambda31-lambda12*lambda21*lambda33+lambda11*lambda22*lambda33
    invdet=1/detLambda
    
    #Matrix of the coordinate transformation (Ver)
    lambda11v=1
    lambda13v=1
    lambda32v=1
    lambda12v=(omega03v**2-omega2v**2)/omega03v**2
    lambda21v=omega02v**2/(omega02v**2-omega1v**2)
    lambda22v=((omega01v**2+mu21*omega02v**2+mu31*omega03v**2-omega2v**2)/omega02v**2/mu21)*(1-(omega2v/omega03v)**2)-(M03/M02)*(omega03v/omega02v)**2
    lambda23v=omega02v**2/(omega02v**2-omega3v**2)
    lambda31v=omega03v**2/(omega03v**2-omega1v**2)
    lambda33v=omega03v**2/(omega02v**2-omega3v**2)
    detLambdav=lambda13v*lambda32v*lambda21v-lambda11v*lambda32v*lambda23v-lambda13v*lambda22v*lambda31v+lambda12v*lambda23v*lambda31v-lambda12v*lambda21v*lambda33v+lambda11v*lambda22v*lambda33v
    invdetv=1/detLambdav
    
    #Normal Mode Masses (hor)
    m1=lambda11**2*M01+lambda21**2*M02+lambda31**2*M03
    m2=lambda12**2*M01+lambda22**2*M02+lambda32**2*M03
    m3=lambda13**2*M01+lambda23**2*M02+lambda33**2*M03

    #Normal Mode Masses (ver)
    m1v=lambda11v**2*M01+lambda21v**2*M02+lambda31v**2*M03
    m2v=lambda12v**2*M01+lambda22v**2*M02+lambda32v**2*M03
    m3v=lambda13v**2*M01+lambda23v**2*M02+lambda33v**2*M03
    
    #Matrix N (Hor)
    N11=(invdet*m1/M01)*(lambda22*lambda33-lambda32*lambda23)
    N12=(invdet*m1/M02)*(lambda13*lambda32-lambda12*lambda33)
    N13=(invdet*m1/M03)*(lambda12*lambda23-lambda13*lambda22)
    N21=(invdet*m2/M01)*(lambda23*lambda31-lambda21*lambda33)
    N22=(invdet*m2/M02)*(lambda11*lambda33-lambda13*lambda31)
    N23=(invdet*m2/M03)*(lambda13*lambda21-lambda11*lambda23)
    N31=(invdet*m3/M01)*(lambda32*lambda21-lambda22*lambda31)
    N32=(invdet*m3/M02)*(lambda12*lambda31-lambda11*lambda32)
    N33=(invdet*m3/M03)*(lambda22*lambda11-lambda12*lambda21)
    
    #Matrix N (ver)
    N11v=(invdetv*m1v/M01)*(lambda22v*lambda33v-lambda32v*lambda23v)
    N12v=(invdetv*m1v/M02)*(lambda13v*lambda32v-lambda12v*lambda33v)
    N13v=(invdetv*m1v/M03)*(lambda12v*lambda23v-lambda13v*lambda22v)
    N21v=(invdetv*m2v/M01)*(lambda23v*lambda31v-lambda21v*lambda33v)
    N22v=(invdetv*m2v/M02)*(lambda11v*lambda33v-lambda13v*lambda31v)
    N23v=(invdetv*m2v/M03)*(lambda13v*lambda21v-lambda11v*lambda23v)
    N31v=(invdetv*m3v/M01)*(lambda32v*lambda21v-lambda22v*lambda31v)
    N32v=(invdetv*m3v/M02)*(lambda12v*lambda31v-lambda11v*lambda32v)
    N33v=(invdetv*m3v/M03)*(lambda22v*lambda11v-lambda12v*lambda21v)
    
    
    #Uncoupled Stochastic Thermal Forces (hor)
    Fterm01=np.zeros(len(f),dtype = "complex_")
    Fterm02=np.zeros(len(f),dtype = "complex_")
    Fterm03=np.zeros(len(f),dtype = "complex_")
    Fterm01v=np.zeros(len(f),dtype = "complex_")
    Fterm02v=np.zeros(len(f),dtype = "complex_")
    Fterm03v=np.zeros(len(f),dtype = "complex_")
    
    for i in range(len(w)):
        Fterm01[i]=(4*kb*Ts1*M01/ tau01[i])**0.5
        Fterm02[i]=(4*kb*Ts2*M02/ tau02[i])**0.5
        Fterm03[i]=(4*kb*Ts3*M03/ tau03[i])**0.5

    #Uncoupled Stochastic Thermal Forces (ver)
    for i in range(len(w)):
        Fterm01v[i]=(4*kb*Ts1*M01/ tau01v[i])**0.5
        Fterm02v[i]=(4*kb*Ts2*M02/ tau02v[i])**0.5
        Fterm03v[i]=(4*kb*Ts3*M03/ tau03v[i])**0.5
        
        #Normal Mode Quality Factors
    A1n=np.zeros(len(f),dtype = "complex_")
    A2n=np.zeros(len(f),dtype = "complex_")
    A3n=np.zeros(len(f),dtype = "complex_")
    Hd1=np.zeros(len(f),dtype = "complex_")
    Hd2=np.zeros(len(f),dtype = "complex_")
    Hd3=np.zeros(len(f),dtype = "complex_")
    Q1=np.zeros(len(f),dtype = "complex_")
    Q2=np.zeros(len(f),dtype = "complex_")
    Q3=np.zeros(len(f),dtype = "complex_")
    tau1=np.zeros(len(f),dtype = "complex_")
    tau2=np.zeros(len(f),dtype = "complex_")
    tau3=np.zeros(len(f),dtype = "complex_")
    A1nv=np.zeros(len(f),dtype = "complex_")
    A2nv=np.zeros(len(f),dtype = "complex_")
    A3nv=np.zeros(len(f),dtype = "complex_")
    Hd1v=np.zeros(len(f),dtype = "complex_")
    Hd2v=np.zeros(len(f),dtype = "complex_")
    Hd3v=np.zeros(len(f),dtype = "complex_")
    Q1v=np.zeros(len(f),dtype = "complex_")
    Q2v=np.zeros(len(f),dtype = "complex_")
    Q3v=np.zeros(len(f),dtype = "complex_")
    tau1v=np.zeros(len(f),dtype = "complex_")
    tau2v=np.zeros(len(f),dtype = "complex_")
    tau3v=np.zeros(len(f),dtype = "complex_")
    T1=np.zeros(len(f),dtype = "complex_")
    T2=np.zeros(len(f),dtype = "complex_")
    T3=np.zeros(len(f),dtype = "complex_")
    T1v=np.zeros(len(f),dtype = "complex_")
    T2v=np.zeros(len(f),dtype = "complex_")
    T3v=np.zeros(len(f),dtype = "complex_")
    Tn1=np.zeros(len(f),dtype = "complex_")
    Tn2=np.zeros(len(f),dtype = "complex_")
    Tn3=np.zeros(len(f),dtype = "complex_")
    Tn1v=np.zeros(len(f),dtype = "complex_")
    Tn2v=np.zeros(len(f),dtype = "complex_")
    Tn3v=np.zeros(len(f),dtype = "complex_")
    Sxn1=np.zeros(len(f),dtype = "complex_")
    Sxn2=np.zeros(len(f),dtype = "complex_")
    Sxn3=np.zeros(len(f),dtype = "complex_")
    Sxth_modal=np.zeros(len(f),dtype = "complex_")
    h2xth_modal=np.zeros(len(f),dtype = "complex_")
    hxth_modal=np.zeros(len(f),dtype = "complex_")
    Sxn1v=np.zeros(len(f),dtype = "complex_")
    Sxn2v=np.zeros(len(f),dtype = "complex_")
    Sxn3v=np.zeros(len(f),dtype = "complex_")
    Sxth_modalv=np.zeros(len(f),dtype = "complex_")
    h2xth_modalv=np.zeros(len(f),dtype = "complex_")
    hxth_modalv=np.zeros(len(f),dtype = "complex_")
    NumZtot22inv=np.zeros(len(f),dtype = "complex_")
    DenZtot22inv=np.zeros(len(f),dtype = "complex_")
    Ztot22inv=np.zeros(len(f),dtype = "complex_")
    ReZinv22=np.zeros(len(f),dtype = "complex_")
    Sxth_FDT=np.zeros(len(f),dtype = "complex_")
    hxth_FDT=np.zeros(len(f),dtype = "complex_")
    h2xth_FDT=np.zeros(len(f),dtype = "complex_")
    NumZtot22invv=np.zeros(len(f),dtype = "complex_")
    DenZtot22invv=np.zeros(len(f),dtype = "complex_")
    Ztot22invv=np.zeros(len(f),dtype = "complex_")
    ReZinv22v=np.zeros(len(f),dtype = "complex_")
    Sxth_FDTv=np.zeros(len(f),dtype = "complex_")
    hxth_FDTv=np.zeros(len(f),dtype = "complex_")
    h2xth_FDTv=np.zeros(len(f),dtype = "complex_")
    h2xth_modalTOT=np.zeros(len(f),dtype = "complex_")
    h2xth_FDTTOT=np.zeros(len(f),dtype = "complex_")
    
   
    
    
    for i in range(len(w)):
    #hor
        A1n[i]=omega01*M01/Q01[i]
        A2n[i]=omega02*M02/Q02[i]
        A3n[i]=omega03*M03/Q03[i]
        Hd1[i]=A1n[i]+A2n[i]*(lambda21-1)**2+A3n[i]*(lambda31-1)**2
        Hd2[i]=A1n[i]*lambda12**2+A2n[i]*(lambda12-lambda22)**2+A3n[i]*(lambda12-1)**2
        Hd3[i]=A1n[i]+A2n[i]*(lambda23-1)**2+A3n[i]*(lambda33-1)**2
        Q1[i]=omega1*m1/Hd1[i]
        Q2[i]=omega2*m2/Hd2[i]
        Q3[i]=omega3*m3/Hd3[i]
        tau1[i]=Q1[i]/omega1
        tau2[i]=Q2[i]/omega2
        tau3[i]=Q3[i]/omega3
    #ver
        A1nv[i]=omega01v*M01/Q01v[i]
        A2nv[i]=omega02v*M02/Q02v[i]
        A3nv[i]=omega03v*M03/Q03v[i]
        Hd1v[i]=A1nv[i]+A2nv[i]*(lambda21v-1)**2+A3nv[i]*(lambda31v-1)**2
        Hd2v[i]=A1nv[i]*lambda12v**2+A2nv[i]*(lambda12v-lambda22v)**2+A3nv[i]*(lambda12v-1)**2
        Hd3v[i]=A1nv[i]+A2nv[i]*(lambda23v-1)**2+A3nv[i]*(lambda33v-1)**2
        Q1v[i]=omega1v*m1v/Hd1v[i]
        Q2v[i]=omega2v*m2v/Hd2v[i]
        Q3v[i]=omega3v*m3v/Hd3v[i]
        tau1v[i]=Q1v[i]/omega1v
        tau2v[i]=Q2v[i]/omega2v
        tau3v[i]=Q3v[i]/omega3v
        #Normal Transfer Functions (hor)
        T1[i]=(1/m1)*(-(w[i]**2-omega1**2)+1j*w[i]/tau1[i])**(-1)
        T2[i]=(1/m2)*(-(w[i]**2-omega2**2)+1j*w[i]/tau2[i])**(-1)
        T3[i]=(1/m3)*(-(w[i]**2-omega3**2)+1j*w[i]/tau3[i])**(-1)
        #Normal Transfer Functions (ver)
        T1v[i]=(1/m1v)*(-(w[i]**2-omega1v**2)+1j*w[i]/tau1v[i])**(-1)
        T2v[i]=(1/m2v)*(-(w[i]**2-omega2v**2)+1j*w[i]/tau2v[i])**(-1)
        T3v[i]=(1/m3v)*(-(w[i]**2-omega3v**2)+1j*w[i]/tau3v[i])**(-1)
        #Transfer functions (hor)
        Tn1[i]=(N11*T1[i]*lambda21+N21*T2[i]*lambda22+N31*T3[i]*lambda23)
        Tn2[i]=(N12-N11)*T1[i]*lambda21+(N22-N21)*T2[i]*lambda22+(N32-N31)*T3[i]*lambda23
        Tn3[i]=(N13-N11)*T1[i]*lambda21+(N23-N21)*T2[i]*lambda22+(N33-N31)*T3[i]*lambda23
        #Transfer functions (ver)
        Tn1v[i]=(N11v*T1v[i]*lambda21v+N21v*T2v[i]*lambda22v+N31v*T3v[i]*lambda23v)
        Tn2v[i]=(N12v-N11v)*T1v[i]*lambda21v+(N22v-N21v)*T2v[i]*lambda22v+(N32v-N31v)*T3v[i]*lambda23v
        Tn3v[i]=(N13v-N11v)*T1v[i]*lambda21v+(N23v-N21v)*T2v[i]*lambda22v+(N33v-N31v)*T3v[i]*lambda23v
        #Thermal Noise with Modal Threatment (hor)
        Sxn1[i]=Fterm01[i]**2*Tn1[i]*np.conj(Tn1[i])
        Sxn2[i]=Fterm02[i]**2*Tn2[i]*np.conj(Tn2[i])
        Sxn3[i]=Fterm03[i]**2*Tn3[i]*np.conj(Tn3[i])
        Sxth_modal[i]=Sxn1[i]+Sxn2[i]+Sxn3[i]
        h2xth_modal[i]=(2/Lharm)**2*Sxth_modal[i]
        hxth_modal[i]=(2/Lharm)*Sxth_modal[i]**0.5
        #Thermal Noise with Modal Threatment (ver)
        Sxn1v[i]=Fterm01v[i]**2*Tn1v[i]*np.conj(Tn1v[i])
        Sxn2v[i]=Fterm02v[i]**2*Tn2v[i]*np.conj(Tn2v[i])
        Sxn3v[i]=Fterm03v[i]**2*Tn3v[i]*np.conj(Tn3v[i])
        Sxth_modalv[i]=Sxn1v[i]+Sxn2v[i]+Sxn3v[i]
        h2xth_modalv[i]=(2/Lharm)**2*Sxth_modalv[i]
        hxth_modalv[i]=(2/Lharm)*Sxth_modalv[i]**0.5
      
#Thermal Noise with FDT Threatment  (hor)
        NumZtot22inv[i]=(1j*w[i]*(-(-w[i]**2+omega01**2+mu21*omega02**2+mu31*omega03**2+1j*w[i]*(1/tau01[i]+mu21/tau02[i]+mu31/tau03[i]))*(-w[i]**2+omega03**2+1j*w[i]/tau03[i])+mu31*(1j*w[i]+omega03**2*tau03[i])**2/tau03[i]**2))
        DenZtot22inv[i]=(M02*((mu21*(1j*w[i]+omega02**2*tau02[i])**2/tau02[i]**2-(-w[i]**2+omega02**2+1j*w[i]/tau02[i])*(-w[i]**2+omega01**2+mu21*omega02**2+mu31*omega03**2+1j*w[i]*(1/tau01[i]+mu21/tau02[i]+mu31/tau03[i])))*(-w[i]**2+omega03**2+1j*w[i]/tau03[i])+(mu31*(-1j*w[i]+(w[i]**2-omega02**2)*tau02[i])*(w[i]-1j*omega03**2*tau03[i])**2)/tau02[i]/tau03[i]**2))
        Ztot22inv[i]=NumZtot22inv[i]/DenZtot22inv[i]
        ReZinv22[i]=np.real(Ztot22inv[i])
        Sxth_FDT[i]=(4*kb*T/w[i]**2)*ReZinv22[i]
        hxth_FDT[i]=(2/Lharm)*Sxth_FDT[i]**0.5
        h2xth_FDT[i]=(2/Lharm)**2*Sxth_FDT[i]
    
#Thermal Noise with FDT Threatment  (ver)
        NumZtot22invv[i]=(1j*w[i]*(-(-w[i]**2+omega01v**2+mu21*omega02v**2+mu31*omega03v**2+1j*w[i]*(1/tau01v[i]+mu21/tau02v[i]+mu31/tau03v[i]))*(-w[i]**2+omega03v**2+1j*w[i]/tau03v[i])+mu31*(1j*w[i]+omega03v**2*tau03v[i])**2/tau03v[i]**2))
        DenZtot22invv[i]=(M02*((mu21*(1j*w[i]+omega02v**2*tau02v[i])**2/tau02v[i]**2-(-w[i]**2+omega02v**2+1j*w[i]/tau02v[i])*(-w[i]**2+omega01v**2+mu21*omega02v**2+mu31*omega03v**2+1j*w[i]*(1/tau01v[i]+mu21/tau02v[i]+mu31/tau03v[i])))*(-w[i]**2+omega03v**2+1j*w[i]/tau03v[i])+(mu31*(-1j*w[i]+(w[i]**2-omega02v**2)*tau02v[i])*(w[i]-1j*omega03v**2*tau03v[i])**2)/tau02v[i]/tau03v[i]**2))
        Ztot22invv[i]=NumZtot22invv[i]/DenZtot22invv[i]
        ReZinv22v[i]=np.real(Ztot22invv[i])
        Sxth_FDTv[i]=(4*kb*T/w[i]**2)*ReZinv22v[i]
        hxth_FDTv[i]=(2/Lharm)*Sxth_FDTv[i]**0.5
        h2xth_FDTv[i]=(2/Lharm)**2*Sxth_FDTv[i]
    
#Total Thermal Noise
    h2xth_modalTOT=(h2xth_modal+thetaHV**2*h2xth_modalv)*Lharm**2
    h2xth_FDTTOT=(h2xth_FDT+thetaHV**2*h2xth_FDTv)*Lharm**2

    return h2xth_modalTOT, h2xth_FDTTOT
    
    
def STNViol(f,Susp,ifo):

    
    #-------------Physical parameters----------------------------------
    Tmirror=Susp.Stage[0].Temp
    Trm=Susp.Stage[1].Temp
    Tmario=Susp.Stage[2].Temp
    g=const.g    #gravitational acceleration (m/s^2)
    kb=const.kB  #Boltzmann constant (J/K)
    w=2*pi*f
    Lharm=ifo.Infrastructure.Length
    Mmirror=Susp.Stage[0].Mass   #Mirror mass
    nw=Susp.Stage[0].NWires
    Lmirror=Susp.Stage[0].Length
    dwmirror=ifo.Suspension.Stage[0].WireDiameter
    
    WireMat0=Susp.Stage[0].WireMaterial
    WireMat1=Susp.Stage[1].WireMaterial
    WireMat2=Susp.Stage[2].WireMaterial
    
    alphamir=Susp[WireMat0].Alpha # FS expansion coefficient [1/K]
    betamir=Susp[WireMat0].dlnEdT# FS 1/E dE/dT [1/K]
    Cmir=Susp[WireMat0].C# FS Thermal coefficient [J/(kg K)]
    Kmir=Susp[WireMat0].K# FS thermal conduction [W/(m K)]
    Ymir=Susp[WireMat0].Y#
    rhomir=Susp[WireMat0].Rho#
    phimir=Susp[WireMat0].Phi# FS bulk loss angle
    ds=Susp[WireMat0].Dissdepth# Surface depth
    Qvhmirror=Susp.Stage[0].Qvh
    
    rwmirror=dwmirror/2
    I2mirror=pi/4*rwmirror**4
    Amirror=pi*rwmirror**2
    Lambda=Mmirror*g/nw
    omega2g=(g/Lmirror)**0.5
    k2v=nw*Amirror*Ymir/Lmirror
    omega02v=(k2v/Mmirror)**0.5
    thetaHV=Susp.VHCoupling.theta
    GammaMir=Lambda
    w=2*pi*f
    nmodi=100
    ke=(Lambda/I2mirror/Ymir)**0.5
    
    phithT=np.zeros(len(f),dtype = "complex_")
    phiT=np.zeros(len(f),dtype = "complex_")
    phiP=np.zeros(len(f),dtype = "complex_")
    omegaviol=np.zeros(nmodi,dtype = "complex_")
    iviol=np.zeros(len(f),dtype = "complex_")
    phiviol=np.zeros(nmodi,dtype = "complex_")
    Trasf=np.zeros([nmodi,len(f)],dtype = "complex_")
    Somma=np.zeros(len(f),dtype = "complex_")
    
    
    #Mirror Pendulum Loss Angle
    TensMirror=Mmirror*g/nw
    DFmirror=(1/Lmirror)*(nw*Ymir*I2mirror/Mmirror/g)**0.5;    #Dilution factor
    mu=4/rwmirror
    #phie=(muw*hbs*Sw/Vw)*phisurf;  %eccess losses
    phie=mu*ds*phimir
    #Thermoelastic noise
    Delta=(Ymir*Tmirror/rhomir/Cmir)*(alphamir-betamir*(Lambda/Amirror/Ymir))**2;
    tau=((Cmir*(2*rwmirror)**2)/2.16/2/pi/Kmir)*rhomir
    phiST=phie+phimir
    for i in range(len(w)):
        phithT[i]=Delta*w[i]*tau/(1+w[i]**2*tau**2)
        
    phiT=phithT+phiST
    phiP=DFmirror*phiT
    
    phitheT=np.zeros(nmodi,dtype = "complex_")
    phiTot=np.zeros(nmodi,dtype = "complex_")
    ##Mode frequencies and losses
    for n in range(nmodi):
        omegaviol[n]=2*pi*((n+1)/2/Lmirror)*(GammaMir/Amirror/rhomir)**0.5*(1+(2/Lmirror)*(Ymir*I2mirror/GammaMir)**0.5)
        phitheT[n]=Delta*omegaviol[n]*tau/(1+omegaviol[n]**2*tau**2)
        phiTot[n]=phiST+phitheT[n]
        phiviol[n]=(2*phiTot[n]/ke/Lmirror)*(1+(((n+1)*pi)**2/2/Lmirror/ke))
        
    viol1=omegaviol[0]/2/pi
    viol2=omegaviol[1]/2/pi
    viol3=omegaviol[2]/2/pi
    phiviol1=phiviol[0]
    phiviol2=phiviol[1]
    phiviol3=phiviol[2]
    
    Factor=4*4*kb*Tmirror*2*rhomir*rwmirror**2*Lmirror/pi/Mmirror**2
    
    noise=np.zeros(len(w),dtype = "complex_")
    for i in range(len(w)):
        Somma[i]=0
        for n in range(nmodi):
            Trasf[n,i]=(1/(n+1)**2)*(phiviol[n]*omegaviol[n]**2)/((omegaviol[n]**2-w[i]**2)**2+(omegaviol[n]**2*phiviol[n])**2)
            Somma[i]=Somma[i]+Trasf[n,i]
        noise[i]=((2)**2)*(Factor*Somma[i]/w[i])
            
    return noise 
        



