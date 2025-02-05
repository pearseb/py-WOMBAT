import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot1D(no3,dfe,phy,zoo,det,phymu,zoomu,chlc,phyfec,zoofec,detfec,phync,limnit,limdfe,dic,Grid):
    
    col1 = 'k'
    col2 = 'firebrick'
    col3 = 'goldenrod'
    
    lab1 = 'Nitrate (µM)'
    lab2 = 'dissolved Fe (nM)'
    lab3 = 'Phytoplankton (µM C)'
    lab4 = 'Zooplankton (µM C)'
    lab5 = 'Detritus (µM C)'
    #lab6 = 'CaCO$_3$ (µM)'
    lab7 = '$\mu$(Phy) (day$^{-1}$)'
    lab8 = '$\mu$(Zoo) (day$^{-1}$)'
    lab9 ='Chl:C'
    lab10 ='Phy Fe:C (µmol mol$^{-1}$)'
    lab11 ='Zoo Fe:C (µmol mol$^{-1}$)'
    lab12 ='Det Fe:C (µmol mol$^{-1}$)'
    lab13 ='DIC (µM)'
    lab14 ='Phy N limitation'
    lab15 ='Phy dFe limitation'
    
    fig = plt.figure(figsize=(14,12))
    gs = GridSpec(3,3)
    
    ax1 = plt.subplot(gs[0,0]) 
    ax2 = plt.subplot(gs[0,1]) 
    ax3 = plt.subplot(gs[0,2]) 
    ax4 = plt.subplot(gs[1,0]) 
    ax5 = plt.subplot(gs[1,1]) 
    ax6 = plt.subplot(gs[1,2]) 
    ax7 = plt.subplot(gs[2,0]) 
    ax8 = plt.subplot(gs[2,1]) 
    
    ax1.plot(no3, Grid.zgrid, color=col1, label=lab1)
    ax1.plot(dfe*1e3, Grid.zgrid, color=col2, label=lab2)
    ax1.legend()
    
    ax2.plot(phy, Grid.zgrid, color=col1, label=lab3)
    ax2.plot(zoo, Grid.zgrid, color=col2, label=lab4)
    ax2.plot(chlc*12*phy, Grid.zgrid, color=col3, label='Chlorophyll (mg m$^{-3}$)')
    ax2.legend()
    
    ax3.plot(det, Grid.zgrid, color=col1, label=lab5)
    #ax3.plot(cal, Grid.zgrid, color=col2, label=lab6)
    ax3.legend(loc='lower left')
    
    ax4.plot(phymu*86400, Grid.zgrid, color=col1, label=lab7)
    ax4.plot(zoomu*86400, Grid.zgrid, color=col2, label=lab8)
    ax4.legend()
    
    ax5.plot(chlc, Grid.zgrid, color=col1, label=lab9)
    ax5.legend()
    
    ax6.plot(phyfec, Grid.zgrid, color=col1, label=lab10)
    ax6.plot(zoofec, Grid.zgrid, color=col2, label=lab11)
    ax6.plot(detfec, Grid.zgrid, color=col3, label=lab12)
    ax6.legend()
    
    ax7.plot(dic, Grid.zgrid, color=col1, label=lab13)
    ax7.legend()
    
    ax8.plot(limnit, Grid.zgrid, color=col1, label=lab14)
    ax8.plot(limdfe, Grid.zgrid, color=col2, label=lab15)
    ax8.legend()
    
    ax1.set_ylim(-500,0)
    ax2.set_ylim(-500,0)
    ax3.set_ylim(-500,0)
    ax4.set_ylim(-500,0)
    ax5.set_ylim(-500,0)
    ax6.set_ylim(-500,0)
    ax6.set_xlim(0,200)
    ax7.set_ylim(-500,0)
    ax8.set_ylim(-500,0)
    
    return fig

