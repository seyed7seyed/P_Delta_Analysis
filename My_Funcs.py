# (auto) importing modules needed
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from sympy import *
import copy

# (auto) multi-sectioning
def MS( wBay, Supports, L_tinyBeam ):
    
    nBeams_Main = len(wBay)       # number of Main Beams  eg: 2
    #nSupports = nBeams_Main + 1
    wBay_Main   = wBay.copy()     # width of Main Bays    eg: 500, 600

    WBAY, SUPPORTS, NPIECES = [],[],[]
    s = 0
    SUPPORTS = [ Supports[0] ]
    for i in wBay_Main:
        nPieces  = int(i/L_tinyBeam)
        WBAY     =   WBAY   + [L_tinyBeam]*nPieces
    
        if i%L_tinyBeam >= L_tinyBeam/2: 
            WBAY = WBAY + [i%L_tinyBeam]
            nPieces = nPieces+1
        else                           : 
            WBAY[-1] = WBAY[-1]+i%L_tinyBeam 
    
        NPIECES.append( nPieces )
        s = s+1
        SUPPORTS = SUPPORTS + ['mid']*(nPieces-1) + [Supports[s]]

    SUPPORTS_IDX = [0]
    for j in NPIECES:  SUPPORTS_IDX = SUPPORTS_IDX + [ SUPPORTS_IDX[-1] + j ]

    # Assigning
    Supports     = SUPPORTS       # Type of all Supports                   eg: [ 'pinned', 'mid', 'roller', 'mid', 'fixed' ]
    Supports_Idx = SUPPORTS_IDX   # Index of Main Supports                 eg: [    0,               2,               4    ]
    wBay         = WBAY           # length of all bays (pieces)            eg: [         250 , 250    ,    300 , 300       ]
    nPieces      = NPIECES        # number of bays (pieces) in each beam   eg: [             2        ,        2           ]
    nBay         = len(WBAY)      # number of all bays (pieces)            eg: 4
    
    return [ nBeams_Main, wBay_Main, Supports, Supports_Idx, wBay, nPieces, nBay ]


# (auto) Points & Connectivity Matrices
def PC( wBay ): 
    
    # Points
    Xwalk = [0]
    for i in wBay:
        w = Xwalk[-1] + i
        Xwalk.append(w)
    Ywalk = [0] 
    X=len(Xwalk)
    Y=len(Ywalk) 
    x,y, index = [],[],[]
    idx=0
    for j in range(0,Y):
        for i in range(0,X):
            x.append( int(Xwalk[i]) )
            y.append( int(Ywalk[j]) )
            index.append(idx)
            idx = idx+1
    Dict = {  'x': x
            , 'y': y  }
    Points = pd.DataFrame( data=Dict, index=index )
    
    # Connectivity
    x,y, index = [],[],[]
    idx=0
    for j in range(0,Y):
        for i in range(0,X-1): 
            p0 = i + j*X
            p1 = p0 + 1
            x.append( int(p0) )
            y.append( int(p1) )
            index.append(idx)
            idx = idx+1
    Dict = {  'from_point': x
            , 'to_point': y
           }
    CnB = pd.DataFrame( data=Dict, index=index )
    
    return [ Points, CnB ]



# define a function called Shape
# to check shape of Beam
def  Beam_Shape( Points, CnB, wBay_Main, wBay, Supports ):
    
    fig, ax = plt.subplots( figsize=(12,3) )
        
    for i in CnB.index: 
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = Points.loc[i0,'y']
        y1 = Points.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=4, c='dodgerblue', ls='-' )

    for i in Points.index:
        x = Points.loc[i,'x']
        y = Points.loc[i,'y']
        if   Supports[i]=='fixed' : marker, ec, fc, y = 's', 'r', 'deeppink', y-0.0
        elif Supports[i]=='pinned': marker, ec, fc, y = '^', 'g', 'lime'    , y-0.1
        elif Supports[i]=='roller': marker, ec, fc, y = 'o', 'b', 'cyan'    , y-0.1
        elif Supports[i]=='free'  : marker, ec, fc, y = 'x', 'b', 'b'       , y-0.0
        elif Supports[i]=='mid'   : marker, ec, fc, y = '' , 'b', 'b'       , y-0.0
        ax.scatter( x, y, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        if   Supports[i]=='fixed' :
            ax.scatter( x, y+0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
            ax.scatter( x, y-0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        
    # title
    plt.title( 'Beam Shape', font='MV BOLI', fontsize=16 )
    
    # xticks
    Xwalk = [0]
    for i in wBay_Main:
        w = Xwalk[-1] + i 
        Xwalk.append(w)
    xtks = np.array(Xwalk,dtype='f8')
    ax.set_xticks( xtks )
    plt.xticks(font='MV BOLI',rotation=45,fontsize=14) 
    
    # yticks
    Ywalk = [0]
    ytks = np.array(Ywalk,dtype='f8')
    ax.set_yticks( ytks )
    plt.yticks(font='MV BOLI',rotation=0,fontsize=14)
    
    # ylim
    plt.ylim([-1,+1])
    
    # grid
    plt.grid('on',alpha=.1)
    
    # save image
    plt.savefig('Beam_Shape.png', dpi=120) 
    plt.show()
    
    return None



# (auto) Finding Static Coefficients

# define a function called Static_Coeff 
# to find Static Analysis Coefficients: 
#
#   NOP:    Number Of Points
#   NOD:    Number of "Degrees of freedom"
#   RD:     Restrained "Degrees of freedom"
#   FD:     Free "Degrees of freedom"
#   NFD:    Number of Free "Degrees of Freedom"
#   NB:     Number of Beams
#   IndxB:  ['near_x','near_y','near_z','far_x','far_y','far_z']  of  all beams

def Static_Coeff( Points, CnB, wBay, Supports ):
    
    # NOP NOD RD FD NFD
    NOP=Points.shape[0]
    NOD=NOP*3
    
    RD=[]
    for i in Points.index:
        if   Supports[i]=='fixed' : LIST=[ 3*i+0, 3*i+1, 3*i+2 ]
        elif Supports[i]=='pinned': LIST=[ 3*i+0, 3*i+1        ]
        elif Supports[i]=='roller': LIST=[        3*i+1        ]
        elif Supports[i]=='free'  : LIST=[                     ]
        elif Supports[i]=='mid'   : LIST=[                     ]
        for i in LIST:
            RD.append(i)
    
    FD=[]
    for i in range(NOD):
        if i not in RD:
            FD.append(i)
    
    NRD=len(RD)
    NFD=len(FD)
    
    # NX
    NB = CnB.shape[0]
    
    # IndxB
    c0,c1,c2, c3,c4,c5 = [],[],[], [],[],[]
    index, idx = [], 0  
    for i in range( 0, NB ):
        c0.append( 3*CnB.iloc[i,0]+0 )
        c1.append( 3*CnB.iloc[i,0]+1 )
        c2.append( 3*CnB.iloc[i,0]+2 )
        c3.append( 3*CnB.iloc[i,1]+0 )
        c4.append( 3*CnB.iloc[i,1]+1 )
        c5.append( 3*CnB.iloc[i,1]+2 )
        index.append(idx)
        idx = idx+1 
    Dict = {  'near_x':c0  ,  'near_y':c1  ,  'near_z' : c2
            , 'far_x' :c3  ,  'far_y' :c4  ,  'far_z'  : c5
        }
    IndxB = pd.DataFrame( data=Dict, index=index )
    
    return [NOP, NOD, RD, FD, NFD, NB, IndxB]


# (auto) multi-applying  ---  applying  E, A, I, W  to all pieces
def Multi_Beams(  E_Beams, Sections, W_Beams, nBeams_Main, nPieces ):
    A_Beams, I_Beams = [],[];
    for i in range(len(Sections)):   
        A_Beams.append( int(Sections[i][0]) )
        I_Beams.append( int(Sections[i][1]) )
    
    E_BEAMS, A_BEAMS, I_BEAMS, W_BEAMS = [],[],[],[]
    for i in range(nBeams_Main):
        E_BEAMS  =  E_BEAMS  +  [ E_Beams[i] ] * nPieces[i]
        A_BEAMS  =  A_BEAMS  +  [ A_Beams[i] ] * nPieces[i]
        I_BEAMS  =  I_BEAMS  +  [ I_Beams[i] ] * nPieces[i]
        W_BEAMS  =  W_BEAMS  +  [ W_Beams[i] ] * nPieces[i]

    #E_Beams = E_BEAMS
    #A_Beams = A_BEAMS
    #I_Beams = I_BEAMS
    #W_Beams = W_BEAMS
    
    Beams = pd.DataFrame(columns=['E','A','I','W'])
    Beams['E'], Beams['A'], Beams['I'], Beams['W'] = E_BEAMS, A_BEAMS, I_BEAMS, W_BEAMS
    
    return Beams


def Beam_Color( Beams ):
    
    Colors = ['dodgerblue','deeppink','lawngreen','orange','purple','olive','pink']
    df = Beams[['E','A','I']].copy()

    df_Set = df.copy().drop_duplicates()
    df_Set.index = range( df_Set.shape[0] )
    df_Set['C'] = Colors[:df_Set.shape[0]]

    C = []
    for j in range(df.shape[0]):
        A = df.loc[j,'A']
        I = df.loc[j,'I']
        for i in range(df_Set.shape[0]):
            set_A, set_I, set_C = df_Set.loc[i,['A','I','C']]
            if set_A==A and set_I==I: C.append(set_C)
                
    return C


# define a function called Plot_Sections
# to check sections of beam
def Plot_Sections( Points, CnB, wBay_Main, wBay, Supports, Beams ):
    
    fig, ax = plt.subplots( figsize=(12,3) )
        
    for i in CnB.index: 
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = Points.loc[i0,'y']
        y1 = Points.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        c = Beams.loc[i,'C']
        ax.plot( x, y, lw=4, c=c, ls='-', alpha=0.85 )

    for i in Points.index:
        x = Points.loc[i,'x']
        y = Points.loc[i,'y']
        if   Supports[i]=='fixed' : marker, ec, fc, y = 's', 'lightgray', 'lightgray', y-0.0
        elif Supports[i]=='pinned': marker, ec, fc, y = '^', 'lightgray', 'lightgray'    , y-0.1
        elif Supports[i]=='roller': marker, ec, fc, y = 'o', 'lightgray', 'lightgray'    , y-0.1
        elif Supports[i]=='free'  : marker, ec, fc, y = 'x', 'lightgray', 'lightgray'       , y-0.0
        elif Supports[i]=='mid'   : marker, ec, fc, y = '' , 'lightgray', 'lightgray'       , y-0.0
        ax.scatter( x, y, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        if   Supports[i]=='fixed' :
            ax.scatter( x, y+0.1, marker=marker, s=150, ec='lightgray', fc='lightgray', alpha=0.75 )
            ax.scatter( x, y-0.1, marker=marker, s=150, ec='lightgray', fc='lightgray', alpha=0.75 )
        
    # title
    plt.title( 'Beam Sections', font='MV BOLI', fontsize=16 )
    
    # xticks
    Xwalk = [0]
    for i in wBay_Main:
        w = Xwalk[-1] + i 
        Xwalk.append(w)
    xtks = np.array(Xwalk,dtype='f8')
    ax.set_xticks( xtks )
    plt.xticks(font='MV BOLI',rotation=45,fontsize=14) 
    
    # yticks
    Ywalk = [0]
    ytks = np.array(Ywalk,dtype='f8')
    ax.set_yticks( ytks )
    plt.yticks(font='MV BOLI',rotation=0,fontsize=14)
    
    # ylim
    plt.ylim([-1,+1])
    
    # grid
    plt.grid('on',alpha=.1)
    
    # save image
    plt.savefig('Beam_Sections.png', dpi=120)
    plt.show()
    
    
# define a function called Beam_Load_Shape
# to check load of beams

def  Beam_Dist_Load_Shape( Points, CnB, wBay_Main, wBay, Supports, Beams, W_Beams ):
    
    fig, ax = plt.subplots( figsize=(12,3) )
        
    W = Beams['W'].copy()
    W = 0.5*W / W.max()  
    for i in CnB.index: 
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = Points.loc[i0,'y']
        y1 = Points.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        y2 = [ y0+W[i], y1+W[i] ]
        ax.fill_between(x, y, y2, color='cyan')
        ax.plot( x, y, lw=4, c='dodgerblue', ls='-' )

    for i in Points.index:
        x = Points.loc[i,'x']
        y = Points.loc[i,'y']
        if   Supports[i]=='fixed' : marker, ec, fc, y = 's', 'lightgray', 'lightgray', y-0.0
        elif Supports[i]=='pinned': marker, ec, fc, y = '^', 'lightgray', 'lightgray', y-0.1
        elif Supports[i]=='roller': marker, ec, fc, y = 'o', 'lightgray', 'lightgray', y-0.1
        elif Supports[i]=='free'  : marker, ec, fc, y = 'x', 'lightgray', 'lightgray', y-0.0
        elif Supports[i]=='mid'   : marker, ec, fc, y = '' , 'lightgray', 'lightgray', y-0.0
        ax.scatter( x, y, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        if   Supports[i]=='fixed' :
            ax.scatter( x, y+0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
            ax.scatter( x, y-0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
            
    for i in range(len(wBay_Main)):
        x = sum(wBay_Main[:i]) + wBay_Main[i]/2 -sum(wBay)/20
        y = 0.1 + 0.5*W_Beams[i]/max(W_Beams)
        plt.text( x, y, str(W_Beams[i]), font='MV BOLI', fontsize=14 )
        
    # title
    plt.title( 'Distributed Loads', ha='center', va='center', ma='center', font='MV BOLI', fontsize=16 )
    
    # xticks
    Xwalk = [0]
    for i in wBay_Main:
        w = Xwalk[-1] + i 
        Xwalk.append(w)
    xtks = np.array(Xwalk,dtype='f8')
    ax.set_xticks( xtks )
    plt.xticks(font='MV BOLI',rotation=45,fontsize=14) 
    
    # yticks
    Ywalk = [0]
    ytks = np.array(Ywalk,dtype='f8')
    ax.set_yticks( ytks )
    plt.yticks(font='MV BOLI',rotation=0,fontsize=14)
    
    # ylim
    plt.ylim([-1,+1])
    
    # grid
    plt.grid('on',alpha=.1)
    
    # save image
    plt.savefig('Distributed_Loads.png', dpi=120)
    plt.show()
    
    
# (auto) Nodal Forces
def Nodal_Forces( Points,  Point_Loads,  NOD, nBeams_Main, nPieces ): 
    
    # Multi-Sectioning
    PL = [0]
    for i in range(nBeams_Main):
        PL[-1] = Point_Loads[i]
        PL = PL + [0]*nPieces[i]
    PL[-1] = Point_Loads[-1]
    Point_Loads = PL
    
    NF = pd.DataFrame( columns=['Load','Dummy'], index=range(NOD) ).replace( float('nan'), 0.0 )
    for  i in Points.index:  NF.loc[3*i+1,'Load'] = -Point_Loads[i]
    #for i in Points.index:  NF.loc[3*i  ,'Load'] = +0.001 # disturbing load
    return NF


# checking NF of beams
def NF_Shape( Points, CnB, wBay_Main, wBay, Supports, NF ):
    
    fig, ax = plt.subplots( figsize=(12,3) )
        
    for i in CnB.index: 
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = Points.loc[i0,'y']
        y1 = Points.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=4, c='dodgerblue', ls='-' )
    
    y_Max = max(abs(NF['Load']))
    w = sum(wBay)/85
    for i in NF.index: 
        if  i % 3 == 1:
            x = Points.loc[ (i-1)/3, 'x' ]
            y = NF.loc[i,'Load'] / y_Max *.75
            if   y>0:  
                ax.arrow( x, -y, 0, y, head_width=.15, width=w, fc='pink', ec='deeppink'   )
                plt.text( x-1.5*w, -y-.20, str(int(+NF.loc[i,'Load'])), font='MV BOLI', fontsize=14 )
            elif y<0: 
                ax.arrow( x, -y, 0, y, head_width=.15, width=w, fc='cyan', ec='dodgerblue' )  
                plt.text( x-1.5*w, -y+.05, str(int(-NF.loc[i,'Load'])), font='MV BOLI', fontsize=14 )  
            

    for i in Points.index:
        x = Points.loc[i,'x']
        y = Points.loc[i,'y']
        if   Supports[i]=='fixed' : marker, ec, fc, y = 's', 'lightgray', 'lightgray', y-0.0
        elif Supports[i]=='pinned': marker, ec, fc, y = '^', 'lightgray', 'lightgray', y-0.1
        elif Supports[i]=='roller': marker, ec, fc, y = 'o', 'lightgray', 'lightgray', y-0.1
        elif Supports[i]=='free'  : marker, ec, fc, y = 'x', 'lightgray', 'lightgray', y-0.0
        elif Supports[i]=='mid'   : marker, ec, fc, y = '' , 'lightgray', 'lightgray', y-0.0 
        ax.scatter( x, y, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        if   Supports[i]=='fixed' :
            ax.scatter( x, y+0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
            ax.scatter( x, y-0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        
    # title
    plt.title( 'NF', font='MV BOLI', fontsize=16 )
    
    # xticks
    Xwalk = [0]
    for i in wBay_Main:
        w = Xwalk[-1] + i 
        Xwalk.append(w)
    xtks = np.array(Xwalk,dtype='f8')
    ax.set_xticks( xtks )
    plt.xticks(font='MV BOLI',rotation=45,fontsize=14) 
    
    # yticks
    Ywalk = [0]
    ytks = np.array(Ywalk,dtype='f8')
    ax.set_yticks( ytks )
    plt.yticks(font='MV BOLI',rotation=0,fontsize=14)
    
    # ylim
    plt.ylim([-1,+1])
    
    # grid
    plt.grid('on',alpha=.1)
    
    # save image
    plt.savefig('NF.png', dpi=120)
    plt.show()
    
    return None


# (auto) Elastic Stiffness Matrices
def ke_frame( A, E, I, L ):
    EI, EA, L2, L3 = E*I, E*A, L**2, L**3
    r = [  'c1', 'c2', 'c3', 'c4', 'c5', 'c6'  ] 
    c = {  'c1': [ EA/L   , 0        ,  0      , -1*EA/L,  0       ,  0       ]
         , 'c2': [ 0      , 12*EI/L3 ,  6*EI/L2,  0     , -12*EI/L3,  6*EI/L2 ]
         , 'c3': [ 0      , 6*EI/L2  ,  4*EI/L ,  0     , -6*EI/L2 ,  2*EI/L  ]
         , 'c4': [-1*EA/L , 0        ,  0      ,  EA/L  ,  0       ,  0       ]
         , 'c5': [ 0      , -12*EI/L3, -6*EI/L2,  0     ,  12*EI/L3, -6*EI/L2 ]  
         , 'c6': [ 0      , 6*EI/L2  , 2*EI/L  ,  0     , -6*EI/L2 ,  4*EI/L  ] 
         }  
    df = pd.DataFrame( data=c , index=r )
    return df

def T_frame( c, s ): 
    r = [  'c1', 'c2', 'c3', 'c4', 'c5', 'c6' ] 
    c = {  'c1': [  c  ,  s  ,   0  ,  0  ,  0  ,   0  ]  
         , 'c2': [ -s  ,  c  ,   0  ,  0  ,  0  ,   0  ]
         , 'c3': [  0  ,  0  ,   1  ,  0  ,  0  ,   0  ]
         , 'c4': [  0  ,  0  ,   0  ,  c  ,  s  ,   0  ]
         , 'c5': [  0  ,  0  ,   0  , -s  ,  c  ,   0  ]
         , 'c6': [  0  ,  0  ,   0  ,  0  ,  0  ,   1  ]
         }  
    df = pd.DataFrame( data=c , index=r )
    return df.T


# (auto) defining a function called KeQf
# to form:
#    Elastic Stiffness Matrices, Ke, for all elements
#    External Distributed Loads Matrix, Qf
def KeQf(  Points,  NOD
         , CnB, NB, IndxB, Beams ):
    
    # Empty Ke Qf
    KE = np.zeros([NOD,NOD],'float64')
    KE = pd.DataFrame( KE,columns = np.arange(0,NOD) )
    QF = np.zeros([NOD,2],'float64')
    QF = pd.DataFrame( QF,columns = ['Load','Dummy'] )
    
    # Assigning Properties  
    LB,  TB,  keB,  KeB, qfB, QfB  =  [],[],[],[],[],[]
        
    for i in range(0,NB): 
        dx = Points.iloc[CnB.iloc[i,1],0]-Points.iloc[CnB.iloc[i,0],0]
        dy = Points.iloc[CnB.iloc[i,1],1]-Points.iloc[CnB.iloc[i,0],1]
        L  = (dx**2+dy**2)**0.5;     LB.append(L)
        E  = Beams.loc[i,'E'];      
        A  = Beams.loc[i,'A'];         
        I  = Beams.loc[i,'I'];          
        w  = Beams.loc[i,'W'];          
        c,s= dx/L, dy/L;
        T  = T_frame(c,s);           TB.append(T)
        Indx = list(IndxB.iloc[i,:])
        qf   = [0,w*L/2,w*L**2/12,0,w*L/2,-w*L**2/12] 
        Dict = {'qf':qf, 'dummy':np.zeros(6,'float')}
        qf = pd.DataFrame( data=Dict, index=range(0,6) ) 
        qfB.append(qf)
        Qf = T.T @ qf['qf'].values;                              
        QF.loc[Indx,'Load'] = QF.loc[Indx,'Load'] + Qf.values;
        ke = ke_frame(A,E,I,L);      keB.append(ke)   
        Ke = T.T @ ke @ T;           KeB.append(Ke)
        KE.iloc[Indx,Indx] = KE.iloc[Indx,Indx]+Ke.values;
    Beams['L'] = LB
    
    return [ KE,QF, LB,TB,keB,KeB,qfB ]


def Analyze_Beam( NF,  NOD,  FD, KE, QF
                , NB, IndxB, TB, keB, KeB, Beams, qfB ): 
            
    ## k  kX KX  
    K=KE
            
    kB, KB = [],[];
    for i in range(0,NB): 
        kB.append( keB[i] )
        KB.append( KeB[i] )    
     
    # Kred
    Kred = K.iloc[FD,FD]
        
    # U
    U = np.zeros([NOD,2],'float64')
    U = pd.DataFrame( U,columns = ['dis','Dummy'] ) 
         
    ## Gauss Ellimination  ax=b => x=?
    a = Kred.values
    b = (  NF.loc[FD,'Load'] - QF.loc[FD,'Load']  ).values
    n = len(b)
    x = np.zeros(n,'float64')

    #    Forward Elimination
    for k in range(n-1):
        for i in range(k+1,n):
            fctr = a[i,k] / a[k,k]
            for j in range(k,n):
                a[i,j] = a[i,j] - fctr*a[k,j]
            b[i] = b[i] - fctr*b[k]
    
    #    Back Substitution
    x[n-1] = b[n-1] / a[n-1,n-1]
    for i in range(n-2,-1,-1):
        Sum = b[i]
        for j in range(i+1,n):
            Sum = Sum - a[i,j]*x[j]
        x[i] = Sum / a[i,i]
           
    # U    
    U.loc[FD,'dis'] = x.copy()
        
    ## member forces
      
    #    Beam
    uB,UB,qB = [],[],[]
    for i in range(0,NB):
        Indx = list(IndxB.iloc[i,:])
        T    = TB[i] 
        k    = kB[i] 
        UB.append(  U.loc[Indx,'dis'].values  )    
        uB.append( T @ UB[-1] )
        qB.append( k @ uB[-1] )
         
    ## Reactions 
    R = np.zeros([NOD,2],'float64')
    R = pd.DataFrame( R, columns = ['Reactions','Dummy'] )
        
    #   Beam
    for i in range(0,NB):
        Indx = list(IndxB.iloc[i,:])
        T    = TB[i]
        q    = qB[i]
        Q    = T.T @ q  
        R.loc[Indx,'Reactions'] = R.loc[Indx,'Reactions'] + Q.values   
    
    return [ qB, R, U ]


# Converting Directions (D.S.M. to Standard.Static) 
def D2S( NB, qB, R  ):
    
    qBs = copy.deepcopy(qB) 
    for i in range(0,NB):
        for j in [0,2,4]: qBs[i].iloc[j] = -qBs[i].iloc[j]
            
    return [ qBs ]



# Global Displacements, Support Reactions & Internal Member Forces
# (auto) define a function called xyzNVM
# to generate Global Displacements, Structural Reaction, & Internal Member Forces

def xyzNVM( NB, U, R, qBs, nBay ):
        
    Ux = list( U.loc[ np.arange(0,len(U),3), 'dis' ] )
    Uy = list( U.loc[ np.arange(1,len(U),3), 'dis' ] )
    Uz = list( U.loc[ np.arange(2,len(U),3), 'dis' ] )
    index = np.arange(0,len(U)/3)
    Dict={'Ux':Ux,'Uy':Uy,'Uz':Uz}; Uxyz=pd.DataFrame(data=Dict,index=index)
    
    Rx = list( R.loc[ np.arange(0,len(R),3), 'Reactions' ] )
    Ry = list( R.loc[ np.arange(1,len(R),3), 'Reactions' ] )
    Rz = list( R.loc[ np.arange(2,len(R),3), 'Reactions' ] )
    index = np.arange(0,len(R)/3)
    Dict={'Rx':Rx,'Ry':Ry,'Rz':Rz}; Rxyz=pd.DataFrame(data=Dict,index=index)
    n = nBay+1;                     Rxyz=round( Rxyz.iloc[range(0,n),:], 2 )
            
    # Beam: left right    
    AxB_l, AxB_r, VB_l, VB_r, MB_l, MB_r = [],[],[],[],[],[]
    for i in range(0,NB):
        AxB_l.append( qBs[i].iloc[0] )
        AxB_r.append( qBs[i].iloc[3] )
        VB_l.append(  qBs[i].iloc[1] )
        VB_r.append(  qBs[i].iloc[4] )
        MB_l.append(  qBs[i].iloc[2] )
        MB_r.append(  qBs[i].iloc[5] )
    index=np.arange(0,NB)
    Dict={'left':AxB_l,'right':AxB_r}; AxB=pd.DataFrame(data=Dict,index=index)
    Dict={'left':VB_l, 'right':VB_r};   VB=pd.DataFrame(data=Dict,index=index)
    Dict={'left':MB_l, 'right':MB_r};   MB=pd.DataFrame(data=Dict,index=index)
    
    return [ Uxyz, Rxyz, AxB,VB,MB ]


# Adding Member-Forces to Elements DataFrame
def Add_member_forces( AxB,VB,MB, Beams ):

    # Beams
    Beams['Axial_Load'  ] = (np.round(  AxB['right'].values   ,0)).astype('int64')
    Beams['Shear_left'  ] = (np.round(  VB['left'  ].values   ,0)).astype('int64')
    Beams['Shear_right' ] = (np.round(  VB['right' ].values   ,0)).astype('int64')
    Beams['Moment_left' ] = (np.round(  MB['left'  ].values  ,-2)).astype('int64')
    Beams['Moment_right'] = (np.round(  MB['right' ].values  ,-2)).astype('int64')
    
    return [ Beams ]


# define a function to find Max or Min Uy of Main bays
def Find_Uy_MxMn( Points, Supports_Idx, Supports, NB, Uxyz ):
    
    # adding index of 'non-free' supports  +  'free' supports @ begining & end of continuous beam
    A = []
    if Supports[ 0]=='free': A.append(0)
    for i in Supports_Idx:
        if Supports[i]!='free': A.append(i)
    if Supports[-1]=='free'   : A.append(NB)
    
    # Uy into abs(Uy)
    U = Uxyz.copy().loc[:,['Uy']]
    U['Uy'] = abs(U['Uy'])
    
    # defining begin-end indexes of each beam
    B = []
    for i in range( len(A) ):
        try:
            B.append( [A[i],A[i+1]] )
        except:
            pass
        
    # finding index of max abs(Uy) for each beam
    I = [] 
    for b in B:
        C  = U.copy().loc[ float(b[0]):float(b[1]), ['Uy'] ]
        mx = C.loc[:,'Uy'].max()
        for i in C.index:
            if C.loc[i,'Uy'] == mx: I.append(i)
                
    # finding X, Max & Min of Uy
    Uy_MxMn = pd.concat( [ Points.loc[I,['x']], Uxyz.loc[I,['Uy']] ], axis=1 )
    Uy_MxMn['Uy']   =  np.round(Uy_MxMn['Uy'],2) 
    Uy_MxMn.columns = ['X','Uy']

    return Uy_MxMn


# 'V', 'M', None  ---  Visualize V or M on "Deformed Shape"
def Plot_Deformed( Points, Supports_Idx, Supports, wBay_Main, wBay, Beams, CnB,NB, Uxyz, Uy_MxMn, VM ):

    NewX = Points['x']+     Uxyz['Ux']
    NewY = Points['y']+0.65*Uxyz['Uy']/max(abs(Uxyz['Uy']))
    
    NewPoints = Points.copy() 
    NewPoints['x']=NewX.copy()
    NewPoints['y']=NewY.copy()
    
    fig, ax = plt.subplots( figsize=(12,3) )
    
    # Undeformed
    for i in CnB.index: 
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = Points.loc[i0,'y']
        y1 = Points.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=.75, c='gray', ls=':' )

    # deformed        
    for i in CnB.index: 
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = NewPoints.loc[i0,'x']
        x1 = NewPoints.loc[i1,'x']
        y0 = NewPoints.loc[i0,'y']
        y1 = NewPoints.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        if VM not in ['V','M']:
            ax.plot( x, y, lw=4, c='g', ls='-' )
    
    # Shear
    Beam_Shear = Beams['Shear_left']+Beams['Shear_right']
    LW = abs(  Beam_Shear / abs(Beam_Shear).max() ) * 4
    for i in CnB.index:
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = NewPoints.loc[i0,'x']
        x1 = NewPoints.loc[i1,'x']
        y0 = NewPoints.loc[i0,'y']
        y1 = NewPoints.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        if VM == 'V':
            ax.plot( x, y, lw=LW[i], c='dodgerblue', ls='-' )

    # Moment
    Beam_Moment = (Beams['Moment_left']+Beams['Moment_right'])/2
    LW = abs(  Beam_Moment / abs(Beam_Moment).max() ) * 4
    ML = Beams['Moment_left' ] / abs(Beam_Moment).max()
    MR = Beams['Moment_right'] / abs(Beam_Moment).max()
    for i in CnB.index:
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = NewPoints.loc[i0,'x']
        x1 = NewPoints.loc[i1,'x']
        y0 = NewPoints.loc[i0,'y']
        y1 = NewPoints.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        if Beam_Moment[i] >=0:  c='g'
        else                 :  c='r'
        if VM == 'M':
            ax.plot( x, y, lw=LW[i], c=c, ls='-' )
        
    # Pink Line    
    for i in Uy_MxMn.index:
        x0 = Uy_MxMn.loc[i,'X']
        x1 = Uy_MxMn.loc[i,'X']
        y0 = 0
        y1 = 0.65*Uy_MxMn.loc[i,'Uy']/max(abs(Uy_MxMn['Uy']))
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=2, c='deeppink', ls='-', marker='*' )
        if   y1>=0:  plt.text( x0, y1+0.10, str(Uy_MxMn.loc[i,'Uy']), ha='center', font='MV BOLI', fontsize=14 )
        elif y1< 0:  plt.text( x0, y1-0.27, str(Uy_MxMn.loc[i,'Uy']), ha='center', font='MV BOLI', fontsize=14 )
        

    for i in NewPoints.index:
        x = NewPoints.loc[i,'x']
        y = NewPoints.loc[i,'y']
        if   Supports[i]=='fixed' : marker, ec, fc, y = 's', 'lightgray', 'lightgray', y-0.0
        elif Supports[i]=='pinned': marker, ec, fc, y = '^', 'lightgray', 'lightgray', y-0.1
        elif Supports[i]=='roller': marker, ec, fc, y = 'o', 'lightgray', 'lightgray', y-0.1
        elif Supports[i]=='free'  : marker, ec, fc, y = 'x',      'gray',      'gray', y-0.0
        elif Supports[i]=='mid'   : marker, ec, fc, y = '' ,      'gray',      'gray', y-0.0
        ax.scatter( x, y, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        if   Supports[i]=='fixed' :
            ax.scatter( x, y+0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
            ax.scatter( x, y-0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        
    # title
    if   VM not in ['V','M']:   TITLE =           'Deformed Shape'
    elif VM == 'V'          :   TITLE =  'Shear on Deformed Shape'
    elif VM == 'M'          :   TITLE = 'Moment on Deformed Shape'
    plt.title( TITLE
             #+'\nUy = '+str(Uy_MxMn['Uy'].tolist())+' (mm)'
             , font='MV BOLI', fontsize=16 )
    
    # xticks
    A = []
    if Supports[ 0]=='free'   : A.append( 0 )
    n = 0
    for i in Supports_Idx:
        if Supports[i]!='free': A.append( sum(wBay_Main[:n]) )
        n = n+1
    if Supports[-1]=='free'   : A.append( sum(wBay_Main)     )
        
    Xwalk = list( set( A + Uy_MxMn['X'].tolist() ))
    xtks = np.array(Xwalk,dtype='f8') 
    ax.set_xticks( xtks )
    plt.xticks(font='MV BOLI',rotation=45,fontsize=14) 
    
    # yticks
    Ywalk = [0]
    ytks = np.array(Ywalk,dtype='f8')
    ax.set_yticks( ytks )
    plt.yticks(font='MV BOLI',rotation=0,fontsize=14)
    
    # legend    
    if   VM not in ['V','M']:   
        ax.plot( [0,0], [0,0], lw=4, c='g', ls='-', label='Deformed' )
    elif VM == 'V'          :   
        ax.plot( [0,0], [0,0], lw=4, c='dodgerblue', ls='-', label='Shear' )
    elif VM == 'M'          :   
        ax.plot( [0,0], [0,0], lw=4, c='r', ls='-', label='--ve Moment' )
        ax.plot( [0,0], [0,0], lw=4, c='g', ls='-', label='+ve Moment' )
    plt.legend()
    
    # ylim
    plt.ylim([-1,+1])
    
    # save image
    if   VM not in ['V','M']:   Image_Name = 'Deformed Shape'
    elif VM == 'V'          :   Image_Name = 'Deformed Shape with Shear'
    elif VM == 'M'          :   Image_Name = 'Deformed Shape with Moment'
    plt.savefig(Image_Name+'.png', dpi=120) 
    plt.show()
 
    return None

# define a function to find 0, Max or Min Shear of Main bays
def Find_V_MxMn( NB, Beams, Supports_Idx, Supports, wBay_Main, wBay, L_tinyBeam ):
    
    # Shear @ left-Most Support
    X = [ 0 ]                             # X & V  save location of all values
    V = [ Beams.loc[0,'Shear_left'] ]
    
    # Shear @ all supports between left-most & right-most ones
    X_MxMn, V_MxMn = X.copy(), V.copy()   # X_MxMn & V_MxMn  save  location of Min or Max Values
    for i in range(1,NB-1):
        xl = Beams.loc[:i-1,'L'].sum()
        xr = Beams.loc[:i  ,'L'].sum()
        vl = Beams.loc[i,'Shear_left' ] 
        vr = Beams.loc[i,'Shear_right']
        # @ Main Supports, for discontinuity, both left & right Shears are important
        if i in Supports_Idx[:-1] and Supports[i] != 'mid':
            X[-1] = xl    # drag X[-1] to location of Main-Support
            X.append(xl); X_MxMn.append(X[-2]); X_MxMn.append(X[-1])
            V.append(vl); V_MxMn.append(V[-2]); V_MxMn.append(V[-1])
        # @ mid Supports, average shear is considered
        else:
            x = (xl+xr)/2
            v = (vl+vr)/2
            X.append(x)
            V.append(v)
          
    # Shear @ Right-Most Support
    i = Supports_Idx[-1]
    xr = sum(wBay)
    vr = Beams.loc[i-1,'Shear_right'] 
    X.append(xr); X_MxMn.append(xr)
    V.append(vr); V_MxMn.append(vr)
    
    # Saving Values of X & V in a dataframe
    df = pd.DataFrame()
    df['X'] = X.copy()   # all values
    df['V'] = V.copy()   # all values
    V_Shear = df.copy()
    
    # V_MxMn
    X = []
    for i in range(len(wBay_Main)+1):
        X.append( sum(wBay_Main[:i]) )
    I = []
    for i in V_Shear.index:
        if V_Shear.loc[i,'X'] in X: 
            I.append(i)
    Z = []
    V = V_Shear.copy()
    V.loc[:,'V'] = abs(V.loc[:,'V'])
    for i in V.index:
        try:
            ADD = False
            if V.loc[i-1,'V']>=V.loc[i,'V'] and V.loc[i,'V']<=V.loc[i+1,'V']:
                ADD = True
                for j in I:
                    if abs(V.loc[j,'X']-V.loc[i,'X']) <=  1.99*L_tinyBeam: 
                        ADD=False
            if ADD==True: Z.append(i)
        except:
            pass
    V_MxMn = V_Shear.copy().loc[I+Z,:]
    V_MxMn = V_MxMn.drop_duplicates()
    V_MxMn = V_MxMn.sort_index()
    
    return [ V_Shear, V_MxMn ]


def Plot_Shear( Points, Supports, wBay_Main, wBay, CnB,NB, V_Shear, V_MxMn ): 
    
    fig, ax = plt.subplots( figsize=(12,3) )
    
    # Undeformed
    for i in CnB.index: 
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = Points.loc[i0,'y']
        y1 = Points.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=.75, c='gray', ls=':' )
            
    # Shear
    X = V_Shear['X'].tolist()
    V = V_Shear['V'].tolist()    
    LW = abs(  V / max(abs(np.array(V))) ) * 4
    for i in range(len(X)-1):
        x0 = X[i  ]
        x1 = X[i+1]
        y0 = V[i  ] / max(abs(np.array(V))) * .70
        y1 = V[i+1] / max(abs(np.array(V))) * .70
        x  = [x0,x1]
        y  = [y0,y1]
        ax.plot( x, y, lw=(LW[i]+LW[i+1])/2, c='dodgerblue', ls='-' )
        ax.fill_between( x, [0,0], y, color='cyan', alpha=0.75 )
        
    # Text
    for i in V_MxMn.index:
        x =     V_MxMn.loc[i,'X']
        y = .70*V_MxMn.loc[i,'V']/max(abs(V_MxMn['V']))
        if   y>=0:  plt.text( x, y+0.10, str(int(V_MxMn.loc[i,'V'])), ha='center', font='MV BOLI', fontsize=14 )
        elif y< 0:  plt.text( x, y-0.27, str(int(V_MxMn.loc[i,'V'])), ha='center', font='MV BOLI', fontsize=14 )
        
    # Supports
    for i in Points.index:
        x = Points.loc[i,'x']
        y = Points.loc[i,'y']
        if   Supports[i]=='fixed' : marker, ec, fc, y = 's', 'lightgray', 'lightgray', y-0.0
        elif Supports[i]=='pinned': marker, ec, fc, y = '^', 'lightgray', 'lightgray', y-0.1
        elif Supports[i]=='roller': marker, ec, fc, y = 'o', 'lightgray', 'lightgray', y-0.1
        elif Supports[i]=='free'  : marker, ec, fc, y = 'x',      'gray',      'gray', y-0.0
        elif Supports[i]=='mid'   : marker, ec, fc, y = '' ,      'gray',      'gray', y-0.0
        ax.scatter( x, y, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        if   Supports[i]=='fixed' :
            ax.scatter( x, y+0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
            ax.scatter( x, y-0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        
    # title
    X = V_MxMn['X'].tolist()
    V = V_MxMn['V'].tolist() 
    plt.title( 'Shear (KN)', font='MV BOLI', fontsize=16 )
    
    # xticks
    Xwalk = [0]
    for i in wBay_Main:
        w = Xwalk[-1] + i 
        Xwalk.append(w)
    Xwalk = list( set( Xwalk + X ))
    xtks = np.array(Xwalk,dtype='f8') 
    ax.set_xticks( xtks )
    plt.xticks(font='MV BOLI',rotation=45,fontsize=14)
    
    # yticks
    Ywalk = [0]
    ytks = np.array(Ywalk,dtype='f8')
    ax.set_yticks( ytks )
    plt.yticks(font='MV BOLI',rotation=0,fontsize=14)
    
    # legend    
    ax.plot( [0,0], [0,0], lw=4, c='dodgerblue', ls='-', label='Shear' )
    plt.legend()
    
    # ylim
    plt.ylim([-1,+1])
    
    # grid
    plt.grid('on',alpha=.1)
    
    # save image
    plt.savefig('Shear.png', dpi=120) 
    plt.show() 
 
    return None



# finding 0, Max or Min Moment of Main bays
def Find_M_MxMn( NOP, NB, Beams ):
    
    A = Beams[ ['L','Moment_left','Moment_right'] ].copy() 
    
    X = [0]
    for i in A.index:
        X.append( X[-1]+A.loc[i,'L'] )
        
    A['XL'] = X[:-1]
    A['XR'] = X[+1:]
    A['ML'] = A['Moment_left' ]
    A['MR'] = A['Moment_right']
    A['X' ] = (A['XL']          + A['XR']) / 2
    A['M' ] = (A['Moment_left'] + A['Moment_right']) / 2
    A = A[ ['XL','XR','ML','MR','X','M'] ]
    
    MAX, MIN = [],[]
    if A.loc[0,'M'] < A.loc[1,'M']:   MIN.append([ 0, A.loc[0,'X'], A.loc[0,'M'] ])
    else                          :   MAX.append([ 0, A.loc[0,'X'], A.loc[0,'M'] ])
        
    for i in range(1,NOP-2,1):
    
        i_X_0  = A.loc[i-1,'X']
        i_M_0  = A.loc[i-1,'M']
    
        i_X_1  = A.loc[i,'X']
        i_M_1  = A.loc[i,'M']
    
        i_X_2  = A.loc[i+1,'X']
        i_M_2  = A.loc[i+1,'M']
       
        Left_Slope  = i_M_1 - i_M_0
        Right_Slope = i_M_2 - i_M_1
    
        if   Left_Slope<=0 and 0<=Right_Slope:   MIN.append([ i, i_X_1, i_M_1 ])
        elif Left_Slope>=0 and 0>=Right_Slope:   MAX.append([ i, i_X_1, i_M_1 ])
            
    if A.loc[NB-1,'M'] < A.loc[NB-2,'M']:   MIN.append([ NB-1, A.loc[NB-1,'X'], A.loc[NB-1,'M'] ])
    else                                :   MAX.append([ NB-1, A.loc[NB-1,'X'], A.loc[NB-1,'M'] ])
        
    MIN = pd.DataFrame( MIN, columns=['Idx','X','M'] )
    MAX = pd.DataFrame( MAX, columns=['Idx','X','M'] )
    MIN.index = MIN['Idx']
    MAX.index = MAX['Idx']
    MIN = MIN[['X','M']]
    MAX = MAX[['X','M']]
    
    for i in MIN.index:
        XL = A.loc[i,'XL'];   ML = A.loc[i,'ML']
        XR = A.loc[i,'XR'];   MR = A.loc[i,'MR']
        if ML <= MR: X,M = XL, ML
        else       : X,M = XR, MR
        MIN.loc[i,'X'] = X
        MIN.loc[i,'M'] = int(np.round(M/10**3,0))
    
    for i in MAX.index:
        XL = A.loc[i,'XL'];   ML = A.loc[i,'ML']
        XR = A.loc[i,'XR'];   MR = A.loc[i,'MR']
        if   ML > MR: X,M = XL, ML
        elif ML < MR: X,M = XR, MR
        else        : X,M = (XL+XR)/2, (ML+MR)/2
        MAX.loc[i,'X'] = X
        MAX.loc[i,'M'] = int(np.round(M/10**3,0))

    MIN['M'] = MIN['M'].astype(int)
    MAX['M'] = MAX['M'].astype(int)
    
    # averaging X of neighbors with the same M MAX
    for i in MAX.index:
        try:
            if MAX.loc[i,'M'] == MAX.loc[i+1,'M']:
                X = (MAX.loc[i,'X']+MAX.loc[i+1,'X']) / 2
                MAX.loc[i  ,'X'] = X
                MAX.loc[i+1,'X'] = X
        except:
            pass
    
    # removing duplicates from MAX
    MIN = MIN.drop_duplicates()
    MAX = MAX.drop_duplicates()
    
    return [ MIN, MAX ]



# (auto) define a function to find Flexure Point
def Find_Flexture_Points( Points, NOP, Beams ):
    
    # Moment of all beams
    B = Beams[ ['Moment_left','Moment_right'] ].copy()   
    B['Moment'] = (B['Moment_left']+B['Moment_right'])/2   
    
    # Indexes of points of sign-change 
    I = []                                                 
    for i in range(NOP-3):
        if B.loc[i,'Moment'] * B.loc[i+1,'Moment'] < 0:
            I.append(i)     
            I.append(i+1)   
            I.append(i+2)
            
    # averaging @ points of sign-change
    X  = Points.loc[I,['x']]
    X.index = range( X.shape[0] )
    XX = []
    for i in range( 0, X.shape[0]-1, 3 ):
        xx = (X.loc[i,'x']+X.loc[i+1,'x'])/2  ; XX.append(xx)
        xx = (X.loc[i+1,'x']+X.loc[i+2,'x'])/2; XX.append(xx)
        XX.append(0.0)   # adding Extra 0s for ease of calculations
    
    # applying the average values
    B = B.loc[I,['Moment']]
    B.index = range(B.shape[0])
    B['x'] = XX
    
    # removing Extra 0s
    Keep = []
    for i in B.index:
        if B.loc[i,'x'] > 0.0: Keep.append(i)
    B = B.loc[Keep,:]
    B.index = range(B.shape[0])
    
    # Flexure Points
    Flex = []
    for i in range(0,B.shape[0],2):
        M1 = abs(B.loc[i  ,'Moment'])
        M2 = abs(B.loc[i+1,'Moment'])
        x1 =     B.loc[i  ,'x']
        x2 =     B.loc[i+1,'x']
        x = x1 + (x2-x1)*M1/(M1+M2)
        x = np.round(x,1)
        Flex.append(x)
    
    return Flex


def Plot_Moment( Points, Supports, wBay_Main, wBay, CnB,NB,Beams, M_MIN, M_MAX, Flex ): 
    
    fig, ax = plt.subplots( figsize=(12,3) )
    
    # Undeformed
    for i in CnB.index: 
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = Points.loc[i0,'y']
        y1 = Points.loc[i1,'y']
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=.75, c='gray', ls=':' ) 
    
    # Moment
    Beam_Moment = (Beams['Moment_left']+Beams['Moment_right'])/2
    LW = abs(  Beam_Moment / abs(Beam_Moment).max() ) * 4
    ML = Beams['Moment_left' ] / abs(Beam_Moment).max()
    MR = Beams['Moment_right'] / abs(Beam_Moment).max()
    for i in CnB.index:
        i0 = CnB.loc[i,'from_point']
        i1 = CnB.loc[i,'to_point'  ]
        x0 = Points.loc[i0,'x']
        x1 = Points.loc[i1,'x']
        y0 = ML[i]*.7
        y1 = MR[i]*.7
        x = [x0,x1]
        y = [-y0,-y1]
        if Beam_Moment[i] >=0:
            ax.plot( x, y, lw=LW[i], c='g', ls='-' )
            ax.fill_between( x, [0,0], y, color='lawngreen', alpha=0.07 )
        else                 :
            ax.plot( x, y, lw=LW[i], c='r', ls='-' )
            ax.fill_between( x, [0,0], y, color='deeppink' , alpha=0.07 )
        
    for x in Flex:
        x0 = x
        x1 = x
        y0 = -.25
        y1 = +.25
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=1.5, c='dodgerblue', ls='--', marker='' )
        
    for i in M_MIN.index:
        x0 = M_MIN.loc[i,'X']
        x1 = M_MIN.loc[i,'X']
        y0 = 0
        y1 = -.7*M_MIN.loc[i,'M'] / max( abs(M_MIN['M']).tolist() + abs(M_MAX['M']).tolist() )
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=1.5, c='r', ls='--', marker='' )
        if   y1>=0:  plt.text( x1, y1+.1, str(int(M_MIN.loc[i,'M'])), ha='center', font='MV BOLI', fontsize=14 )
        elif y1< 0:  plt.text( x1, y1-.2, str(int(M_MIN.loc[i,'M'])), ha='center', font='MV BOLI', fontsize=14 )
        
    for i in M_MAX.index:
        x0 = M_MAX.loc[i,'X']
        x1 = M_MAX.loc[i,'X']
        y0 = 0
        y1 = -.7*M_MAX.loc[i,'M'] / max( abs(M_MIN['M']).tolist() + abs(M_MAX['M']).tolist() )
        x = [x0,x1]
        y = [y0,y1]
        ax.plot( x, y, lw=1.5, c='g', ls='--', marker='' )
        if   y1>=0:  plt.text( x1, y1+.10, str(int(M_MAX.loc[i,'M'])), ha='center', font='MV BOLI', fontsize=14 )
        elif y1< 0:  plt.text( x1, y1-.25, str(int(M_MAX.loc[i,'M'])), ha='center', font='MV BOLI', fontsize=14 )
        

    # Supports
    for i in Points.index:
        x = Points.loc[i,'x']
        y = Points.loc[i,'y']
        if   Supports[i]=='fixed' : marker, ec, fc, y = 's', 'red' , 'red' , y-0.0
        elif Supports[i]=='pinned': marker, ec, fc, y = '^', 'red' , 'red' , y-0.1
        elif Supports[i]=='roller': marker, ec, fc, y = 'o', 'red' , 'red' , y-0.1
        elif Supports[i]=='free'  : marker, ec, fc, y = 'x', 'gray', 'gray', y-0.0
        elif Supports[i]=='mid'   : marker, ec, fc, y = '' , 'gray', 'gray', y-0.0
        ax.scatter( x, y, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        if   Supports[i]=='fixed' :
            ax.scatter( x, y+0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
            ax.scatter( x, y-0.1, marker=marker, s=150, ec=ec, fc=fc, alpha=0.75 )
        
    # title
    plt.title( 'Moment (KNm)', font='MV BOLI', fontsize=16 )
    
    # xticks
    Xwalk = [0, sum(wBay_Main)]
    Xwalk = list( set( Xwalk + Flex + M_MIN['X'].tolist() + M_MAX['X'].tolist() ))
    xtks = np.array(Xwalk,dtype='f8') 
    ax.set_xticks( xtks )
    plt.xticks(font='MV BOLI',rotation=45,fontsize=14)
    
    # yticks
    Ywalk = [0]
    ytks = np.array(Ywalk,dtype='f8')
    ax.set_yticks( ytks )
    plt.yticks(font='MV BOLI',rotation=0,fontsize=14)
    
    # legend    
    ax.plot( [0,0], [0,0], lw=4, c='r', ls='-', label='--ve M (KNm)  /\ @ Supports' )
    ax.plot( [0,0], [0,0], lw=4, c='g', ls='-', label='+ve M (KNm)  \/ @ Mid-Spans' )
    plt.legend(loc='best')
    
    # ylim
    plt.ylim([-1,+1])
    
    # grid
    plt.grid('on',alpha=.1)
    
    # save image
    plt.savefig('Moment.png', dpi=120) 
    plt.show() 
 
    return None


def Plot_Support_Reactions( Rxyz, nPieces ):
    
    eps = 10**(-12)
    
    R, RR = Rxyz.copy(), Rxyz.copy()
    R['Rx'] = R['Rx'].values / (eps+max(np.abs(R['Rx'].values)))
    R['Ry'] = R['Ry'].values / (eps+max(np.abs(R['Ry'].values)))
    R['Rz'] = R['Rz'].values / (eps+max(np.abs(R['Rz'].values)))

    I = [0]
    for i in nPieces: I = I + [I[-1]+i]
    
    R  =  R.loc[I,:]
    RR = RR.loc[I,:]

    R.index  = range( R.shape[0])
    RR.index = range(RR.shape[0])

    n = 1+len(nPieces)
    xR,yR,zR = [],[],[]
    for i in range(0,n):
        xR.append(2*i+0.5)
        yR.append(2*i+1.0)
        zR.append(2*i+1.5)
    
    fig, ax = plt.subplots( figsize=(12,5) )
    for i in range(0,n):
        plt.bar( xR,R['Rx'], width=0.5, color='cyan'      ,  edgecolor='blue'  )
        plt.bar( yR,R['Ry'], width=0.5, color='lightgreen',  edgecolor='green' )
        plt.bar( zR,R['Rz'], width=0.5, color='deeppink'  ,  edgecolor='red'   )
    ax.legend(['Rx (KN)','Ry (KN)','Rz (KNm)'])

    # Decoration
    plt.title( 'Support Reactions',font='MV BOLI',fontsize=16)
    plt.xlabel( 'Supports from left to right',font='MV BOLI',fontsize=16)

    for i, val in enumerate( np.round( RR['Rx'].values, 0 ) ): 
        plt.text(xR[i], R['Rx'][i], int(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':16})
    for i, val in enumerate( np.round( RR['Ry'].values, 0 ) ): 
        plt.text(yR[i], R['Ry'][i], int(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':16})
    for i, val in enumerate( np.round( RR['Rz'].values/10**3, 0 ) ):
        plt.text(zR[i], R['Rz'][i], int(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':16})

    # xtick
    xtick_labels = []
    for i in range(1,n+1):
        xtick_labels.append('Support_'+str(i))
    ax.set_xticks( yR ) 
    ax.set_xticklabels( xtick_labels )
    plt.xticks(font='MV BOLI',rotation=0,fontsize=14)  

    # ytick
    ytick_labels = [' '] 
    ax.set_yticks( [0] ) 
    ax.set_yticklabels( ytick_labels ) 
    
    # ylim
    plt.ylim([-1.2,+1.2])
    
    # grid
    plt.grid('on',alpha=.1)
    
    # save image
    plt.savefig('Support_Reactions.png', dpi=120) 
    plt.show()
    
    return None


# (auto) 
def Find_Accuracy( wBay_Main, W_Beams, Supports, NF, R ):
    LOAD = 0  # (KN)  Applied Load
    REAC = 0  # (KN)  Caculated Reactions

    for i in range( len(Supports) ):
        if ( (Supports[i] == 'free') or (Supports[i] == 'mid') ):   LOAD = LOAD + NF.loc[3*i+1,'Load'     ]
        else                                                    :   REAC = REAC +  R.loc[3*i+1,'Reactions']
    for i in range( len(wBay_Main) )                            :   LOAD = abs(LOAD) + W_Beams[i] * wBay_Main[i]
    
    Accuracy = np.round( (REAC/LOAD)*100, 1 )
    return Accuracy



# from L to R   >>>>>>
# from 0 to 6
def Moment_Redist_L2R( left_idx, right_idx, Beams, redist ):   # eg: 0, 6, Beams, 0.20
    I0, I1 = left_idx, right_idx
    M_init = Beams.loc[I0,'Moment_left']
    DM     = M_init * redist
    n      = abs(I1-I0)
    step   = DM/n
    ML, MR, Idx = [],[], []
    for i in range(I0,I1,1):
        Idx.append(i)
        ml = Beams.loc[i,'Moment_left' ] - DM;   ML.append(round(ml))
        DM = DM - step
        mr = Beams.loc[i,'Moment_right'] - DM;   MR.append(round(mr))
    df         = pd.DataFrame([ML,MR]).T
    df.index   = Idx
    df.columns = ['Moment_left','Moment_right']
    #df         = df.sort_index(ascending=True)
    return df



# from R to L   <<<<<<
# from 6 to 0
def Moment_Redist_R2L( left_idx, right_idx, Beams, redist ):   # eg: 0, 6, Beams, 0.20
    I0, I1 = right_idx, left_idx
    M_init = Beams.loc[I0-1,'Moment_right']
    DM     = M_init * redist
    n      = abs(I0-I1)
    step   = DM/n
    ML, MR, Idx = [],[],[]
    for i in range(I0,I1,-1):
        Idx.append(i-1)
        mr = Beams.loc[i-1,'Moment_right'] - DM;   MR.append(round(mr))
        DM = DM - step
        ml = Beams.loc[i-1,'Moment_left' ] - DM;   ML.append(round(ml))
    df         = pd.DataFrame([ML,MR]).T
    df.index   = Idx
    df.columns = ['Moment_left','Moment_right']
    df         = df.sort_index(ascending=True)
    return df



def Moment_Redistribute( Supports_Idx, Supports, Beams, redist ):
    
    I = [ Supports_Idx[0] ]                    # first support
    for i in Supports_Idx[1:-1]: 
        if Supports[i] != 'free': I.append(i)  # 'non-free' inner supports
    I.append( Supports_Idx[-1] )               # last support

    df_redist = Beams.copy().loc[ :, ['L','Moment_left','Moment_right'] ]

    for i in range( len(I)-1 ):
    
        left_idx, right_idx = I[i], I[i+1]
        L2R, R2L            = True, True
    
        if Supports[left_idx]=='free' or Supports[right_idx]=='free': 
            L2R = False
            R2L = False
        
        try:
            left_neighbour_idx = I[i-1]
            if Supports[left_neighbour_idx]=='free': L2R = False
        except:
            pass
        
        try:
            right_neighbour_idx = I[i+2]
            if Supports[right_neighbour_idx]=='free': R2L = False
        except:
            pass
    
        if L2R:
            df = Moment_Redist_L2R( left_idx, right_idx, df_redist, redist )
            for i in df.index:
                df_redist.loc[i,'Moment_left' ] = df.loc[i,'Moment_left' ]
                df_redist.loc[i,'Moment_right'] = df.loc[i,'Moment_right']

        if R2L:
            df = Moment_Redist_R2L( left_idx, right_idx, df_redist, redist )
            for i in df.index:
                df_redist.loc[i,'Moment_left' ] = df.loc[i,'Moment_left' ]
                df_redist.loc[i,'Moment_right'] = df.loc[i,'Moment_right']
                
    return df_redist