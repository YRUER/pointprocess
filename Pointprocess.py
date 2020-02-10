import numpy as np
from matplotlib import pyplot, patches
import matplotlib.colors as colors
import scipy.stats as stats
import scipy.optimize as opt
from shapely.geometry import *
from shapely.ops import unary_union
import copy as copy

class Pointprocess(object):

    #S: list of multipoints, region: polygon, intFunc: callable
    def __init__(self,S=[],region=None,intFunc=lambda x,y:1):

        self.S=S
        self.region=region
        
        #setting boundary box and removing points outside of region (if possible)
        if(region):
            self.minBox=region.bounds
            f=lambda x,y: intFunc(x,y) if self.region.contains(Point(x,y)) else -0.0001
            self.intFunc=np.vectorize(f)
            if(S):
                nOfSamples=len(S)
                for n in range(nOfSamples):
                    S[n]=S[n].intersection(region)
        elif(S):
            self.minBox=unary_union(S).bounds
            self.region=unary_union(S).bounds
            self.intFunc=np.vectorize(intFunc)
        else:
            self.minBox=()
            self.intFunc=np.vectorize(intFunc)

    def plotProcess(self):

        '''plots: points of the first realization (S[0]) pointprocess,
        region and intensity'''
        
        fig, ax=pyplot.subplots()
        xmin=self.minBox[0]
        ymin=self.minBox[1]
        xmax=self.minBox[2]
        ymax=self.minBox[3]

        if(self.region):
            patch=patches.Polygon(np.array(self.region.exterior),
                alpha=0.5,linestyle='-',linewidth=1,fill=False)
            ax.add_patch(patch)
        if(self.S):
            ax.scatter([p.x for p in self.S[0]],[p.y for p in self.S[0]],c='black')
            X,Y=np.meshgrid(np.linspace(xmin,xmax,200),np.linspace(ymin,ymax,200))
            intensity=self.intFunc(X,Y)
            colorMap=copy.copy(pyplot.cm.coolwarm)
            colorMap.set_under('w')

            im=ax.imshow(intensity, interpolation='bilinear',
                cmap=colorMap,
                norm=colors.Normalize(vmin=0),
                aspect='auto',
                origin='lower',
                extent=[xmin,xmax,ymin,ymax],
                alpha=0.9)
            ax.set_title('')
            cbar=fig.colorbar(im, extend='both', shrink=1, ax=ax)
            cbar.set_label('intensity')

            pyplot.plot()
            pyplot.show()

class PoissonProcess(Pointprocess):

    def __init__(self,S=[],region=None,intFunc=lambda x,y:1):
        Pointprocess.__init__(self,S,region,intFunc)

    def simHomogeneousPPP(self,intensity):
        '''returns realization of homogeneous Poissonprocess with intensity intensity'''

        N=np.random.poisson(lam=intensity*self.region.area)
        count=0
        accPoints=[]

        while(count<N):

            p=Point(np.random.uniform(self.minBox[0],self.minBox[2]),
                    np.random.uniform(self.minBox[1],self.minBox[3]))
            if(self.region.contains(p)):
                accPoints.append(p)
                count+=1

        return MultiPoint(accPoints)

    def simPPP(self,save=False):
        '''returns realization of (in)homogeneous Poissonprocess with intensity function intFunc,
        may be stored in S (save=True)'''
        

        out=opt.brute(lambda x:-1*self.intFunc(x[0],x[1]),
                                    ranges=((self.minBox[0],self.minBox[2],0.5),
                                    (self.minBox[1],self.minBox[3],0.5)),
                                    Ns=100,
                                    full_output=True)
        #heuristic upper bound of intensity function
        upper=-2*out[1]
        homPPP=self.simHomogeneousPPP(upper)
        thinPP=[p for p in homPPP if upper*np.random.ranf()<=self.intFunc(p.x,p.y)]
        inHomPPP=MultiPoint(thinPP)

        if(save):
            self.S.append(inHomPPP)

        return inHomPPP



if __name__=='__main__':
    pass
    '''
    PP=Pointprocess(S=[MultiPoint([(0,0),(0,1),(1,1),(5,5)])],region=Polygon([(-1,-1),(2,-1),(2,2),(-1,3)]),intFunc=lambda x,y:(x-1)**2+(y-1)**2)
    PP.plotProcess()
    PPP=PoissonProcess(region=Polygon([(-1,-1),(2,-1),(2,2),(-1,3)]),intFunc=lambda x,y:10*(x-1)**2)
    PPP.simPPP(save=True)
    PPP.plotProcess()
    '''










