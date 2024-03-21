from pylab import *
import scipy.stats as st
import scipy.io as sio
import pickle as cp
from tqdm import tqdm
import os

# Define localtion of data, and where to store things
root_path = os.path.dirname(os.path.abspath(__file__))
# Path to images
data_path = os.path.join(root_path, 'data/').replace('\\','/')
# Where to save stuff
log_path = os.path.join(root_path, 'logs/').replace('\\','/')

### Define filters ###

def DoG_filt(std1 = 3, std2 = 4, A1 = 1., A2 = 1., pat_size = 16):
    '''Filter of 2D diff. of Gaussians.'''
    
    fx, fy = meshgrid(linspace(-pat_size/2,pat_size/2,pat_size),
                    linspace(-pat_size/2,pat_size/2,pat_size))
    rho = sqrt(fx**2+fy**2)
    
    # Divide by d1 and d2 to cancel 1D to 2D scaling
    filt = A1*st.norm.pdf(rho, scale = std1)/(sqrt(2*pi)*std1) - \
           A2*st.norm.pdf(rho, scale = std2)/(sqrt(2*pi)*std2)
    filt = (filt-mean(filt))/norm(filt)

    return filt

def gabor(A=1.0, x0=0.0, y0=0.0, sigx=0.3*5, sigy=0.6*5, f=1.0/5, psi=pi/2,
          ang=0.0, ran=None, n=16, normed=True):
    '''A,ran should not matter. 
       f, psi, sigx, sigy is for shape. 
       Hint: send sigx=sigx/f, to scale with the frequency.
       x0, y0, ang is positioning.   '''

    g=zeros([n,n])
    if ran is None:
        ran=(n-1)/2.
    xs = linspace(-ran,ran,n)
    ys = linspace(-ran,ran,n)
    y,x = meshgrid(ys,xs)
    
    xp = (x-x0)*cos(ang) + (y-y0)*sin(ang)
    yp = -(x-x0)*sin(ang) + (y-y0)*cos(ang)
    
    g = A*cos(2*pi*f*xp + psi)*exp(-(xp/(sqrt(2)*sigx))**2 \
                                        -(yp/(sqrt(2)*sigy))**2)

    if normed:
        if norm(g.flatten()) > 1E-10:
            g = g/norm(g.flatten())
        else:
            return ones([n,n])/norm(ones([n,n]))
    return g

def get_gabor_combinations(f=[0.5], psi=[0.0], sigx=[3.0], sigy=[1.0], ang=[0.0], n=16):
    ''' Get many combinations of gabor parameters. '''
    gabors = []
    names = []

    for ppsi in psi:
        for ff in f:
            for ssigy in sigy:
                for ssigx in sigx:
                    for aang in ang:
                        gabors.append(gabor(f = ff, psi = ppsi, sigx = ssigx/ff, 
                                                sigy = ssigy/ff, ang = aang, n=n))
                        names.append('g_f%.1f_p%.1f_s%.1f_a%.1f_t%.1f' % 
                                     (ff, ppsi, ssigx, ssigy, aang))

    return gabors, names

def random_filt(n = 16):
    ''' Random noise 2D filter. '''
    a = randn(n,n)
    a -= mean(a)
    return a/norm(a)

def fourier_filt(freq1 = 2.0, freq2 = None, n = 16, normed = True):
    ''' 2D fourier filter. '''
    if freq2 is None:
        freq2 = freq1
    a = zeros([n,n])
    xs=linspace(-n/2,n/2,n)
    for i in arange(n):
        for j in arange(n):
            a[i,j] = sin(2.0*pi*xs[i]/freq1)*cos(2.0*pi*xs[j]/freq2)
    if normed:
        a = a-mean(a.flatten())
        a = a/norm(a.flatten())
     
    return a

def filter_bank(psize = 16):
    ''' Collection of different 2D filters '''

    names = []
    a = []
    a += [fourier_filt(psize*1.5,psize*2*1.5,n=psize)]
    a += [fourier_filt(2,n=psize)]
    a += [fourier_filt(4,n=psize)]
    a += [fourier_filt(psize//2,n=psize)]
    a += [fourier_filt(psize,n=psize)]
    names += ['fourier45','fourier2', 'fourier4', 'fourier8', 'fourier16']
    
    a += [random_filt(psize)]
    names.append('random')

    p = zeros([psize,psize])
    p[psize//2-1:psize//2+1,psize//2-1:psize//2+1] = 1
    a += [p]
    a += [-p]
    names.append('pixel')
    names.append('pixelneg')

    f = 1.0/array([3, 6, 9])
    psi = [0.0, pi/2]
    ang = [pi/3]
    sigx = [0.1, 0.3]
    sigy = [0.4, 0.6, 1.0]
    g, na = get_gabor_combinations(f=f, psi=psi, sigy=sigy, sigx=sigx, 
                                   ang=ang, n=psize)
    a += g
    names += na
    
    a += [DoG_filt(pat_size=psize)]
    names.append('DoG')
    
    a = array(a)
    
    return a, names



### Helpers to define activation functions ###

def integrate_func_vec(f, step = 0.1, lim = 12):
    ''' Numerically integrate a function f(x). '''
    xi = arange(-lim,lim,step)
    yi = f(xi)
    ri = cumsum(yi)*step
    return ri

def int_func(f, step = 0.005, lim = 20):
    ''' Lambda function to numerical integral of f(x). '''
    r = integrate_func_vec(f, step, lim)
    return lambda x:r[int32((x+lim)/step)]

def inv_func(f, step = 0.005, lim1 = -40, lim2 = 40):
    ''' Numerically inverts a function f(x). '''
    s0 = f(lim1)
    sf = f(lim2)
    sn = (sf-s0)/step
    vs = zeros(int32(sn)+1)
    
    ic = 1
    vs[0] = s0
    for i in arange(lim1,lim2+step/20.0,step/20.0):
        if f(i) > s0 + ic*step:
            vs[ic] = i
            ic += 1

    return lambda x:vs[int32((x+s0)/step)]

def ols_thres(lamb = 3.2, ww = 1.0, beta = 1.0, pos=True):
    ''' We can rewrite O&F energy formulation to find activations
    as a recursive search:
    a_i <= f^-1(y_i - sum{w_i w_j a_j})
    where f(x) = x + lambda S'(x).
    S(x) is the sparsity function. y_i = w_i x.
    for S(x) = log(1+x**2) as in the obfgsriginal paper,
    S' = 2*x/(1+x**2).'''
    
    if pos: # Rectified activation.
        f = inv_func(lambda x:olsf(x,lamb,ww,beta))
        return lambda x:(x>0)*f(x*(x>0))
    else:
        return inv_func(lambda x:olsf(x,lamb,ww,beta))

olsf = lambda x,l=3.2,ww=1.0,beta=1.0:ww*x + l*2*beta*x/(1+(beta*x)**2) # O&F function
th_jump = lambda x,a: x*(x>a) # Threshold and jump function
th_lin = lambda x, a: (x-a)*(x>a) # Rectifier
bcm = lambda y,m,ref=1.0: (y>0)*y*(y-m/ref) # BCM function


### Image processing ###


def load_oef_raw():
    # Load raw images.
    return sio.loadmat(data_path + 'IMAGES_RAW.mat')['IMAGESr'].transpose([2,0,1])

def sample_patches(data, patsize = 16, np = 50000, rotate = True):    
    # Sample random patches from given images.
    nd,nx,ny = data.shape
    pats = zeros([np, patsize, patsize])
    for j in arange(np):
        i = randint(nd)
        
        a = randint(nx-patsize)
        b = randint(ny-patsize)
        pats[j] = data[i,a:a+patsize,b:b+patsize]
        if rotate:
            if rand() > 0.5:
                pats[j] = pats[j,::-1,:]
            if rand() > 0.5:
                pats[j] = pats[j,:,::-1]
            if rand() > 0.5:
                pats[j] = pats[j].T
        
    return pats

def oef_patches(ndata = 10000, patsize = 16):
    # Get random patches.
    im = load_oef_raw()
    p = sample_patches(im, np = ndata, patsize = patsize)
    p = p/std(p)
    
    return p

def general_whitening(data, covp = None):

    if covp is None:
        patt = data.reshape(ndata,-1).T  
        patt = patt-mean(patt,axis=1).reshape(-1,1) # Subtracts pixel mean
        covp = dot(patt,patt.T)/float32(ndata)

    decom = eig(covp)

    # This is the whitening linear tranformation matrix
    whi = dot(decom[1], dot(diag(decom[0]**(-1.0/2)), decom[1].T))
    
    return whi

def oef_whiten_patchfilter(patsize=16,  ndata=10000000):
    ''' Returns set of whtening filters to be applied directly to patches.'''
    logstr = ""
    filestr = log_path + 'oef'+logstr+'_white_filter_np_%d_patsize_%d.pic' % (ndata,patsize)
    try:
        whi = cp.load(open(filestr, 'rb'))
    except IOError as e:
        print(e)
        print("whitening filter not found in ", filestr)
        print('calculating oef whitening filter '+logstr)
        if ndata > 10**6:
            covp = 0.
            niter = int32(ndata//10**6 + 1)
            print("Dividing in %d iterations..."%niter)
            for i in arange(niter):
                print("Iter", i)
                pats = oef_patches(ndata=10**6, patsize = patsize)
                pats = pats.reshape(10**6,-1).T  
                pats = pats-mean(pats,axis=1).reshape(-1,1) #subtracts pixel mean.
                covp += dot(pats,pats.T)/float32(10**6)
            covp /= niter
            whi = general_whitening(pats,covp=covp)
        else:
            pats = oef_patches(ndata=ndata, patsize = patsize)
            whi = general_whitening(pats)

        f = open(filestr,'wb')
        cp.dump(whi,f)
        f.close()     
        
    return whi.reshape(-1,patsize*patsize)

def oef_ideal_white(ndata = 50000, patsize = 16):
    ''' Whithens a set of patches '''
    whi = oef_whiten_patchfilter(patsize=patsize, ndata=10000000)
    
    pats = oef_patches(ndata=ndata, patsize = patsize)
    
    pats = pats.reshape(ndata,-1).T
    pats = pats-mean(pats,axis=1).reshape(-1,1) 
    
    white_pats = dot(whi,pats)
    white_pats = white_pats.T.reshape(-1,patsize,patsize)

    return white_pats, whi


### Functions to test and learn ###

def test_func(data, ws, func, normed = True, zeroed = False):
    ''' Calculates the value <F(wx)>.'''

    # Obs: ws and data must be flatten. ws must have multiple filters.
    if data.ndim > 2:
        data = data.reshape(-1,data.shape[1]*data.shape[2])
    if ws.ndim > 2:
        ws = ws.reshape(-1,ws.shape[1]*ws.shape[2])
    
    dn = dot(data,ws.T)
    if zeroed:
        dn -= mean(dn, 0)
    if normed:
        dn /= std(dn, 0)
    
    # Calculates f(data) and average for each w.
    res = func(dn).mean(0)
            
    return res



### Main ###

# Choose filters
patsize = 16 # Patch size
fs,n1 = filter_bank(psize=patsize) # Get collection of filters
chosen = fs[[5,3,44,0,33]] # Select random, fourier, circle, PC1, gabor filters

# Define activation functions
funcs = [[lambda x,t1=1.,t2=2.: (x>t1)*(x-t1)*(x-t2)/10.,'rectBCM'],
        [lambda x: (x>3.)*(x-3.),'I&F'],
        [ols_thres(), 'O&F'],
        [lambda x:th_jump(x,3.), 'L0'],
         [lambda x:0.2*-tanh(x), 'B&S'],
         ]
names = [f[1] for f in funcs]

def mysave(obj,name):
    f = open(log_path+name,'wb')
    cp.dump(obj,f)
    f.close()
    
def myload(name):
    f = open(log_path+name,'rb')
    r = cp.load(f)
    f.close()
    return r


def calc_winners(pats = None):
    # Calculate <F(wx)> for the selected filters and functions.

    np = 2000000
    patsize=16
    normed = True
    
    if pats is None:
        pats = oef_ideal_white(ndata = np, patsize = patsize)[0]
        pats /= std(pats)

    x = arange(-7,7,0.1)
    res = zeros([len(funcs),chosen.shape[0]])
    for i in arange(len(funcs)):
    
        # Integrates f(x) to F(x).
        ifunc = int_func(funcs[i][0], step=0.05, lim=30.0)
    
        res[i] = test_func(pats.reshape(np,patsize**2), 
                           chosen.reshape(-1,patsize**2), 
                           ifunc, normed = normed, 
                           zeroed = False)

    mysave(res, 'winners_res clean') # Saves results


# Define plotting colours
pan_colors = {"b":'#0047BA',"r":'#E60D2E',"g":'#66D43D',"p":'#70126B',
               "gr":'#878785',"lb":'#009CA3',"y":"#F7E017","br":"#D68F54","k":"#2e271d"}
rainbow_cs = [pan_colors["g"],pan_colors["b"],pan_colors["lb"],pan_colors["br"],pan_colors["p"]]

def removeAxes(ax, which_axis):
    for loc, spine in ax.spines.items():
        if loc in which_axis:
            spine.set_color('none') # don't draw spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


#### For fig 2a ###

def plot_activations():
    
    close(21)
    fig=figure(21) 
    ax = fig.add_subplot(1,1,1)
    plot([-1,4],[0,0],"--k",lw=2)
        
    for i in arange(len(funcs))[::-1]:
        
        x = arange(-0.5,4,0.1)
        fx = funcs[i][0](x)/abs(funcs[i][0](x)).max()
        if funcs[i][0](x[-1]) < 0: # for B&S f
            fx = 0.15*funcs[i][0](x)/abs(funcs[i][0](x)).max()
        
        removeAxes(gca(), ['right', 'top'])

        plot(x,4*fx, linewidth=4, label=funcs[i][1], c=rainbow_cs[i])
        yticks([0,2,4])
        xticks([0,2,4])
        xlim([-0.5,4])
        ylim([-1.0,4.5])

        ax.spines['left'].set_bounds(-1,4)
        ax.spines['bottom'].set_bounds(-0.5,4.)


#### For fig 2b ###

def plot_winners(res=None):
    
    res = myload('winners_res clean') # Loads results
    
    close(29); figure(29) 
    
    for i in arange(res.shape[0]):
        
        nres = (res[i]-res[i].min())/(res[i].max()-res[i].min()) # Normalized values.
    
        yoffset = 0.1
        width = 0.12
        bar(arange(5)+(i-2.5)*width, yoffset+nres, width, color=rainbow_cs[i],bottom=-yoffset,edgecolor = "none")

        xlim(-0.5,4.5)
        xticks(arange(5), [])
        yticks([0,1])
        ylim([-yoffset,1.1])
        gca().spines['left'].set_bounds(-yoffset,1)
    
#### For fig 2c, learn filters for different functions ###

transfs = [lambda u: th_lin(u,3.),
        lambda u: bcm(th_lin(u,0.5),0.5)/10.,
        lambda u: th_jump(u,3.),
        ols_thres(),
        lambda u: 0.5*-tanh(u),

        lambda u: (u**3)/1e2,
        lambda u: -sin(u),
        lambda u: u,
        lambda u: (u-2.)*(u>2.) + (u < -2.)*(-u-2),
        lambda u: -cos(u),

        lambda u: -th_lin(u,3.),
        lambda u: -bcm(th_lin(u,0.5),0.5)/10.,
        lambda u: -th_jump(u,3.),
        lambda u: -ols_thres()(u),
        lambda u: 0.5*tanh(u),

        lambda u: -(u**3)/1e2,
        lambda u: sin(u),
        lambda u: -u,
        lambda u: -(u-2.)*(u>2.) - (u < -2.)*(-u-2),
        lambda u: cos(u)]

def trial(data=None, iis = None, eta = None): 
  # Run simulation for each f.

  ndata = 1000000

  if iis is None: # Choose some functions only
    iis = arange(len(transfs))

  if data is None: # Load data separately
    data = oef_ideal_white(ndata=ndata,patsize=patsize)[0]
    data = data.reshape(data.shape[0], -1)
    data = data/std(data)
  
  nb = 100 # Minibatch size
  n_iter = 10000000 # Number of iterations

  nd, nx = data.shape
  ny = 9

  for i in iis:
  
    transf = transfs[i]

    if eta is None:
      eta = 0.0002 # Learning rate. May need tuning to each activation function.

    total_iter = 0
    print_iter = 1000000
      
    w = randn(ny,nx)/sqrt(nx) # Define weights
    
    print("Nr. f:", i)
    print('Learning', n_iter, 'iterations.')
    print("Input dimension:", nx)
    print("Output dimension:", ny)
    
    permii = permutation(nd)
    for ii in tqdm(arange(n_iter/nb)):
        t = ii*nb
        iis = int32(arange(ii*nb,ii*nb+nb)%nd)
        x = data[permii[iis]].T

        # Inference
        u = dot(w,x)
        y = transf(u)
            
        batch_dw = dot(y,x.T)  # Plasticity
        w = w + eta*batch_dw # Update

        for j in arange(ny):
            w[j] = w[j]/norm(w[j],2) # Normalize to norm 1
        
        total_iter += nb

        if t%nd == 0:
          print("permuting new batch. t:", t)
          permii = permutation(nd)
    
    print('Finished', n_iter, 'iterations.')

    mysave(w,"nonlinear manyfs simu w %d" % i)


def plot_2c():
    ''' Plot results '''
    ai = array([1,0,3,2,4])
    fig_factor = 4/2.54
    figw = 3.*fig_factor
    figh = 6.*fig_factor

    close(20)
    fig=figure(20,figsize=(figw,figh))
    nf = len(transfs)
    
    nlin = 10
    ncol = 2
    for i in arange(len(transfs)):
        icol = i//nlin
        ilin = i%nlin

        # Plot activation functions.

        ax=subplot(nlin,2*ncol,ilin*2*ncol+icol*2+1)
        ax.set_aspect(1.2, adjustable='box')
        removeAxes(gca(), ['right', 'top'])
        
        x = arange(-6,6,0.1)
        plot([-6,6],[0,0],"--k",lw=1)
        plot([0,0],[-4.5,4.5],"--k",lw=1)
        
        fx = transfs[i](x)/abs(transfs[i](x)).max() 
            
        if i in [0,1,2,3,4]:
          plot(x,4*fx, linewidth=2,c=rainbow_cs[ai[i]])
        else:
          plot(x,4*fx, linewidth=2,c=rainbow_cs[1])
        yticks([])
        xticks([])
        xlim([-6,6])
        ylim([-4.5,4.5])

        # Plot learned filters

        ax=subplot(nlin,2*ncol,ilin*2*ncol+icol*2+2)
        frame1 = gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])

        res = array(myload("nonlinear manyfs simu w %d" % i))
        
        w = res[0].reshape(16,16)
        lim_val = abs(w).max()

        imshow(w,vmin=-lim_val,vmax=lim_val, interpolation = 'nearest')
        removeAxes(gca(), ['right', 'top', 'bottom', 'left'])

    return fig
        

