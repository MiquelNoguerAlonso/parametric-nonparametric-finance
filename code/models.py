"""Auditable JAX generators. No data access or training occurs on import.

All t-elliptical samplers use one chi-square variate per observation;
t-independent is an explicitly named ablation. Fitted models accept/return
data in original units. Checkpoints are plain NPZ arrays plus JSON trees.
"""
import json, time
from pathlib import Path
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar, minimize
import jax
import jax.numpy as jnp
from jax import tree_util as tu
from jax.scipy.special import gammaln, logsumexp
from jax.scipy.linalg import solve_triangular


def mlp_init(k, sizes, last=.05):
    keys=jax.random.split(k,len(sizes)-1)
    return [(jax.random.normal(q,(a,b))*(last if i==len(keys)-1 else 1)/np.sqrt(a),jnp.zeros(b))
            for i,(q,a,b) in enumerate(zip(keys,sizes[:-1],sizes[1:]))]

def mlp(p,x,act=jnp.tanh):
    for w,b in p[:-1]: x=act(x@w+b)
    return x@p[-1][0]+p[-1][1]

def adam_init(p): return (tu.tree_map(jnp.zeros_like,p),tu.tree_map(jnp.zeros_like,p),jnp.array(0))

def adam(p,g,state,lr,b1=.9,b2=.999):
    m,v,t=state;t=t+1
    # Global clipping prevents an isolated t observation from destroying a fit.
    gn=jnp.sqrt(sum(jnp.sum(a*a) for a in tu.tree_leaves(g)))
    g=tu.tree_map(lambda a:a*jnp.minimum(1.,100./(gn+1e-8)),g)
    m=tu.tree_map(lambda a,b:b1*a+(1-b1)*b,m,g)
    v=tu.tree_map(lambda a,b:b2*a+(1-b2)*b*b,v,g)
    p=tu.tree_map(lambda q,a,b:q-lr*(a/(1-b1**t))/(jnp.sqrt(b/(1-b2**t))+1e-8),p,m,v)
    return p,(m,v,t)

def log_normal(z): return -.5*jnp.sum(z*z+jnp.log(2*jnp.pi),axis=-1)

def log_t(z,nu,independent=False):
    if independent:
        return jnp.sum(gammaln((nu+1)/2)-gammaln(nu/2)-.5*jnp.log(nu*jnp.pi)
                       -(nu+1)/2*jnp.log1p(z*z/nu),axis=-1)
    d=z.shape[-1]
    return (gammaln((nu+d)/2)-gammaln(nu/2)-d/2*jnp.log(nu*jnp.pi)
            -(nu+d)/2*jnp.log1p(jnp.sum(z*z,axis=-1)/nu))

def noise(k,n,d,kind='gaussian',nu=3.):
    a,b=jax.random.split(k)
    if kind=='gaussian':return jax.random.normal(a,(n,d))
    if kind=='laplace':return jax.random.laplace(a,(n,d))/jnp.sqrt(2.)
    shape=(n,d) if kind=='t-independent' else (n,1)
    w=2*jax.random.gamma(b,nu/2,shape)
    return jax.random.normal(a,(n,d))/jnp.sqrt(w/nu)

def save_tree(path,p,metadata):
    arrays={}
    def enc(v):
        if isinstance(v,dict):return {'dict':{k:enc(x) for k,x in v.items()}}
        if isinstance(v,(list,tuple)):return {'tuple' if isinstance(v,tuple) else 'list':[enc(x) for x in v]}
        key=f'a{len(arrays)}';arrays[key]=np.asarray(v);return {'array':key}
    tree=enc(p);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    path.with_suffix('.json').write_text(json.dumps({'tree':tree,'metadata':metadata},indent=2))

def load_tree(path):
    a=np.load(path);s=json.loads(Path(path).with_suffix('.json').read_text())
    def dec(v):
        if 'array' in v:return jnp.asarray(a[v['array']])
        if 'dict' in v:return {k:dec(x) for k,x in v['dict'].items()}
        if 'tuple' in v:return tuple(dec(x) for x in v['tuple'])
        return [dec(x) for x in v['list']]
    return dec(s['tree']),s['metadata']

def tmap(y,nufrom,nuto):
    y=np.asarray(y,dtype=float)
    return np.sign(y)*st.t.isf(st.t.sf(np.abs(y),nufrom),nuto)

class Neural:
    def __init__(self,kind,d,seed=0,base='gaussian',fixed_nu=None,P=None,steps=None):
        self.kind,self.d,self.seed,self.base=kind,d,seed,base
        self.fixed_nu=fixed_nu;self.P=np.eye(d) if P is None else np.asarray(P)
        self.C=jnp.asarray(np.linalg.cholesky(self.P));self.steps=steps
        self.k=jax.random.PRNGKey(seed);self.trace=[]
        self.latent=min(6,d);self.width=32 if d<=5 else 64
        ids=np.arange(d)
        self.idx=[(np.where(ids%2==v)[0],np.where(ids%2!=v)[0]) for v in (0,1)]*(2 if d<=5 else 3)

    def nextkey(self):self.k,k=jax.random.split(self.k);return k

    def flow_forward(self,p,z,inverse=False):
        if inverse:
            x=(z-p['b'])*jnp.exp(-p['a']);ld=-p['a'].sum()*jnp.ones(len(z))
            seq=zip(self.idx,p['layers'])
        else:
            x=z;ld=jnp.zeros(len(z));seq=reversed(list(zip(self.idx,p['layers'])))
        for (c,d),net in seq:
            o=mlp(net,x[:,c]);s=jnp.tanh(o[:,:len(d)]);t=o[:,len(d):]
            if inverse:x=x.at[:,d].set((x[:,d]-t)*jnp.exp(-s));ld=ld-s.sum(1)
            else:x=x.at[:,d].set(x[:,d]*jnp.exp(s)+t);ld=ld+s.sum(1)
        if not inverse:x=x*jnp.exp(p['a'])+p['b'];ld=ld+p['a'].sum()
        return x,ld

    def nu(self,p):return self.fixed_nu if self.fixed_nu is not None else 1.05+jax.nn.softplus(p['nu_raw'])

    def flow_logpdf(self,p,x):
        z,ld=self.flow_forward(p,x,True)
        u=solve_triangular(self.C,z.T,lower=True).T
        if self.base=='gaussian':lp=log_normal(u)
        elif self.base=='laplace':lp=jnp.sum(-jnp.sqrt(2.)*jnp.abs(u)-jnp.log(jnp.sqrt(2.)),1)
        else:lp=log_t(u,self.nu(p),self.base=='t-independent')
        return lp-jnp.log(jnp.diag(self.C)).sum()+ld

    def chol(self,p):
        ii=np.tril_indices(self.d,-1)
        return jnp.zeros((self.d,self.d)).at[ii].set(p['off'])+jnp.diag(jnp.exp(p['diag']))

    def decode(self,p,z):
        o=mlp(p['dec'],z);mu=o[...,:self.d];sd=jax.nn.softplus(o[...,self.d:])+.001
        if self.kind=='vae-scale':
            # Strictly positive coefficient guarantees linear scale growth on a cone.
            sd=sd+(.01+jax.nn.softplus(p['amp']))*jnp.abs(z[...,np.arange(self.d)%self.latent])
        return mu,sd

    def vae_logw(self,p,x,eps):
        h=mlp(p['enc'],x);mu,lv=h[...,:self.latent],h[...,self.latent:]
        lv=jnp.clip(lv,-12.,12.);z=mu+jnp.exp(.5*lv)*eps
        m,sd=self.decode(p,z);L=self.chol(p)
        u=solve_triangular(L,((x-m)/sd).T,lower=True).T
        lp=log_normal(u)-jnp.log(sd).sum(-1)-p['diag'].sum()
        prior=log_normal(z) if self.base=='gaussian' else log_t(z,3.)
        posterior=log_normal(eps)-.5*lv.sum(-1)
        return lp+prior-posterior

    def sample_std(self,p,k,n):
        a,b=jax.random.split(k)
        if self.kind=='flow':
            z=noise(a,n,self.d,self.base,self.nu(p))@self.C.T
            return self.flow_forward(p,z)[0]
        if self.kind.startswith('vae'):
            z=noise(a,n,self.latent,self.base,3.);m,sd=self.decode(p,z)
            return m+sd*(jax.random.normal(b,(n,self.d))@self.chol(p).T)
        return mlp(p,noise(a,n,self.d,self.base,3.),jax.nn.leaky_relu)

    def fit(self,X,V,checkpoint=None,risk_validation=None,checkpoint_observer=None,
            checkpoint_steps=None,allow_invalid_prefix=False):
        start=time.time();X=np.asarray(X);V=np.asarray(V)
        self.checkpoint_steps=None if checkpoint_steps is None else set(checkpoint_steps)
        self.allow_invalid_prefix=allow_invalid_prefix
        self.loc=X.mean(0);self.scale=np.maximum(X.std(0),1e-5)
        # On the meta-t copula scale, preserve the specified identity base.
        if not np.array_equal(self.P,np.eye(self.d)):
            self.loc=np.zeros(self.d);self.scale=np.ones(self.d)
        x=jnp.asarray((X-self.loc)/self.scale,dtype=jnp.float32)
        v=jnp.asarray((V-self.loc)/self.scale,dtype=jnp.float32)
        w=self.width
        if self.kind=='flow':
            p={'a':jnp.zeros(self.d),'b':jnp.zeros(self.d),'nu_raw':jnp.array(2.9),
               'layers':[mlp_init(self.nextkey(),[len(c),w,w,2*len(d)]) for c,d in self.idx]}
            if hasattr(self,'initialize_flow_output'):p=self.initialize_flow_output(p,x)
            steps=self.steps or 3000;lr=.0015
            loss=lambda p,x,k:-self.flow_logpdf(p,x).mean()
            valid=lambda p:float(-self.flow_logpdf(p,v).mean())
        elif self.kind.startswith('vae'):
            p={'enc':mlp_init(self.nextkey(),[self.d,w,w,2*self.latent]),
               'dec':mlp_init(self.nextkey(),[self.latent,w,w,2*self.d]),
               'off':jnp.zeros(self.d*(self.d-1)//2),'diag':jnp.zeros(self.d)}
            if self.kind=='vae-scale':p['amp']=jnp.full(self.d,-3.)
            steps=self.steps or 4000;lr=.001
            loss=lambda p,x,k:-self.vae_logw(p,x,jax.random.normal(k,(len(x),self.latent))).mean()
            kval=self.nextkey()
            @jax.jit
            def vfn(p):
                ep=jax.random.normal(kval,(8,len(v),self.latent))
                lw=jax.vmap(lambda e:self.vae_logw(p,v,e))(ep)
                return -(logsumexp(lw,0)-jnp.log(8.)).mean()
            valid=lambda p:float(vfn(p))
        else:
            self._fit_gan(x,v,risk_validation,checkpoint_observer)
            self.fit_seconds=time.time()-start
            if checkpoint:self.save(checkpoint)
            return self
        state=adam_init(p)
        @jax.jit
        def step(p,state,k):
            a,b=jax.random.split(k);batch=x[jax.random.randint(a,(256,),0,len(x))]
            return adam(p,jax.grad(loss)(p,batch,b),state,lr)
        if self.kind=='flow':valid_jit=jax.jit(lambda p:-self.flow_logpdf(p,v).mean());valid=lambda p:float(valid_jit(p))
        native_valid=valid
        if risk_validation is not None:
            risk_draw=jax.jit(lambda p:self.sample_std(p,jax.random.PRNGKey(424242),20000))
            valid=lambda p:risk_validation(np.asarray(risk_draw(p))*self.scale+self.loc)
        best=np.inf;pbest=p;beststep=0
        if self.checkpoint_steps is not None and 0 in self.checkpoint_steps:
            value=valid(p);self.trace.append([0,value])
            if checkpoint_observer is not None:checkpoint_observer(self,p,0,value,native_valid(p))
            if np.isfinite(value):best,pbest=value,p
        for it in range(steps):
            p,state=step(p,state,self.nextkey())
            if ((it+1)%250==0 if self.checkpoint_steps is None else it+1 in self.checkpoint_steps):
                value=valid(p);self.trace.append([it+1,value])
                if checkpoint_observer is not None:checkpoint_observer(self,p,it+1,value,native_valid(p))
                if np.isfinite(value) and value<best:best,pbest,beststep=value,p,it+1
        self.last_p=p
        if not np.isfinite(best) and not self.allow_invalid_prefix:raise RuntimeError('No finite validation checkpoint')
        self.p=pbest;self.best_step=beststep;self.fit_seconds=time.time()-start
        if checkpoint:self.save(checkpoint)
        return self

    def _fit_gan(self,x,v,risk_validation=None,checkpoint_observer=None):
        w=64 if self.d<=5 else 128
        p=mlp_init(self.nextkey(),[self.d,w,w,self.d],last=1.)
        c=mlp_init(self.nextkey(),[self.d,w,w,1],last=1.)
        sp,sc=adam_init(p),adam_init(c);ema=p
        act=lambda a:jax.nn.leaky_relu(a,negative_slope=.01)
        G=lambda p,z:mlp(p,z,act);C=lambda c,z:mlp(c,z,act)[...,0]
        def closs(c,p,b,z,u):
            fake=G(p,z);mix=u*b+(1-u)*fake
            g=jax.vmap(jax.grad(lambda a:C(c,a[None])[0]))(mix)
            return C(c,fake).mean()-C(c,b).mean()+10*jnp.mean((jnp.linalg.norm(g,axis=1)-1)**2)
        @jax.jit
        def cs(c,sc,p,k):
            a,b,e=jax.random.split(k,3);batch=x[jax.random.randint(a,(256,),0,len(x))]
            grad=jax.grad(closs)(c,p,batch,noise(b,256,self.d,self.base),jax.random.uniform(e,(256,1)))
            return adam(c,grad,sc,.0002,.5,.9)
        @jax.jit
        def gs(p,sp,c,k):
            grad=jax.grad(lambda q:-C(c,G(q,noise(k,256,self.d,self.base))).mean())(p)
            return adam(p,grad,sp,.0002,.5,.9)
        kv=self.nextkey();zv=noise(kv,4096,self.d,self.base)
        # Fixed projection discrepancy on validation data only, not tail truth.
        proj=np.concatenate([np.eye(self.d),np.ones((1,self.d))/np.sqrt(self.d)],axis=0)
        qs=np.linspace(.01,.99,99);vq=np.quantile(np.asarray(v)@proj.T,qs,axis=0)
        gen_eval=jax.jit(lambda q:G(q,zv));best=np.inf;beststep=0;pbest=p
        if risk_validation is not None:
            risk_draw=jax.jit(lambda p:self.sample_std(p,jax.random.PRNGKey(424242),20000))
        if self.checkpoint_steps is not None and 0 in self.checkpoint_steps:
            value=(risk_validation(np.asarray(risk_draw(ema))*self.scale+self.loc)
                   if risk_validation is not None else float(np.mean(np.abs(np.quantile(np.asarray(gen_eval(ema))@proj.T,qs,axis=0)-vq))))
            self.trace.append([0,value])
            if checkpoint_observer is not None:checkpoint_observer(self,ema,0,value,None)
            if np.isfinite(value):best,pbest=value,ema
        for it in range(self.steps or 3000):
            for _ in range(5):c,sc=cs(c,sc,p,self.nextkey())
            p,sp=gs(p,sp,c,self.nextkey());ema=tu.tree_map(lambda a,b:.995*a+.005*b,ema,p)
            if ((it+1)%250==0 if self.checkpoint_steps is None else it+1 in self.checkpoint_steps):
                value=(risk_validation(np.asarray(risk_draw(ema))*self.scale+self.loc)
                       if risk_validation is not None else float(np.mean(np.abs(np.quantile(np.asarray(gen_eval(ema))@proj.T,qs,axis=0)-vq))))
                self.trace.append([it+1,value])
                if checkpoint_observer is not None:checkpoint_observer(self,ema,it+1,value,None)
                if np.isfinite(value) and value<best:best,pbest,beststep=value,ema,it+1
        self.last_p=ema
        if not np.isfinite(best) and not self.allow_invalid_prefix:raise RuntimeError('No finite GAN checkpoint')
        self.p=pbest;self.best_step=beststep
        self.lipschitz_bound=float(np.prod([np.linalg.norm(np.asarray(a),2) for a,b in self.p]))

    def sample(self,n=100000,seed=12345):
        # Batched evaluation avoids retaining an enormous computation graph.
        key=jax.random.PRNGKey(seed);out=[]
        fn=jax.jit(lambda k:self.sample_std(self.p,k,10000))
        for _ in range((n+9999)//10000):key,k=jax.random.split(key);out.append(np.asarray(fn(k)))
        return np.concatenate(out)[:n]*self.scale+self.loc

    def logpdf(self,X,K=64,seed=2026):
        if self.kind=='gan':return None
        x=np.asarray((X-self.loc)/self.scale,dtype=np.float32);out=[]
        if self.kind=='flow':fn=jax.jit(lambda b:self.flow_logpdf(self.p,b))
        else:
            @jax.jit
            def fn(b,k):
                ep=jax.random.normal(k,(K,len(b),self.latent))
                lw=jax.vmap(lambda e:self.vae_logw(self.p,b,e))(ep)
                return logsumexp(lw,0)-jnp.log(float(K))
        key=jax.random.PRNGKey(seed)
        for start in range(0,len(x),256):
            key,k=jax.random.split(key);b=jnp.asarray(x[start:start+256])
            out.append(np.asarray(fn(b) if self.kind=='flow' else fn(b,k)))
        return np.concatenate(out)-np.log(self.scale).sum()

    def save(self,path):
        md={k:getattr(self,k) for k in ['kind','d','seed','base','fixed_nu','steps','best_step','fit_seconds','trace']}
        md.update(P=self.P.tolist(),loc=self.loc.tolist(),scale=self.scale.tolist())
        if hasattr(self,'lipschitz_bound'):md['lipschitz_bound']=self.lipschitz_bound
        save_tree(path,self.p,md)

    @classmethod
    def load(cls,path):
        p,m=load_tree(path);obj=cls(**{k:m[k] for k in ['kind','d','seed','base','fixed_nu','P','steps']})
        obj.p=p;obj.loc=np.array(m['loc']);obj.scale=np.array(m['scale'])
        for k in ['best_step','fit_seconds','trace']:setattr(obj,k,m[k])
        if 'lipschitz_bound' in m:obj.lipschitz_bound=m['lipschitz_bound']
        return obj

class MetaT:
    def fit(self,X):
        # Explicit bounded numerical MLE, rather than clipping a fitted df.
        # Existing saved checkpoints are intentionally not refitted on load.
        marginal=[]
        for i in range(X.shape[1]):
            initial=np.array(st.t.fit(X[:,i]));initial[0]=np.clip(initial[0],1.05,200.)
            p0=[initial[0],initial[1],np.log(initial[2])]
            def objective(p):return -np.mean(st.t.logpdf(X[:,i],p[0],loc=p[1],scale=np.exp(p[2])))
            fit=minimize(objective,p0,method='L-BFGS-B',bounds=[(1.05,200.),(None,None),(-20.,20.)],options={'maxiter':2000,'ftol':1e-12})
            if not fit.success or not np.isfinite(fit.fun):raise RuntimeError('Constrained Student fit failed: '+str(fit.message))
            marginal.append([fit.x[0],fit.x[1],np.exp(fit.x[2])])
        self.marg=np.array(marginal)
        self.d=X.shape[1];P=np.eye(self.d)
        for i in range(self.d):
            for j in range(i):P[i,j]=P[j,i]=np.sin(np.pi*st.kendalltau(X[:,i],X[:,j]).statistic/2)
        eig,vec=np.linalg.eigh(P);P=(vec*np.maximum(eig,1e-4))@vec.T
        self.P=P/np.sqrt(np.diag(P)[:,None]*np.diag(P)[None,:])
        z=(X-self.marg[:,1])/self.marg[:,2]
        def obj(nu):
            y=tmap(z,self.marg[:,0],nu)
            return -np.mean(st.multivariate_t.logpdf(y,shape=self.P,df=nu)-st.t.logpdf(y,nu).sum(1))
        grid=[2.1,3.,4.,6.,8.,12.,20.,50.]
        vals=[obj(v) for v in grid];j=int(np.argmin(vals))
        fit=minimize_scalar(obj,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded',options={'xatol':.02})
        self.nu=float(fit.x);return self

    def to_y(self,X):return tmap((X-self.marg[:,1])/self.marg[:,2],self.marg[:,0],self.nu)
    def from_y(self,Y):return self.marg[:,1]+self.marg[:,2]*tmap(Y,self.nu,self.marg[:,0])
    def logjac(self,X,Y):return (st.t.logpdf((X-self.marg[:,1])/self.marg[:,2],self.marg[:,0])-np.log(self.marg[:,2])-st.t.logpdf(Y,self.nu)).sum(1)
    def sample(self,n=100000,seed=12345):
        r=np.random.default_rng(seed);Y=r.standard_normal((n,self.d))@np.linalg.cholesky(self.P).T
        Y*=np.sqrt(self.nu/r.chisquare(self.nu,(n,1)));return self.from_y(Y)
    def logpdf(self,X):
        Y=self.to_y(X);return st.multivariate_t.logpdf(Y,shape=self.P,df=self.nu)+self.logjac(X,Y)
    def save(self,path):Path(path).write_text(json.dumps({'marg':self.marg.tolist(),'P':self.P.tolist(),'nu':self.nu},indent=2))
    @classmethod
    def load(cls,path):
        v=json.loads(Path(path).read_text());o=cls();o.marg=np.array(v['marg']);o.P=np.array(v['P']);o.nu=v['nu'];o.d=len(o.marg);return o
