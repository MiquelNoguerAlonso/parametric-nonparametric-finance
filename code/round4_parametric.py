"""Training-only joint Student likelihood and Ledoit-Wolf Gaussian controls."""
import json
import numpy as np
from scipy import stats,optimize,special,linalg
from sklearn.covariance import LedoitWolf

class JointStudent:
    def fit(self,x,maxiter=1200,tol=1e-8):
        x=np.asarray(x,dtype=float);n,d=x.shape;records=[]
        cov=np.cov(x,rowvar=False);center=x.mean(0)
        for start in [3.,8.,30.]:
            mu=center.copy();scatter=cov*(start-2)/start;nu=start;previous=-np.inf;trace=[]
            for it in range(maxiter):
                chol=np.linalg.cholesky(scatter);r=x-mu
                delta=np.square(linalg.solve_triangular(chol,r.T,lower=True,check_finite=False)).sum(0)
                weight=(nu+d)/(nu+delta)
                newmu=(weight[:,None]*x).sum(0)/weight.sum();r=x-newmu
                newscatter=(r.T*weight)@r/n;newscatter=(newscatter+newscatter.T)/2
                chol=np.linalg.cholesky(newscatter)
                delta=np.square(linalg.solve_triangular(chol,r.T,lower=True,check_finite=False)).sum(0)
                logdet=2*np.log(np.diag(chol)).sum()
                def nll(v):
                    return float(-special.gammaln((v+d)/2)+special.gammaln(v/2)+.5*d*np.log(v*np.pi)+.5*logdet+(v+d)/2*np.log1p(delta/v).mean())
                q=optimize.minimize_scalar(nll,bounds=(2.05,200.),method='bounded',options={'xatol':1e-7})
                candidates=[(nu,nll(nu)),(2.05,nll(2.05)),(200.,nll(200.)),(float(q.x),float(q.fun))]
                newnu,value=min(candidates,key=lambda a:a[1]);ll=-value
                if ll<previous-1e-7:raise RuntimeError('Student likelihood decreased')
                change=max(np.max(abs(newmu-mu)),np.max(abs(newscatter-scatter)),abs(newnu-nu)/(1+nu))
                trace.append(ll);mu,scatter,nu=newmu,newscatter,newnu
                if it>5 and abs(ll-previous)<tol*(1+abs(ll)) and change<1e-5:break
                previous=ll
            records.append(dict(start=start,mu=mu,scatter=scatter,nu=nu,loglik=ll,iterations=it+1,converged=it+1<maxiter,trace=trace))
        best=max(records,key=lambda r:r['loglik'])
        if not best['converged']:raise RuntimeError('Joint Student did not converge')
        self.mu=best['mu'];self.scatter=best['scatter'];self.nu=best['nu'];self.d=d
        self.diagnostics=[{k:v for k,v in r.items() if k not in ['mu','scatter','trace']} for r in records]
        self.trace=best['trace'];return self

    def logpdf(self,x):return stats.multivariate_t.logpdf(x,loc=self.mu,shape=self.scatter,df=self.nu)
    def sample(self,n,seed):
        rng=np.random.default_rng(seed);z=rng.standard_normal((n,self.d))@np.linalg.cholesky(self.scatter).T
        return self.mu+z*np.sqrt(self.nu/rng.chisquare(self.nu,(n,1)))
    def risk(self,projections,alpha):
        p=np.atleast_2d(projections);mu=-p@self.mu;scale=np.sqrt(np.einsum('ij,jk,ik->i',p,self.scatter,p))
        q=stats.t.isf(alpha,self.nu);e=(self.nu+q*q)/(self.nu-1)*stats.t.pdf(q,self.nu)/alpha
        return mu+scale*q,mu+scale*e
    def save(self,path):
        path.write_text(json.dumps(dict(kind='student',mu=self.mu.tolist(),scatter=self.scatter.tolist(),nu=self.nu,diagnostics=self.diagnostics,trace=self.trace),indent=2))
    @classmethod
    def load(cls,path):
        z=json.loads(path.read_text());o=cls();o.mu=np.array(z['mu']);o.scatter=np.array(z['scatter']);o.nu=z['nu'];o.d=len(o.mu);o.diagnostics=z['diagnostics'];o.trace=z['trace'];return o

class ShrinkGaussian:
    def fit(self,x):
        q=LedoitWolf().fit(x);self.mu=q.location_;self.scatter=q.covariance_;self.shrinkage=float(q.shrinkage_);self.d=len(self.mu);return self
    def logpdf(self,x):return stats.multivariate_normal.logpdf(x,self.mu,self.scatter)
    def sample(self,n,seed):return np.random.default_rng(seed).multivariate_normal(self.mu,self.scatter,n)
    def risk(self,projections,alpha):
        p=np.atleast_2d(projections);mu=-p@self.mu;sd=np.sqrt(np.einsum('ij,jk,ik->i',p,self.scatter,p));q=stats.norm.isf(alpha)
        return mu+sd*q,mu+sd*stats.norm.pdf(q)/alpha
    def save(self,path):path.write_text(json.dumps(dict(kind='shrink-gaussian',mu=self.mu.tolist(),scatter=self.scatter.tolist(),shrinkage=self.shrinkage),indent=2))
    @classmethod
    def load(cls,path):
        z=json.loads(path.read_text());o=cls();o.mu=np.array(z['mu']);o.scatter=np.array(z['scatter']);o.shrinkage=z['shrinkage'];o.d=len(o.mu);return o
