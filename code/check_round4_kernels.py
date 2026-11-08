"""Independent distribution, optimizer, and dense nesting checks."""
import json,copy
import numpy as np,jax,jax.numpy as jnp
from scipy import stats,linalg
from round4_parametric import JointStudent,ShrinkGaussian
from round4_repair import nested_parameters
from round3_models import DenseFlow
from models import mlp_init
from round4_data import ROOT,OUT,get_data

rng=np.random.default_rng(486134);x=stats.multivariate_t.rvs(loc=[.2,-.1,.4],shape=[[1,.4,.1],[.4,2,.2],[.1,.2,.8]],df=5,size=4000,random_state=rng)
m=JointStudent().fit(x);checks={}
eps=1e-4
checks['df_likelihood_score']=float((stats.multivariate_t.logpdf(x,loc=m.mu,shape=m.scatter,df=m.nu+eps).mean()-stats.multivariate_t.logpdf(x,loc=m.mu,shape=m.scatter,df=m.nu-eps).mean())/(2*eps))
chol=np.linalg.cholesky(m.scatter);r=x-m.mu;delta=np.square(linalg.solve_triangular(chol,r.T,lower=True)).sum(0);w=(m.nu+3)/(m.nu+delta)
checks['location_score']=float(np.max(abs((r*w[:,None]).mean(0))))
checks['scatter_fixed_point']=float(np.max(abs((r.T*w)@r/len(x)-m.scatter)))
checks['likelihood_monotone']=bool(np.min(np.diff(m.trace))>=-1e-7)
p=np.array([.2,.3,.5]);q,e=m.risk(p,.01);parts=[-m.sample(50000,k)@p for k in range(20)]
from experiment import es
v=np.array([es(a,.01) for a in parts]);checks['analytic_es_mc_z']=float((v.mean()-e[0])/(v.std(ddof=1)/np.sqrt(len(v))))
assert checks['location_score']<1e-5 and checks['scatter_fixed_point']<1e-5 and abs(checks['df_likelihood_score'])<1e-5 and checks['likelihood_monotone'] and abs(checks['analytic_es_mc_z'])<5
data=get_data('crsp100',2016);n=DenseFlow('flow',100,data['seed'],base='gaussian');n.loc=data['X'].mean(0);n.scale=data['X'].std(0)
pp={'a':jnp.zeros(100),'b':jnp.zeros(100),'nu_raw':jnp.array(2.9),'layers':[mlp_init(n.nextkey(),[len(c),64,64,2*len(d)]) for c,d in n.idx]}
pp=n.initialize_flow_output(pp,jnp.asarray((data['X']-n.loc)/n.scale));n.p=nested_parameters(n,pp,data,'Flow-G','lu','crsp100',2016)
expected=stats.multivariate_normal.logpdf(data['V'],data['X'].mean(0),np.cov(data['X'],rowvar=False)+1e-7*np.eye(100));checks['dense_gaussian_nested_logpdf_error']=float(np.max(abs(n.logpdf(data['V'])-expected)))
assert checks['dense_gaussian_nested_logpdf_error']<1e-3
st=JointStudent.load(OUT/'checkpoints/original_crsp100_2016_Student-MLE.json');n.base='t-elliptical';n.p=nested_parameters(n,pp,data,'Flow-t','lu','crsp100',2016)
checks['dense_student_nested_logpdf_error']=float(np.max(abs(n.logpdf(data['V'])-st.logpdf(data['V']))));assert checks['dense_student_nested_logpdf_error']<1e-3
print(json.dumps(checks,indent=2));(OUT/'analysis/kernel_checks.json').write_text(json.dumps(checks,indent=2))
