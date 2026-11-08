"""Independent reconstruction and scope checks for the Wishart revision."""
import argparse,hashlib,json,zipfile,io
from pathlib import Path
import numpy as np,pandas as pd
from run_wishart_synthetic_completion import risk
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'wishart_completion';A=W/'analysis'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline-archive',type=Path,required=True);ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 checks={};maximum=0.;risk_error=0.;parameter_count=0
 with zipfile.ZipFile(args.baseline_archive) as z:
  for p in (ROOT/'wishart_extension/fits').glob('*.json'):
   assert p.read_bytes()==z.read(str(p.relative_to(ROOT)));parameter_count+=1
  assert (ROOT/'data/crsp_size_portfolios.csv').read_bytes()==z.read('data/crsp_size_portfolios.csv')
  old=pd.read_csv(io.BytesIO(z.read('referee_round4/analysis/synthetic_summary.csv'))).set_index(['study','model']);new=pd.read_csv(ROOT/'referee_round4/analysis/synthetic_summary.csv').set_index(['study','model'])
  mask=old.index!=('heston','Heston-Wishart');assert np.allclose(old[mask].values,new.loc[old.index[mask]].values,atol=1e-12,rtol=0)
 manifest=json.loads((W/'primary/manifest.json').read_text());assert manifest['complete'] and manifest['fits']==10 and parameter_count==20
 for unit in range(10):
  p=W/'primary'/f'heston_{unit}.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['files'][p.name];x=json.loads(p.read_text());path=W/'results'/f'synthetic_d{unit}_b0_n131072_s32_grid32.npz';a=np.load(path)
  for m in x['metrics']:
   e,inf=risk(a['loc'],a['sd'],m['alpha']);risk_error=max(risk_error,abs(e['es']-m['es']),abs(e['var']-m['var']),abs(e['mc_se']-m['scenario_mc_se']));assert abs(m['fit_error']+m['sample_error']-m['total_error'])<1e-10
 assert risk_error<1e-9
 for p,h in json.loads((W/'baseline/primary_control_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
 num_checks=['kernel_checks','defensive_checks','block_checks','moment_aux_checks','flattened_auxiliary_checks','gauge_checks','path_gauge_checks']
 for name in num_checks:
  x=json.loads((A/(name+'.json')).read_text());assert x['passed'],name
 count=0
 for p in (W/'results').glob('y*.json'):
  x=json.loads(p.read_text());f=pd.read_csv(ROOT/x['forecasts']);v=np.log(f.es)+f['var']/f.es-1+np.maximum(f.loss-f['var'],0)/(f.alpha*f.es);maximum=max(maximum,float(np.max(abs(v-f.fz0))));count+=len(f)
  assert (f.es>=f['var']).all() and (f.es>0).all() and np.isfinite(f[['var','es','fz0']].values).all()
  par=ROOT/'wishart_extension/fits'/f"crsp5_{x['year']}.json";assert x['parameter_file_sha256']==hashlib.sha256(par.read_bytes()).hexdigest()
 assert maximum<1e-9
 checks.update(synthetic_fits_checked=10,unchanged_parameter_fits=parameter_count,unchanged_crsp_data=True,unchanged_primary_crsp_outputs=True,unchanged_nonwishart_synthetic_rows=True,max_mixture_risk_reconstruction_error=risk_error,max_forecast_score_reconstruction_error=maximum,forecast_rows_reconstructed=count,analytic_kernel_checks_passed=num_checks,implementation_checks_passed=True)
 if not args.partial:
  status=json.loads((A/'final_status.json').read_text());assert status['audit_complete'];checks.update(crsp_stress_gate_passed=status['crsp_primary_stress_refinement_gate_passed'],crsp_accepted_for_primary_inference=status['crsp_accepted_for_primary_inference'])
 path=A/('release_checks_partial.json' if args.partial else 'release_checks.json');path.write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
