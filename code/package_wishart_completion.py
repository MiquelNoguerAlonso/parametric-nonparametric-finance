"""Package the completed Wishart audit while retaining prior companion archives."""
import argparse,json,re,shutil,zipfile
from pathlib import Path
from package_round4 import ROOT,flatten,digest,archive

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base-archive',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 status=json.loads((ROOT/'wishart_completion/analysis/final_status.json').read_text());assert status['audit_complete']
 checks=json.loads((ROOT/'wishart_completion/analysis/release_checks.json').read_text());assert checks['implementation_checks_passed']
 compact=a.out/'overleaf';compact.mkdir(exist_ok=True)
 source=flatten(ROOT/'main.tex').replace('figures/wishart_paired_mse.pdf','wishart_paired_mse.pdf').replace('figures/round2_annual.pdf','round2_annual.pdf');(compact/'main.tex').write_text(source)
 for name in ['refs.bib']:shutil.copy2(ROOT/name,compact/name)
 for name in ['wishart_paired_mse.pdf','round2_annual.pdf']:shutil.copy2(ROOT/'figures'/name,compact/name)
 (compact/'README.md').write_text('# Parametric and Nonparametric Multivariate Models in Finance\n\nMiquel Noguer i Alonso — Artificial Intelligence Finance Institute\n\nReserved DOI: 10.5281/zenodo.24024955\n\nSet main.tex as root; compile with pdfLaTeX and BibTeX or latexmk -pdf main.tex. The date uses \\today. This is the complete compact manuscript source, including the refined synthetic Wishart evaluation and explicit CRSP numerical-audit decision. Research inputs and checkpoints belong to the separate research archives.\n')
 cp=a.out/'Parametric_and_Nonparametric_Multivariate_Models_in_Finance_Overleaf.zip'
 with zipfile.ZipFile(cp,'w',zipfile.ZIP_DEFLATED) as z:
  for name in ['main.tex','refs.bib','README.md','wishart_paired_mse.pdf','round2_annual.pdf']:z.write(compact/name,name)
 for name in ['Generative_Models_in_Finance_Completed.pdf','Parametric_and_Nonparametric_Multivariate_Models_in_Finance.pdf']:shutil.copy2(ROOT/'main.pdf',a.out/name)
 with zipfile.ZipFile(a.base_archive) as z:main={n for n in z.namelist() if not n.endswith('/') and (ROOT/n).is_file()}
 main|={str(p.relative_to(ROOT)) for p in (ROOT/'code').glob('*.py')}
 w=ROOT/'wishart_completion';extra=set()
 for p in w.rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or p.name=='working_library.json':continue
  rel=p.relative_to(w);name=str(p.relative_to(ROOT))
  if rel.parts[0] in ['results','checkpoints','logs','baseline']:extra.add(name)
  else:main.add(name)
 main|={'main.tex','main.pdf','refs.bib','README.md','PUBLICATION.md','WISHART_RESPONSE.md','wishart_methods.tex','wishart_appendix.tex','expanded_methods.tex','code/run_round4.py'}
 main.discard('MANIFEST.sha256');main={n for n in main if '__pycache__' not in n and not n.endswith(('.aux','.log','.fdb_latexmk','.fls','.out','.blg','.synctex.gz'))}
 # Numerical archive includes its method/source and derived inputs for inspection.
 extra|={str(p.relative_to(ROOT)) for p in (ROOT/'code').glob('*wishart*.py')}
 extra|={str(p.relative_to(ROOT)) for p in (ROOT/'wishart_extension/fits').glob('*.json')}
 extra|={str(p.relative_to(ROOT)) for p in w.rglob('*') if p.is_file() and p.relative_to(w).parts[0] not in ['results','checkpoints','logs','baseline'] and p.name!='working_library.json' and '__pycache__' not in p.parts}
 extra|={'data/crsp_size_portfolios.csv','expansion/data/heston_reference.json','code/experiment.py','code/expanded_experiment.py'}
 (ROOT/'WISHART_MANIFEST.sha256').write_text(''.join(f'{digest(ROOT/n)}  {n}\n' for n in sorted(extra)))
 archive(a.out/'Generative_Models_Finance_Wishart_Numerical_Audit.zip',extra|{'WISHART_MANIFEST.sha256'})
 main.add('WISHART_MANIFEST.sha256');(ROOT/'MANIFEST.sha256').write_text(''.join(f'{digest(ROOT/n)}  {n}\n' for n in sorted(main)))
 archive(a.out/'Generative_Models_Finance_LaTeX.zip',main|{'MANIFEST.sha256'})
 info={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in a.out.iterdir() if p.is_file()}
 (a.out/'release_checksums.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
if __name__=='__main__':main()
