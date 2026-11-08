"""Package the final manuscript without rewriting unchanged numerical archives."""
import argparse,json,shutil,zipfile
from pathlib import Path
from package_round4 import ROOT,flatten,digest,archive

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-archive',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    checks=json.loads((ROOT/'final_submission/final_checks.json').read_text())
    assert checks.get('existing_numerical_outputs_unchanged',checks.get('numerical_outputs_unchanged')) and checks['rendered_pages_reviewed']
    compact=a.out/'overleaf';compact.mkdir(exist_ok=True)
    source=flatten(ROOT/'main.tex')
    for n in ['wishart_paired_mse.pdf','round2_annual.pdf']:
        source=source.replace('figures/'+n,n)
        shutil.copy2(ROOT/'figures'/n,compact/n)
    (compact/'main.tex').write_text(source)
    shutil.copy2(ROOT/'refs.bib',compact/'refs.bib')
    (compact/'README.md').write_text("""# Parametric and Nonparametric Models of Joint Returns: Density, Dependence, and Portfolio Tail Risk

Miquel Noguer i Alonso — Artificial Intelligence Finance Institute
Reserved DOI: 10.5281/zenodo.24024955

These five files contain one complete integrated manuscript, including all appendices. Upload them to one Overleaf project and compile main.tex with pdfLaTeX and BibTeX, or use latexmk -pdf main.tex. No Python or data files are needed to compile. The manuscript date is fixed at November 7, 2026.

The full research package is separate and has six archives; its README.md lists the inputs and current verification commands. The paper contains the completed additive Wishart–CIR synthetic experiment, frozen-law equivalence proof, and conditional numerical audit. Both Wishart specifications are excluded from primary conditional inference because their numerical tolerance is unmet. The recorded limitations are part of the final empirical interpretation.
""")
    cp=a.out/'Parametric_and_Nonparametric_Multivariate_Models_in_Finance_Overleaf.zip'
    with zipfile.ZipFile(cp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['main.tex','refs.bib','README.md','wishart_paired_mse.pdf','round2_annual.pdf']:
            z.write(compact/name,name)
    for n in ['Generative_Models_in_Finance_Completed.pdf','Parametric_and_Nonparametric_Multivariate_Models_in_Finance.pdf']:
        shutil.copy2(ROOT/'main.pdf',a.out/n)
    with zipfile.ZipFile(a.base_archive) as z:
        names={n for n in z.namelist() if not n.endswith('/') and (ROOT/n).is_file()}
    names|={str(p.relative_to(ROOT)) for p in (ROOT/'code').glob('*.py')}
    names|={'main.tex','main.pdf','refs.bib','README.md','PUBLICATION.md','FINALIZATION.md',
            'final_submission/final_checks.json','final_submission/numerical_baseline.json'}
    names|={'theory_main.tex','theory_appendix.tex','ROUND5_RESPONSE.md','ROUND6_RESPONSE.md','ROUND7_RESPONSE.md',
            'final_submission/editorial_numerical_baseline.json'}
    names|={str(p.relative_to(ROOT)) for sub in ['analysis','dcc'] for p in (ROOT/'referee_round5'/sub).rglob('*') if p.is_file()}
    names|={str(p.relative_to(ROOT)) for p in (ROOT/'referee_round5/baseline').rglob('*') if p.is_file() and p.suffix in ['.tex','.bib']}
    names|={'wishart_cir_main.tex','wishart_cir_theory.tex','WISHART_CIR_REVISION.md'}
    names|={'COMPLETION_REVIEW.md'}
    names|={str(p.relative_to(ROOT)) for p in (ROOT/'completion_audit').rglob('*') if p.is_file() and not p.name.endswith('.partial.csv')}
    names|={str(p.relative_to(ROOT)) for p in (ROOT/'wishart_cir_extension').rglob('*') if p.is_file() and not p.name.endswith('.partial.csv')}
    scenario_arrays={str(p.relative_to(ROOT)) for p in (ROOT/'wishart_cir_extension/results').glob('*.npz')}
    archive(a.out/'Wishart_CIR_Scenario_Archive.zip',scenario_arrays)
    names-=scenario_arrays
    names.discard('MANIFEST.sha256')
    names={n for n in names if not n.startswith('wishart_completion/visual_qa/')}
    (ROOT/'MANIFEST.sha256').write_text(''.join(f'{digest(ROOT/n)}  {n}\n' for n in sorted(names)))
    archive(a.out/'Generative_Models_Finance_LaTeX.zip',names|{'MANIFEST.sha256'})
    files=[p for p in a.out.iterdir() if p.is_file() and not p.name.startswith('.') and p.name!='release_checksums.json']
    info={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files}
    (a.out/'release_checksums.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
