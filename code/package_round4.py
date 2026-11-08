"""Package the current release without duplicating the large companion archives."""
import argparse,hashlib,re,shutil,zipfile,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def flatten(path,stack=()):
    if path in stack:raise RuntimeError('Recursive LaTeX input: '+str(path))
    text=path.read_text()
    def sub(m):
        name=m.group(1)
        if '#' in name:return m.group(0)
        p=ROOT/name
        if not p.suffix:p=p.with_suffix('.tex')
        return flatten(p,stack+(path,))
    return re.sub(r'\\(?:input|inputrows)\{([^}]+)\}',sub,text)

def archive(path,names):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(names):z.write(ROOT/name,name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-archive',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    compact=a.out/'overleaf';compact.mkdir(exist_ok=True)
    source=flatten(ROOT/'main.tex').replace('figures/wishart_paired_mse.pdf','wishart_paired_mse.pdf').replace('figures/round2_annual.pdf','round2_annual.pdf')
    (compact/'main.tex').write_text(source);shutil.copy2(ROOT/'refs.bib',compact/'refs.bib')
    for name in ['wishart_paired_mse.pdf','round2_annual.pdf']:shutil.copy2(ROOT/'figures'/name,compact/name)
    (compact/'README.md').write_text('# Parametric and Nonparametric Multivariate Models in Finance\n\nMiquel Noguer i Alonso — Artificial Intelligence Finance Institute\n\nReserved DOI: 10.5281/zenodo.24024955\n\nThis compact Overleaf package is the repaired current manuscript. Set `main.tex` as the root document and compile with pdfLaTeX and BibTeX (or `latexmk -pdf main.tex`). The date uses `\\today`. The five files contain the complete paper and its figures; the separate four-part research archive contains data, checkpoints, code and numerical audits. Earlier unused project files are not inputs to this flattened root document.\n')
    cp=a.out/'Parametric_and_Nonparametric_Multivariate_Models_in_Finance_Overleaf.zip'
    with zipfile.ZipFile(cp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['main.tex','refs.bib','README.md','wishart_paired_mse.pdf','round2_annual.pdf']:z.write(compact/name,name)
    for name in ['Generative_Models_in_Finance_Completed.pdf','Parametric_and_Nonparametric_Multivariate_Models_in_Finance.pdf']:shutil.copy2(ROOT/'main.pdf',a.out/name)
    heavy=[]
    for folder in ['checkpoints','results','pointwise','cache']:
        heavy += [str(p.relative_to(ROOT)) for p in (ROOT/'referee_round4'/folder).rglob('*') if p.is_file()]
    (ROOT/'ROUND4_MANIFEST.sha256').write_text(''.join(f'{digest(ROOT/n)}  {n}\n' for n in sorted(heavy)))
    archive(a.out/'Generative_Models_Finance_Round4_Results_Checkpoints.zip',heavy+['ROUND4_MANIFEST.sha256'])
    with zipfile.ZipFile(a.base_archive) as z:main={n for n in z.namelist() if not n.endswith('/') and (ROOT/n).is_file()}
    main|={str(p.relative_to(ROOT)) for p in (ROOT/'code').glob('*.py')}
    main|={str(p.relative_to(ROOT)) for p in (ROOT/'referee_round4').rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT/'referee_round4').parts for x in ['checkpoints','results','pointwise','cache','logs','__pycache__'])}
    main|={'round4_methods.tex','ROUND4_RESPONSE.md','ROUND4_MANIFEST.sha256','README.md','REVISION_AUDIT.md','PUBLICATION.md','requirements.txt','main.pdf','main.tex','refs.bib'}
    main.discard('MANIFEST.sha256');main={n for n in main if '__pycache__' not in n and not n.endswith(('.aux','.log','.fdb_latexmk','.fls','.out','.blg','.synctex.gz'))}
    (ROOT/'MANIFEST.sha256').write_text(''.join(f'{digest(ROOT/n)}  {n}\n' for n in sorted(main)))
    archive(a.out/'Generative_Models_Finance_LaTeX.zip',main|{'MANIFEST.sha256'})
    info={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in a.out.iterdir() if p.is_file()}
    (a.out/'release_checksums.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))

if __name__=='__main__':main()
