"""Create a self-contained public LaTeX bundle without research data."""
from pathlib import Path
import re,shutil,subprocess,zipfile,json,hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'publication'
PUBLIC=OUT/'overleaf_wishart'

def expand(path,stack=()):
    assert path not in stack, 'Cyclic TeX input'
    text=path.read_text()
    def replace(m):
        if '#' in m.group(1):return m.group(0)
        child=ROOT/m.group(1)
        if not child.suffix:child=child.with_suffix('.tex')
        return expand(child,stack+(path,)).rstrip('\n')
    return re.sub(r'\\input\{([^}]+)\}',replace,text)

def main():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    text=expand(ROOT/'main.tex')
    assets=[]
    for value in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
        source=ROOT/value
        if not source.suffix:source=source.with_suffix('.pdf')
        assert source.is_file(),source
        if source.name not in assets:
            shutil.copy2(source,PUBLIC/source.name);assets.append(source.name)
        text=text.replace('{'+value+'}','{'+source.name+'}')
    (PUBLIC/'main.tex').write_text(text)
    shutil.copy2(ROOT/'refs.bib',PUBLIC/'refs.bib')
    (PUBLIC/'README.md').write_text('''# Parametric and Nonparametric Multivariate Models in Finance

This public manuscript bundle contains the full editable text in main.tex,
references in refs.bib, and the two figures. Compile main.tex with pdfLaTeX
and BibTeX (or latexmk -pdf). The date uses \\today.

Reserved Zenodo DOI: 10.5281/zenodo.24024955. A reserved DOI does not imply publication.

The current revision contains symmetric primary inference, dense-LU and
coordinate-filter controls, and an appendix-only Wishart CRSP numerical pilot.
The complete research archive is distributed separately; this public bundle
does not contain licensed CRSP data, checkpoints, or raw experiment files.
''')
    with (OUT/'overleaf_wishart_build.log').open('w') as log:
        subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=PUBLIC,stdout=log,stderr=subprocess.STDOUT,check=True)
    pdf=OUT/'Parametric_and_Nonparametric_Multivariate_Models_in_Finance.pdf'
    shutil.copy2(PUBLIC/'main.pdf',pdf)
    names=['main.tex','refs.bib','README.md',*assets]
    target=OUT/'Parametric_and_Nonparametric_Multivariate_Models_in_Finance_Overleaf.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(PUBLIC/name,name)
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    print(json.dumps(dict(files=names,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
