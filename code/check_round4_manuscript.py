"""Release-level document checks after LaTeX compilation."""
import json,re
from pathlib import Path
import pdfplumber
from package_round4 import ROOT,flatten

def main():
    log=(ROOT/'main.log').read_text(errors='replace')
    assert not re.search(r'undefined references|Citation .* undefined|Reference .* undefined|Undefined control sequence|Float too large',log,re.I)
    source=flatten(ROOT/'main.tex');assert r'\date{\today' in source and '10.5281/zenodo.24024955' in source
    pages=[];overflow=[]
    with pdfplumber.open(ROOT/'main.pdf') as pdf:
        for i,page in enumerate(pdf.pages,1):
            text=page.extract_text() or '';pages.append(text)
            for word in page.extract_words():
                if word['x0']<35 or word['x1']>page.width-35 or word['top']<30 or word['bottom']>page.height-25:overflow.append(dict(page=i,text=word['text'],box=[word['x0'],word['top'],word['x1'],word['bottom']]))
    assert not overflow,overflow[:20]
    joined='\n'.join(pages);assert '1,110' in joined and '640' in joined and 'Student' in joined
    checks=dict(pages=len(pages),tables=len(re.findall(r'\\begin\{table\}',source)),figures=len(re.findall(r'\\begin\{figure\}',source)),abstract_words=len((ROOT/'referee_round4/analysis/abstract.tex').read_text().split()),unresolved_references=False,margin_overflows=overflow,overfull_boxes=re.findall(r'Overfull \\[hv]box.*',log))
    (ROOT/'referee_round4/analysis/manuscript_checks.json').write_text(json.dumps(checks,indent=2))
    (ROOT/'referee_round4/analysis/page_text.json').write_text(json.dumps(pages,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
