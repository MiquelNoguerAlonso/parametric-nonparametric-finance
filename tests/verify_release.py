"""Verify the six research archives, extract their common tree, and optionally run checks."""
import argparse,hashlib,json,os,stat,subprocess,sys,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archives',type=Path,required=True)
parser.add_argument('--extract-to',type=Path,default=ROOT/'research')
parser.add_argument('--run',action='store_true')
args=parser.parse_args()
manifest=json.loads((ROOT/'release_assets.json').read_text())
lookup={e['name']:e for e in manifest['assets']}
sources=[]
for name in manifest['research_extraction_order']:
    source=args.archives/name
    with source.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=lookup[name]['sha256']:raise SystemExit('Archive hash mismatch: '+name)
    sources.append(source)
    print('Verified archive:',name,flush=True)
dest=args.extract_to.resolve();dest.mkdir(parents=True,exist_ok=True)
seen={}
for source in sources:
    with zipfile.ZipFile(source) as archive:
        for entry in archive.infolist():
            target=(dest/entry.filename).resolve()
            if not target.is_relative_to(dest):raise SystemExit('Unsafe archive path: '+entry.filename)
            if stat.S_ISLNK(entry.external_attr>>16):raise SystemExit('Unexpected symlink: '+entry.filename)
            if entry.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            data=archive.read(entry);sha=hashlib.sha256(data).hexdigest()
            if entry.filename in seen and seen[entry.filename]!=sha:
                raise SystemExit('Conflicting archive versions: '+entry.filename)
            seen[entry.filename]=sha
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    print('Extracted:',source.name,flush=True)
print('Extracted',len(seen),'distinct files.',flush=True)
if args.run:
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    subprocess.run([sys.executable,'code/verify_current_release.py'],cwd=dest,env=env,check=True)
    print('Saved evidence verified. The documented Wishart–CIR numerical gate remains failed.')
