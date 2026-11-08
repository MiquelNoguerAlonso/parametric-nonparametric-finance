"""Package the verified manuscript and reproduction archive with checksums."""
from pathlib import Path
import argparse,hashlib,zipfile,shutil,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
 references=[ROOT/'expansion/data'/f'{law}_reference_losses.npz' for law in ['t','metat']]
 reference_manifest=ROOT/'REFERENCE_MANIFEST.sha256'
 reference_manifest.write_text(''.join(hashlib.file_digest(x.open('rb'),'sha256').hexdigest()+'  '+x.relative_to(ROOT).as_posix()+'\n' for x in references))
 training_roots=[ROOT/'referee_round3'/'filters',ROOT/'referee_round3'/'checkpoints']
 training_files=[x for base in training_roots for x in base.rglob('*') if x.is_file()]
 training_manifest=ROOT/'ROUND3_TRAINING_MANIFEST.sha256'
 training_manifest.write_text(''.join(hashlib.file_digest(x.open('rb'),'sha256').hexdigest()+'  '+x.relative_to(ROOT).as_posix()+'\n' for x in sorted(training_files)))
 training_zip=out/'Generative_Models_Finance_Round3_Checkpoints.zip'
 with zipfile.ZipFile(training_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for x in sorted(training_files)+[training_manifest]:z.write(x,x.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(training_zip) as z:
  assert z.testzip() is None
  for line in z.read('ROUND3_TRAINING_MANIFEST.sha256').decode().splitlines():
   digest,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
 skipdirs={'__pycache__','logs','tmp'};skipsuffix={'.aux','.blg','.fls','.fdb_latexmk','.out','.log','.lock','.pyc','.tmp'}
 files=[x for x in ROOT.rglob('*') if x.is_file() and x not in references and x not in training_files and not set(x.relative_to(ROOT).parts)&skipdirs and x.suffix not in skipsuffix and x.name!='MANIFEST.sha256' and not any(z.startswith('.') for z in x.relative_to(ROOT).parts)]
 manifest=''.join(hashlib.file_digest(x.open('rb'),'sha256').hexdigest()+'  '+x.relative_to(ROOT).as_posix()+'\n' for x in sorted(files));(ROOT/'MANIFEST.sha256').write_text(manifest)
 files.append(ROOT/'MANIFEST.sha256');target=out/'Generative_Models_Finance_LaTeX.zip'
 with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(files):z.write(f,f.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None
  for line in z.read('MANIFEST.sha256').decode().splitlines():
   digest,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
 reference_zip=out/'Generative_Models_Finance_Reference_Samples.zip'
 with zipfile.ZipFile(reference_zip,'w',compression=zipfile.ZIP_STORED) as z:
  for f in references+[reference_manifest]:z.write(f,f.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(reference_zip) as z:
  assert z.testzip() is None
  for line in z.read('REFERENCE_MANIFEST.sha256').decode().splitlines():
   digest,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
 pdf=out/'Generative_Models_in_Finance_Completed.pdf';shutil.copy2(ROOT/'main.pdf',pdf)
 print(json.dumps(dict(training_zip_bytes=training_zip.stat().st_size,training_manifest_verified=True,entries=len(files),zip_bytes=target.stat().st_size,reference_zip_bytes=reference_zip.stat().st_size,pdf_bytes=pdf.stat().st_size,zip_sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),reference_zip_sha256=hashlib.file_digest(reference_zip.open('rb'),'sha256').hexdigest(),pdf_sha256=hashlib.file_digest(pdf.open('rb'),'sha256').hexdigest(),manifest_verified=True,reference_manifest_verified=True,extraction='Extract source, reference-sample and round-three checkpoint ZIP files into the same directory before reproduction'),indent=2))
if __name__=='__main__':main()
