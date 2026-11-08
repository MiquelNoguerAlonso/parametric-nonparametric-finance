"""Keep the archived update-250 state authoritative when replay differs in roundoff."""
import copy,json
from round4_repair import ROOT,OUT,old_records,work

def main():
    audit=OUT/'duplicate_grid_audit';audit.mkdir(exist_ok=True);changed=[]
    for p in sorted((OUT/'results').glob('*.json')):
        if '_step' in p.name:continue
        z=json.loads(p.read_text())
        rules=[k for k,v in z.get('choices',{}).items() if v['early_added'] and v['step']>=250]
        if not rules:continue
        assert all(z['choices'][k]['step']==z['old_choices'][k]['step']==250 for k in rules)
        _,old,trace,paths,mt=old_records(z['study'],z['unit'],z['model'],z['family'],z['seed'])
        (audit/p.name).write_text(json.dumps(dict(reason='Update 250 is an existing candidate, not an added early state; preserve the archived state despite roundoff in replay criteria.',previous_record=z),indent=2))
        meta=OUT/'checkpoints'/(p.stem+'_choices.json');m=json.loads(meta.read_text())
        for rule in rules:
            assert abs(z['choices'][rule]['value']-old[rule]['value'])<1e-5
            m['choices'][rule]=dict(**old[rule],checkpoint=str(paths['250'].relative_to(ROOT)),early_added=False)
        meta.write_text(json.dumps(m,indent=2));p.unlink();ev=OUT/'results'/(p.stem+'_step250_evaluation.json')
        if ev.exists():ev.unlink()
        work(z['study'],z['unit'],z['family'],z['seed'],[z['model']]);changed.append(dict(tag=p.stem,rules=rules))
    if changed:(audit/'normalization.json').write_text(json.dumps(changed,indent=2))
    print('Normalized duplicate existing-grid candidates:',len(changed),flush=True)

if __name__=='__main__':main()
