import modal,subprocess,os
app=modal.App('mi-toy'); vol=modal.Volume.from_name('mi-grails',create_if_missing=True)
img=modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.4.0','transformers==4.46.3','accelerate','numpy').add_local_file(os.path.join(os.path.dirname(__file__),'train_toy.py'),'/root/train_toy.py').add_local_file(os.path.join(os.path.dirname(__file__),'analyze_toy.py'),'/root/analyze_toy.py').add_local_file(os.path.join(os.path.dirname(__file__),'train_toy.py'),'/root/train_toy.py')
@app.function(image=img,gpu='H100',timeout=3*3600,volumes={'/vol':vol})
def train(cond:str,seed:int,steps:int,layers:int):
    out=f'/vol/toy/{cond}_L{layers}_k{steps//1000}_s{seed}'; subprocess.run(['python','/root/train_toy.py','--cond',cond,'--seed',str(seed),'--steps',str(steps),'--layers',str(layers),'--out',out],check=True); vol.commit()
@app.local_entrypoint()
def main(cond:str='mix',seed:int=0,steps:int=6000,layers:int=4): train.remote(cond,seed,steps,layers)

@app.function(image=img,gpu='A10G',timeout=3600,volumes={'/vol':vol})
def analyze(run:str,n:int=160):
    r=subprocess.run(['python','/root/analyze_toy.py',f'/vol/toy/{run}','--n',str(n)],capture_output=True,text=True)
    open(f'/vol/toy/{run}/analysis_gpu.log','w').write(r.stdout+r.stderr); vol.commit(); return r.stdout[-3000:]

@app.local_entrypoint()
def analyze_main(run:str,n:int=160):
    print(analyze.remote(run,n))
