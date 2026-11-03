"""modal run --detach icl/modal_app.py --models Qwen/Qwen3-4B-Base --mode concept12 --extra "seed=1 out=/vol/grails/example"   (one H100 per call, models run sequentially; outputs on the mi-grails volume)"""
import modal,subprocess,os,time,threading
app=modal.App('mi-grails')
vol=modal.Volume.from_name('mi-grails',create_if_missing=True); hf=modal.Volume.from_name('mi-hfcache',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.8.0','transformers>=4.57','accelerate','datasets','numpy','scipy','sentencepiece','protobuf','tiktoken','blobfile')
       .add_local_dir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),remote_path='/root/repo',ignore=['runs','paper','figures','*.log','__pycache__','.git']))
secrets=[modal.Secret.from_dict({'HF_TOKEN':os.environ['HF_TOKEN']})] if os.environ.get('HF_TOKEN') else []
@app.function(image=image,gpu='H100',timeout=12*3600,volumes={'/vol':vol,'/root/.cache/huggingface':hf},secrets=secrets,cpu=8,memory=65536)
def go(models:str,mode:str,extra:str='',script:str='run.py'):
    stop=False
    def committer():
        while not stop: time.sleep(60); vol.commit()
    threading.Thread(target=committer,daemon=True).start()
    for name in models.split(','):
        cmd=f'python /root/repo/icl/{script} {name} {mode} {extra}'; print('>>>',cmd,flush=True)
        tag=name.split('/')[-1]; os.makedirs(f'/vol/grails/{tag}',exist_ok=True); log=open(f'/vol/grails/{tag}/{mode}.log','a')
        p=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in p.stdout: print(line,end='',flush=True); log.write(line); log.flush()
        p.wait(); print('exit',p.returncode,flush=True); vol.commit()
    stop=True; vol.commit()
@app.local_entrypoint()
def main(models:str='Qwen/Qwen3-4B-Base',mode:str='concept12',extra:str='',script:str='run.py'): go.remote(models,mode,extra,script)
