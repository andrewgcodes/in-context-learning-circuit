"""Fig 1: the ICL circuit in Qwen3-4B-Base, IOI Figure-2 idiom.
Layout engine: every box is sized from its measured text plus padding, boxes in a column share a width,
lines run only in gutters or along empty token rows, and a collision check (text / box / line) must pass
before anything is written. Units: 0.1 inch."""
import sys, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
W_IN,H_IN=9.4,5.4; WU,HU=W_IN*10,H_IN*10
fig=plt.figure(figsize=(W_IN,H_IN)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,WU); ax.set_ylim(0,HU); ax.axis('off')
R=fig.canvas.get_renderer()
FC={'dup':'#ead9f7','wr':'#fde4cc','pool':'#d8f1d8','q':'#f7e8c6','rd':'#f9d3d3','br':'#dedcf2','cp':'#cfe7f1'}
EC={'dup':'#8e5fb0','wr':'#c2703a','pool':'#3a9a3a','q':'#b08a2a','rd':'#c0303a','br':'#5a5aa8','cp':'#2a7aa0'}
TEXTS=[]; BOXES=[]; PATHS=[]
def bbox(t):
    b=t.get_window_extent(R).transformed(ax.transData.inverted()); return [b.x0,b.y0,b.x1,b.y1]
def text(x,y,s,**kw):
    t=ax.text(x,y,s,**kw); TEXTS.append(t); return t
def measure(s,**kw):
    t=ax.text(0,0,s,**kw); b=bbox(t); t.remove(); return b[2]-b[0], b[3]-b[1]
PADX,PADY,GAPL=1.3,0.7,0.35
TITLE=dict(fontsize=9.2,ha='center',va='center'); SUB=dict(fontsize=7.2,ha='center',va='center',color='#555555')
def box_size(title,sub):
    tw,th=measure(title,**TITLE); sw,sh=measure(sub,**SUB); return max(tw,sw)+2*PADX, th+sh+GAPL+2*PADY, th, sh
def draw_box(x0,yc,w,title,sub,key):
    _,h,th,sh=box_size(title,sub); y0=yc-h/2
    ax.add_patch(mp.FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0,rounding_size=0.6',fc=FC[key],ec=EC[key],lw=1.0))
    text(x0+w/2,y0+PADY+sh+GAPL+th/2,title,**TITLE); text(x0+w/2,y0+PADY+sh/2,sub,**SUB)
    b=dict(x0=x0,x1=x0+w,y0=y0,y1=y0+h,key=key); BOXES.append(b); return b
HEAD_L,HEAD_W=0.95,0.38
def path(pts,color,ls='-',hop=False,name=''):
    PATHS.append(dict(pts=[tuple(p) for p in pts],color=color,ls=ls,hop=hop,name=name))
# ---------- tokens (column A) ----------
rows=[('S1','The soup was cold'),('->','->'),('L1','bar'),('S2','A lovely evening'),('->','->'),('L2','foo'),('…','…'),
      ('S12','Never going back'),('->','->'),('L12','bar'),('Q','Awful, never again'),('END','->')]
chip={'L1':'#ffd0d0','L2':'#ffd0d0','L12':'#ffd0d0','Q':'#d8e8ff','END':'#ffe9a8'}
CHIP=dict(fontsize=7.8,ha='center',va='center',weight='bold'); TOK=dict(fontsize=8.8,ha='left',va='center',family='monospace')
cw=max(measure(l,**CHIP)[0] for l,_ in rows)+1.4; x_chip=0.6; x_tok=x_chip+cw+0.9
DY=4.1; YTOP=HU-2.2; Y=[YTOP-i*DY for i in range(len(rows))]
tok_right={}
for (lab,txt),y in zip(rows,Y):
    if lab!='…':
        ax.add_patch(mp.FancyBboxPatch((x_chip,y-1.25),cw,2.5,boxstyle='round,pad=0,rounding_size=0.5',fc=chip.get(lab,'#f0f0f0'),ec='none'))
        BOXES.append(dict(x0=x_chip,x1=x_chip+cw,y0=y-1.25,y1=y+1.25,key='chip'))
        text(x_chip+cw/2,y,lab,**CHIP)
    t=text(x_tok,y,txt,**TOK); tok_right[lab if lab not in('->','…') else f'{lab}{len(tok_right)}']=bbox(t)
yL2,yS12,yL12,yQ,yEND=Y[5],Y[7],Y[9],Y[10],Y[11]
XA=max(b[2] for b in tok_right.values())+1.2        # right edge of the token column
def start(lab): return tok_right[lab][2]+0.9      # a line starts just after its token's text
# ---------- columns ----------
G1=2.0; XB=XA+G1
wB=max(box_size('Duplicate Token Heads','16 heads, L1–9')[0],box_size('Writer Heads','~20 heads, L10–23')[0])
G2=3.4; XC=XB+wB+G2
specC=[('Pooling Heads','gate g, L20–23','pool'),('Query Heads','22.10  23.26','q'),('Reader Heads','29.11  28.0  24.8 … (~90)','rd'),('Brake Heads','31.30  22.8  22.2 … (24)','br')]
wC=max(box_size(a,b)[0] for a,b,_ in specC)
G3=4.6; XD=XC+wC+G3
wD=box_size('Copier Heads','35.26  33.13 … L30–35')[0]
dup =draw_box(XB,yL2,wB,'Duplicate Token Heads','16 heads, L1–9','dup')
wr  =draw_box(XB,yS12,wB,'Writer Heads','~20 heads, L10–23','wr')
pool=draw_box(XC,yL2,wC,*specC[0])
rd  =draw_box(XC,yL12+0.7,wC,*specC[2])
qh  =draw_box(XC,yQ-0.9,wC,*specC[1])
br  =draw_box(XC,qh['y0']-1.4-box_size(*specC[3][:2])[1]/2,wC,*specC[3])
cp  =draw_box(XD,yL12-1.0,wD,'Copier Heads','35.26  33.13 … L30–35','cp')
HDR=dict(fontsize=8,ha='center',va='bottom',color='#555555')
text((dup['x0']+dup['x1'])/2,dup['y1']+0.9,'at the label tokens',**HDR); text((pool['x0']+pool['x1'])/2,pool['y1']+0.9,'at END, the query arrow',**HDR)
# ---------- edges ----------
path([(start('L2'),yL2),(XB,yL2)],EC['dup'],name='L2->dup')
path([(start('S12'),yS12),(XB,yS12)],EC['wr'],name='S12->wr')
lane=yL12+DY/2+0.7; xbar=(tok_right['L12'][0]+tok_right['L12'][2])/2
path([(XB+2.2,wr['y0']),(XB+2.2,lane),(xbar,lane),(xbar,tok_right['L12'][3]+0.55)],EC['wr'],name='wr->L12 key')
text(xbar+2.6,lane+0.55,'writes class key',fontsize=7,color=EC['wr'],ha='left',va='bottom')
path([(dup['x1'],yL2),(XC,yL2)],EC['pool'],name='dup->pool')
path([(start('Q'),yQ),(XC,yQ)],EC['q'],hop=True,name='Q->qh')
path([(start('L12'),yL12+0.5),(XC,yL12+0.5)],EC['rd'],name='L12->rd')
xbv=XC-2.1; ybr=(br['y0']+br['y1'])/2
path([(start('L12'),yL12-0.5),(xbv,yL12-0.5),(xbv,ybr),(XC,ybr)],EC['br'],ls=':',name='L12->br')
xg=rd['x1']+1.6; xb=rd['x1']+3.1
path([(pool['x1'],yL2),(xg,yL2),(xg,yL12+1.3),(rd['x1'],yL12+1.3)],EC['pool'],ls='--',name='g->rd')
path([((qh['x0']+qh['x1'])/2,qh['y1']),((qh['x0']+qh['x1'])/2,rd['y0'])],EC['q'],ls='--',name='qh->rd')
path([(rd['x1'],yL12-0.3),(XD,yL12-0.3)],EC['rd'],name='rd->cp')
path([(br['x1'],ybr),(xb,ybr),(xb,yL12-1.7),(XD,yL12-1.7)],EC['br'],name='br->cp')
path([(cp['x1'],yL12-1.0),(cp['x1']+3.2,yL12-1.0)],'#222222',name='logits')
text(cp['x1']+3.9,yL12-1.0,'logits',fontsize=8.5,ha='left',va='center')
# ---------- legend ----------
LEG=dict(fontsize=7.4,va='center',ha='left'); items=[('-','reads keys / values of a token'),('--','acts on the readers\u2019 queries'),(':','reads the most recent label')]
lw_=max(measure(s,**LEG)[0] for _,s in items); LW=lw_+6.4+2*1.0; LH=3*2.3+1.0; lx1=cp['x1']+3.9+measure('logits',fontsize=8.5)[0]; lx0=lx1-LW; ly1=Y[0]+1.3; ly0=ly1-LH
ax.add_patch(mp.FancyBboxPatch((lx0,ly0),LW,LH,boxstyle='round,pad=0,rounding_size=0.5',fc='#fafafa',ec='#999999',lw=0.8)); BOXES.append(dict(x0=lx0,x1=lx0+LW,y0=ly0,y1=ly1,key='legend'))
for i,(ls,s) in enumerate(items):
    yy=ly1-0.5-1.15-i*2.3; ax.plot([lx0+1.0,lx0+5.8],[yy,yy],color='#222222',lw=1.0,ls=ls); text(lx0+6.4,yy,s,**LEG)
# ---------- draw paths with hops and arrowheads ----------
def seg_cross(a,b,c,d):
    (x1,y1),(x2,y2)=a,b; (x3,y3),(x4,y4)=c,d
    if abs(x1-x2)<1e-9 and abs(y3-y4)<1e-9:   # a-b vertical, c-d horizontal
        if min(y1,y2)+0.3<y3<max(y1,y2)-0.3 and min(x3,x4)+0.3<x1<max(x3,x4)-0.3: return (x1,y3)
    if abs(y1-y2)<1e-9 and abs(x3-x4)<1e-9: return seg_cross(c,d,a,b)
    return None
segs=[(i,p['pts'][k],p['pts'][k+1]) for i,p in enumerate(PATHS) for k in range(len(p['pts'])-1)]
hops={}
for i,a,b in segs:
    for j,c,d in segs:
        if i>=j: continue
        X=seg_cross(a,b,c,d)
        if X:
            pi,pj=PATHS[i],PATHS[j]
            if pi['hop']: hops.setdefault(i,[]).append(X)
            elif pj['hop']: hops.setdefault(j,[]).append(X)
            else: sys.exit(f'FAIL crossing between {pi["name"]} and {pj["name"]} at {X}')
for i,p in enumerate(PATHS):
    pts=p['pts']; col=p['color']
    for k,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
        last=(k==len(pts)-2); a=np.array(a,float); b=np.array(b,float); d=(b-a)/np.linalg.norm(b-a)
        end=b-d*(HEAD_L*0.85) if last else b
        pieces=[(a,end)]
        for X in hops.get(i,[]):
            X=np.array(X); new=[]
            for s,e in pieces:
                t=np.dot(X-s,d)
                if 0<t<np.linalg.norm(e-s): new+=[(s,s+d*(t-0.7)),(s+d*(t+0.7),e)]
                else: new.append((s,e))
            pieces=new
        for s,e in pieces: ax.plot([s[0],e[0]],[s[1],e[1]],color=col,lw=1.0,ls=p['ls'],solid_capstyle='butt')
        if last:
            n=np.array([-d[1],d[0]]); tri=[b,b-d*HEAD_L+n*HEAD_W,b-d*HEAD_L-n*HEAD_W]
            ax.add_patch(mp.Polygon(tri,closed=True,fc=col,ec=col,lw=0.5))
# ---------- collision check ----------
fig.canvas.draw()
TB=[(t.get_text(),bbox(t)) for t in TEXTS]
def inflate(b,m): return [b[0]-m,b[1]-m,b[2]+m,b[3]+m]
def rect_overlap(a,b): return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]
def seg_rect(a,b,r):
    (x1,y1),(x2,y2)=a,b
    return rect_overlap([min(x1,x2)-1e-6,min(y1,y2)-1e-6,max(x1,x2)+1e-6,max(y1,y2)+1e-6],r)
fails=[]
for i,(s,a) in enumerate(TB):
    for j,(t,b) in enumerate(TB):
        if i<j and rect_overlap(inflate(a,0.25),b): fails.append(f'text/text: "{s}" vs "{t}"')
    if a[0]<0.2 or a[2]>WU-0.2 or a[1]<0.2 or a[3]>HU-0.2: fails.append(f'text outside canvas: "{s}"')
    for bx in BOXES:
        rb=[bx['x0'],bx['y0'],bx['x1'],bx['y1']]; inside=a[0]>=rb[0]+0.5 and a[2]<=rb[2]-0.5 and a[1]>=rb[1]+0.4 and a[3]<=rb[3]-0.4
        if not inside and rect_overlap(inflate(a,0.4),rb): fails.append(f'text/box: "{s}" vs {bx["key"]}')
for i,a in enumerate(BOXES):
    for j,b in enumerate(BOXES):
        if i<j and rect_overlap([a['x0']-0.9,a['y0']-0.9,a['x1']+0.9,a['y1']+0.9],[b['x0'],b['y0'],b['x1'],b['y1']]) and not(a['key']=='chip' and b['key']=='chip'): fails.append(f'box/box: {a["key"]} vs {b["key"]}')
for i,a,b in segs:
    nm=PATHS[i]['name']
    for s,tb in TB:
        if seg_rect(a,b,inflate(tb,0.45)): fails.append(f'line/text: {nm} vs "{s}"')
    for bx in BOXES:
        rb=[bx['x0'],bx['y0'],bx['x1'],bx['y1']]; infl=inflate(rb,0.45)
        if seg_rect(a,b,infl):
            ok=False
            for P in (a,b):
                on=(abs(P[0]-rb[0])<0.05 or abs(P[0]-rb[2])<0.05) and rb[1]+0.6<=P[1]<=rb[3]-0.6 and abs(a[1]-b[1])<1e-9
                on|=(abs(P[1]-rb[1])<0.05 or abs(P[1]-rb[3])<0.05) and rb[0]+0.6<=P[0]<=rb[2]-0.6 and abs(a[0]-b[0])<1e-9
                ok|=on
            if not ok: fails.append(f'line/box: {nm} vs {bx["key"]}')
    for j,c,d in segs:
        if i>=j: continue
        if abs(a[1]-b[1])<1e-9 and abs(c[1]-d[1])<1e-9 and abs(a[1]-c[1])<0.8 and min(a[0],b[0])<max(c[0],d[0]) and min(c[0],d[0])<max(a[0],b[0]): fails.append(f'parallel lines: {nm} vs {PATHS[j]["name"]}')
        if abs(a[0]-b[0])<1e-9 and abs(c[0]-d[0])<1e-9 and abs(a[0]-c[0])<0.8 and min(a[1],b[1])<max(c[1],d[1]) and min(c[1],d[1])<max(a[1],b[1]): fails.append(f'parallel lines: {nm} vs {PATHS[j]["name"]}')
    if np.hypot(b[0]-a[0],b[1]-a[1])<1.3: fails.append(f'segment too short: {nm}')
print(f'{len(TB)} texts, {len(BOXES)} boxes, {len(segs)} segments, {sum(len(v) for v in hops.values())} hop(s)')
if fails:
    print('\n'.join(sorted(set(fails)))); sys.exit('FAIL: layout check')
fig.savefig('figures/fig1_circuit.pdf',bbox_inches='tight',pad_inches=0.05); fig.savefig('/tmp/fig1_circuit.png',dpi=300,bbox_inches='tight',pad_inches=0.05); print('fig1 v6 written')
