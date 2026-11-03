"""Fig. 1: the algorithm as a stage map along the layers of Qwen3-4B-Base. Short labels; detail in the caption."""
import matplotlib.pyplot as plt, matplotlib.patches as P, figstyle as fs
fs.use()
fig,ax=plt.subplots(figsize=(7.0,3.3)); ax.set_xlim(-4.2,40.5); ax.set_ylim(-0.6,3.3); ax.axis('off')
YS=2.5; YA=0.95
for y,label in ((YS,'label tokens\n(slots)'),(YA,'query\narrow')):
    for d in (-0.5,0.5): ax.plot([0,36],[y+d,y+d],color='#E3E3E3',lw=0.6)
    ax.text(-0.6,y,label,ha='right',va='center',fontsize=8.5)
for x in range(0,36,5): ax.text(x+0.5,-0.3,str(x),ha='center',va='center',fontsize=7,color='#666666')
for x in range(0,37): ax.plot([x,x],[-0.12,-0.06],color='#BBBBBB',lw=0.4)
ax.text(18,-0.55,'layer',ha='center',va='center',fontsize=8,color='#666666')
C={'det':'#6B6B6B','write':fs.PRED,'gate':'#7A3E9D','read':fs.INTERV,'brake':'#B07A00','copy':'#2E8B57','slot':'#F2F2F2'}
def box(x0,x1,y,text,c,h=0.62,ec=None,fsz=7.6):
    ax.add_patch(P.FancyBboxPatch((x0,y-h/2),x1-x0,h,boxstyle='round,pad=0.02,rounding_size=0.12',fc=c,ec=ec or c,lw=0.8))
    ax.text((x0+x1)/2,y,text,ha='center',va='center',fontsize=fsz,color=fs.INTACT if c==C['slot'] else 'white',linespacing=1.15)
# slot lane: layers are [x0,x1) so box from l to l+1
box(1,10,YS,'duplicate\ndetector',C['det'])
box(10,24,YS,'writers\n(class → key)',C['write'])
box(24,30,YS,'slot:\nvalue + key',C['slot'],ec='#999999')
# arrow lane
box(1,14,YA,'encoder\n(sentence class)',C['det'])
box(19.5,23.3,YA+0.17,'gate g',C['gate'],h=0.3,fsz=6.5)
box(19.5,23.3,YA-0.17,'query code',C['write'],h=0.3,fsz=5.6)
box(24,30,YA,'readers\n(softmax + null)',C['read'],fsz=6.4)
box(30,36,YA,'readout\nband',C['copy'])
box(22,35,YA-0.62,'brakes',C['brake'],h=0.2,fsz=5.6)
ax.annotate('',xy=(38.6,YA),xytext=(36.2,YA),arrowprops=dict(arrowstyle='-|>',color=fs.INTACT,lw=0.9)); ax.text(39.2,YA,'logits',ha='left',va='center',fontsize=8.5)
def arr(x0,y0,x1,y1,c,rad=0.0,ls='-'):
    ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='-|>',color=c,lw=0.95,connectionstyle=f'arc3,rad={rad}',linestyle=ls))
def lab(x,y,s,c): ax.text(x,y,s,ha='center',va='center',fontsize=6.8,color=c,bbox=dict(fc='white',ec='none',pad=0.5))
arr(13.5,YA+0.33,16.5,YS-0.33,C['write']);              lab(13.0,1.73,'class',C['write'])
arr(6,YS-0.33,20.5,YA+0.34,C['gate'],rad=-0.12);           lab(14.5,1.42,'repeat marks',C['gate'])
arr(29.75,YS-0.33,29.75,YA+0.33,C['read']);                    lab(29.6,1.73,'match key,\ncopy value',C['read'])
arr(23.35,YA+0.17,24.0,YA+0.17,C['gate']);                lab(21.5,1.62,'g opens the read',C['gate'])
from figcheck import check; check(fig)
fig.savefig('figures/fig1_algorithm.pdf'); fig.savefig('figures/fig1_algorithm.png'); print('fig1 written')
