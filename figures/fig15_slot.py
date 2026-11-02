"""Fig 15: slot anatomy. Schematic of one demonstration's label token: value side (label identity) and key side (sentence class), who writes each and who reads each."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patches as mp, figstyle as fs
fig,ax=plt.subplots(figsize=(7.2,3.2)); ax.set_xlim(0,10); ax.set_ylim(0,5.4); ax.axis('off')
def box(x,y,w,h,text,fc,ec='k',fs_=8,tc='k'):
    ax.add_patch(mp.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',fc=fc,ec=ec,lw=0.8)); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs_,color=tc)
box(0.3,4.4,2.4,0.6,'The soup was cold','#f2f2f2'); box(3.0,4.4,0.6,0.6,'->','#f2f2f2'); box(3.9,4.4,0.7,0.6,'bar','#f2f2f2')
ax.add_patch(mp.FancyBboxPatch((5.3,0.9),2.2,3.3,boxstyle='round,pad=0.05',fc='white',ec='k',lw=0.8)); ax.text(6.4,4.45,'the slot: residual stream at "bar"',ha='center',fontsize=8)
box(5.5,2.7,1.8,1.1,'value\nlabel identity $v(y)$',fs.INTACT,tc='w')
box(5.5,1.1,1.8,1.1,'key\nsentence class $\\kappa u c$',fs.INTERV,tc='w',fs_=7.5)
# value from the token itself
ax.annotate('',xy=(5.5,3.4),xytext=(4.4,4.4),arrowprops=dict(arrowstyle='->',color=fs.INTACT,lw=1.2))
ax.text(4.2,3.55,'embedding and\nearly layers',fontsize=7,color=fs.INTACT,ha='right',va='center')
# writers: sentence -> key
ax.annotate('',xy=(5.5,1.5),xytext=(1.5,4.4),arrowprops=dict(arrowstyle='->',color=fs.INTERV,lw=1.2,connectionstyle='arc3,rad=0.35'))
ax.text(1.3,2.0,'writers, layers 10–23:\nattend from "bar" back to\nits sentence, write $u$',fontsize=7,color=fs.INTERV,ha='center',va='center')
# reader
box(8.2,1.9,1.6,1.4,'reader head\nat the query','#f2f2f2')
ax.annotate('',xy=(8.2,2.3),xytext=(7.3,1.6),arrowprops=dict(arrowstyle='<-',color=fs.INTERV,lw=1.2)); ax.text(8.95,1.75,'$W_K$: match the\nquery class',fontsize=7,color=fs.INTERV,ha='center',va='top')
ax.annotate('',xy=(8.2,3.0),xytext=(7.3,3.2),arrowprops=dict(arrowstyle='->',color=fs.INTACT,lw=1.2)); ax.text(7.85,3.55,'$W_V$: copy\nthe label',fontsize=7,color=fs.INTACT,ha='center',va='bottom')
ax.text(5.0,0.3,'swap the labels: the value side moves the answer.  swap the sentence: the key side moves it.',fontsize=7,ha='center',style='italic')
fig.savefig('figures/fig15_slot.pdf',bbox_inches='tight'); print('fig15 written')
