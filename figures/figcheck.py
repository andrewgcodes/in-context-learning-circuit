"""Collision check for hand-drawn figures: every text must be clear of other text, of patches it is not inside, and of arrows."""
import sys, numpy as np, matplotlib.patches as mp
def check(fig, margin_pt=2.0):
    fig.canvas.draw(); R=fig.canvas.get_renderer(); fails=[]
    for ax in fig.axes:
        texts=[(t.get_text(),t.get_window_extent(R)) for t in ax.texts if t.get_text().strip()]
        onlines={t.get_text() for t in ax.texts if t.get_bbox_patch() is not None}
        patches=[p.get_window_extent(R) for p in ax.patches if isinstance(p,mp.FancyBboxPatch)]
        arrows=[]
        for a in ax.texts:
            if getattr(a,'arrow_patch',None) is not None:
                arrows.append((ax.transData.transform(a.xy),ax.transData.transform(a.xyann)))
        for l in ax.lines:
            xy=l.get_transform().transform(np.c_[l.get_xdata(),l.get_ydata()])
            arrows += [(xy[i],xy[i+1]) for i in range(len(xy)-1)]
        m=margin_pt*fig.dpi/72
        def ov(a,b): return a.x0<b.x1 and b.x0<a.x1 and a.y0<b.y1 and b.y0<a.y1
        def grow(b): 
            from matplotlib.transforms import Bbox; return Bbox([[b.x0-m,b.y0-m],[b.x1+m,b.y1+m]])
        for i,(s,a) in enumerate(texts):
            for j,(t,b) in enumerate(texts):
                if i<j and ov(grow(a),b): fails.append(f'text/text "{s}" vs "{t}"')
            for b in patches:
                inside=a.x0>=b.x0 and a.x1<=b.x1 and a.y0>=b.y0 and a.y1<=b.y1
                if not inside and ov(grow(a),b): fails.append(f'text/patch "{s}"')
            A=grow(a)
            for (p,q) in ([] if s in onlines else arrows):
                xs=np.linspace(p[0],q[0],40); ys=np.linspace(p[1],q[1],40)
                if np.any((xs>A.x0)&(xs<A.x1)&(ys>A.y0)&(ys<A.y1)): fails.append(f'line/text "{s}"')
    fails=sorted(set(fails))
    if fails: print('\n'.join(fails)); sys.exit('FAIL: figure collision check')
    print('figcheck: clean')
