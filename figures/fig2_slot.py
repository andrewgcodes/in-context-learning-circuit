"""Fig 2: slot = value + key. Numbers from analysis/AUDIT_effects.tsv (paired bootstrap recomputed): D61/D61c/D71 (Qwen), D138s (SmolLM3), D161kv/D161key (Llama)."""
import matplotlib.pyplot as plt,numpy as np,figstyle as fs
fs.use()
fig,axs=plt.subplots(1,3,figsize=(7.0,2.6),gridspec_kw={'width_ratios':[1,1,0.8]})
# (a) label exchange, Qwen D61c (intact +3.89) and Llama D161kv (intact +2.84): share of twin effect through values vs keys at label tokens
def share(ax,title,rows):
    labels=[r[0] for r in rows]; eff=[r[1] for r in rows]; lo=[r[2] for r in rows]; hi=[r[3] for r in rows]; col=[r[4] for r in rows]
    fs.effect_bar(ax,labels,eff,lo,hi,col); ax.set_title(title,fontsize=9)
share(axs[0],'label exchange (values)',[
 ('Qwen: K/V, all positions',-7.15,-7.39,-6.90,fs.INTACT),('Qwen: label-token K/V (= full twin)',-7.18,-7.43,-6.94,fs.INTERV),('Qwen: 8 readers, V only',-2.16,-2.41,-1.90,fs.INTERV),('Qwen: 8 readers, K only',+0.56,+0.44,+0.69,fs.CONTROL),
 ('SmolLM3: K/V, all positions',-4.94,-5.38,-4.49,fs.INTACT),('SmolLM3: label-token K/V (= full twin)',-5.55,-5.93,-5.15,fs.MODEL['SmolLM3-3B']),('SmolLM3: label K/V L21–29',-2.94,-3.30,-2.58,fs.MODEL['SmolLM3-3B']),
 ('Llama: K/V, all positions',-4.55,-4.97,-4.15,fs.INTACT),('Llama: label-token K/V (= full twin)',-4.92,-5.33,-4.54,fs.MODEL['Llama-3.2-3B'])])
share(axs[1],'sentence swap (keys)',[
 ('Qwen: twin',-2.87,-3.13,-2.62,fs.INTACT),('Qwen: label K/V L21–29',-2.35,-2.58,-2.12,fs.INTERV),('Qwen: 8 readers, K only',-0.55,-0.67,-0.44,fs.INTERV),('Qwen: 8 readers, V only',+0.10,+0.07,+0.14,fs.CONTROL),
 ('Llama: twin',-0.23,-0.30,-0.17,fs.INTACT),('Llama: label K/V L7–13',-0.15,-0.20,-0.11,fs.MODEL['Llama-3.2-3B']),('Llama: label K/V L14–20',-0.15,-0.19,-0.11,fs.MODEL['Llama-3.2-3B']),('Llama: label K/V L0–6',-0.02,-0.05,+0.01,fs.CONTROL),('Llama: label K/V L21–27',-0.04,-0.08,-0.01,fs.CONTROL),
 ('SmolLM3: twin',-0.33,-0.41,-0.25,fs.INTACT),('SmolLM3: exact span',-0.33,-0.41,-0.25,fs.MODEL['SmolLM3-3B'])])
share(axs[2],'same-class twin (control)',[('Qwen: twin',+0.08,-0.07,+0.23,fs.CONTROL),('Qwen: all handoffs',+0.08,-0.07,+0.23,fs.CONTROL),('Qwen: 9 readers out',+0.01,-0.03,+0.05,fs.CONTROL),('Qwen: label K/V L21–29',-0.06,-0.16,+0.03,fs.CONTROL)])
for ax,s in zip(axs,'abc'): fs.panel_label(ax,s,x=-0.9 if s!='c' else -1.0)
fig.subplots_adjust(wspace=1.35); fig.savefig('figures/fig2_slot.pdf'); fig.savefig('figures/fig2_slot.png'); print('fig2 written')
