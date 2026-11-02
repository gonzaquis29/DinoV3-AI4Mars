"""Bar chart of test mIoU: probe, LoRA by zone and DeepLabV3 baselines. Values come from results/estudio_principal/{P2_table,baseline_table}.csv."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
labels=["Linear probe\n(4.6 k)","LoRA early\n(221 k)","LoRA mid\n(221 k)","LoRA late\n(221 k)","DeepLabV3\nImageNet (39.6 M)","DeepLabV3\nno pretrain."]
m1=[0.674,0.681,0.679,0.681,0.667,0.628]; m2=[0.723,0.730,0.728,0.731,0.713,0.681]
x=np.arange(len(labels)); w=0.38
fig,ax=plt.subplots(figsize=(7.2,3.1))
b1=ax.bar(x-w/2,m1,w,color="#2a6fdb",label="M1"); b2=ax.bar(x+w/2,m2,w,color="#e07b39",label="M2")
for b in list(b1)+list(b2): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.003,f"{b.get_height():.3f}",ha="center",fontsize=7)
ax.set_xticks(x,labels,fontsize=7.5); ax.set_ylim(0.55,0.78); ax.set_ylabel("test mIoU"); ax.legend(frameon=False,fontsize=8,loc="upper right")
ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
fig.savefig("/home/gonzalo/Escritorio/Proyectos PUCP/DinoV3-AI4Mars/results/estudio_principal/figures/comparison_main.png",dpi=200)
