# -*- coding: utf-8 -*-
"""Day10 配图一键重绘：python make_figures.py
产出 CNN 架构图、FlattenConsecutive 机制图（PNG+SVG）和每 10000 步采样的 loss 曲线。
依赖：matplotlib、numpy。
"""
# -*- coding: utf-8 -*-
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'], 'axes.unicode_minus':False})
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'img'; OUT.mkdir(exist_ok=True)
C={'ink':'#172033','muted':'#526174','line':'#64748b','data':'#e2e8f0','embed':'#d1fae5','flat':'#dbeafe','compute':'#ffedd5','wide':'#ede9fe','out':'#fee2e2','blue':'#2563eb','green':'#059669','orange':'#c2410c','purple':'#7c3aed','red':'#b91c1c','grid':'#e2e8f0'}

def rounded(ax,x,y,w,h,title,shape,fill,stroke,note='',fs=14):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=12',facecolor=fill,edgecolor=stroke,linewidth=1.8,zorder=3))
    ax.text(x+w/2,y+h*.62,title,ha='center',va='center',fontsize=fs,color=C['ink'],weight='bold',zorder=4)
    ax.text(x+w/2,y+h*.28,shape,ha='center',va='center',fontsize=12,color=C['muted'],zorder=4)
    if note: ax.text(x+w/2,y-20,note,ha='center',va='top',fontsize=10,color=C['muted'],zorder=4)

def arrow(ax,a,b,color=C['blue'],label='',dy=20,fs=10.5):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,linewidth=2.2,color=color,shrinkA=5,shrinkB=5,zorder=2))
    if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+dy,label,ha='center',va='center',fontsize=fs,color=color,weight='bold',zorder=5)

# ==================== 1. 清晰的 CNN 流水线 + 感受野树 ====================
fig,ax=plt.subplots(figsize=(22,13),dpi=150); ax.set_xlim(0,2200); ax.set_ylim(0,1300); ax.axis('off')
ax.text(1100,1250,'Day10 · 层级 CNN（WaveNet 风格）字符模型',ha='center',fontsize=24,weight='bold',color=C['ink'])
ax.text(1100,1212,'上：实际张量流水线　|　下：感受野从 1→2→4→8 的合并树',ha='center',fontsize=14,color=C['muted'])
steps=[('输入 Xb','(B,8)',C['data'],C['line'],'字符索引'),('Embedding','(B,8,24)',C['embed'],C['green'],'27→24'),('FlattenConsecutive','(B,4,48)',C['flat'],C['blue'],'n=2: T÷2,C×2'),('Linear + BN + Tanh','(B,4,128)',C['compute'],C['orange'],'48→128'),('FlattenConsecutive','(B,2,256)',C['wide'],C['purple'],'n=2: T÷2,C×2'),('Linear + BN + Tanh','(B,2,128)',C['compute'],C['orange'],'256→128'),('Flatten + squeeze','(B,256)',C['wide'],C['purple'],'T=1，去掉 time 维'),('Linear + BN + Tanh','(B,128)',C['compute'],C['orange'],'256→128'),('输出 logits','(B,27)',C['out'],C['red'],'27 类交叉熵')]
BW,BH,GAP=190,116,48; X0,Y0=38,890
ops=['lookup','相邻两行拼接','共享 Linear','扩大感受野','共享 Linear','squeeze(1)','共享 Linear','softmax / loss']
for i,(title,shape,fill,stroke,note) in enumerate(steps):
    x=X0+i*(BW+GAP); title=title.replace(' + ',' +\n') if title=='Linear + BN + Tanh' else title; title=title.replace('FlattenConsecutive','FlattenConsecutive\n') if title.startswith('Flatten') else title
    rounded(ax,x,Y0,BW,BH,title,shape,fill,stroke,note,13.5)
    if i<len(steps)-1: arrow(ax,(x+BW+4,Y0+BH/2),(x+BW+GAP-4,Y0+BH/2),label=ops[i],dy=23,fs=10.5)
# tree positions
ax.text(1100,795,'感受野合并树：每次合并相邻两组，T 减半、每个高层节点看到的字符数翻倍',ha='center',fontsize=15,weight='bold',color=C['ink'])
leaf=np.linspace(560,1640,8); l1=(leaf[0::2]+leaf[1::2])/2; l2=(l1[0::2]+l1[1::2])/2; root=(l2[0]+l2[1])/2
for xs,y,w,h,title,fill,stroke in [(leaf,650,84,48,'RF=1',C['embed'],C['green']),(l1,535,170,54,'RF=2',C['flat'],C['blue']),(l2,420,260,58,'RF=4',C['wide'],C['purple'])]:
    for x in xs: rounded(ax,x-w/2,y,w,h,title,'局部组',fill,stroke,fs=11.5)
rounded(ax,root-190,300,380,64,'RF=8：完整上下文','8 字符 → 一个全局表示',C['out'],C['red'],fs=14)
for child,cy,pcoords,py,pw in [(leaf,650,l1,535,170),(l1,535,l2,420,260),(l2,420,[root],300,380)]:
    for j,px in enumerate(pcoords):
        for cx in child[2*j:2*j+2]: ax.plot([cx,px],[cy,py+({'535':54,'420':58,'300':64}.get(str(py),54))],color=C['line'],lw=1.3,zorder=1)
        ax.text(px,py-15,'merge ×2',ha='center',fontsize=9.5,color=C['blue'])
ax.text(180,510,'T=8\n→4\n→2\n→1',ha='center',fontsize=17,weight='bold',color=C['ink'])
ax.text(180,410,'C=24\n→48\n→256',ha='center',fontsize=12,color=C['muted'])
ax.text(1100,220,'核心规律：每次 T 减半、局部通道拼接；高层节点感受野翻倍，最后覆盖整个 8 字符窗口。',ha='center',fontsize=13,color=C['red'],weight='bold')
fig.savefig(OUT/'day10-cnn-tree.png',bbox_inches='tight',facecolor='white'); fig.savefig(OUT/'day10-cnn-tree.svg',bbox_inches='tight',facecolor='white'); plt.close(fig)

# ==================== 2. FlattenConsecutive 形状变换 ====================
fig,ax=plt.subplots(figsize=(15,8),dpi=150); ax.set_xlim(0,1500); ax.set_ylim(0,800); ax.axis('off')
ax.text(750,755,'FlattenConsecutive(2)：相邻两行拼成一行更宽的通道',ha='center',fontsize=22,weight='bold',color=C['ink'])
ax.text(750,710,'x.view(B, T//2, C*2) —— 数值不变，只改变形状解释；T=1 时再 squeeze(1)',ha='center',fontsize=13,color=C['muted'])
pair=['#86efac','#86efac','#93c5fd','#93c5fd','#fcd34d','#fcd34d','#f0abfc','#f0abfc']
def draw_grid(ax,x0,y0,rows,cols,cw,ch,colors,title,shape):
    for r in range(rows):
        for c in range(cols):
            fill=colors[r] if len(colors)==rows else colors[r%len(colors)]
            ax.add_patch(Rectangle((x0+c*cw,y0+(rows-1-r)*ch),cw-3,ch-3,facecolor=fill,edgecolor='#334155',lw=1.0))
            if cw>=45: ax.text(x0+c*cw+(cw-3)/2,y0+(rows-1-r)*ch+(ch-3)/2,f'c{c}',ha='center',va='center',fontsize=8,color=C['ink'])
    ax.text(x0+cols*cw/2,y0+rows*ch+25,title,ha='center',fontsize=14,weight='bold',color=C['ink'])
    ax.text(x0+cols*cw/2,y0-27,shape,ha='center',fontsize=12,color=C['muted'])
draw_grid(ax,100,190,8,4,50,48,pair,'① 输入：每个时间位置有 C=4 个通道','(B=1, T=8, C=4)')
draw_grid(ax,1030,286,4,8,50,48,['#86efac','#93c5fd','#fcd34d','#f0abfc'],'② 输出：每个组有 2C=8 个通道','(B=1, T=4, C=8)')
ax.add_patch(FancyArrowPatch((555,395),(1005,395),arrowstyle='-|>',mutation_scale=20,color=C['blue'],lw=3))
ax.text(780,435,'view：T=8 → 4，C=4 → 8',ha='center',fontsize=14,color=C['blue'],weight='bold')
ax.text(300,145,'t0+t1',ha='center',fontsize=9.5,color=C['muted']); ax.text(400,145,'t2+t3',ha='center',fontsize=9.5,color=C['muted']); ax.text(500,145,'t4+t5',ha='center',fontsize=9.5,color=C['muted']); ax.text(600,145,'t6+t7',ha='center',fontsize=9.5,color=C['muted'])
ax.text(750,90,'view(B, T//2, C*2) → (B,4,8)；每个 g = 原 t(2g) 的 c0..c3 接 t(2g+1) 的 c0..c3',ha='center',fontsize=12.5,color=C['ink'],weight='bold')
ax.add_patch(FancyBboxPatch((1070,70),340,62,boxstyle='round,pad=0.02,rounding_size=10',facecolor='#fff7ed',edgecolor=C['orange'],lw=1.5))
ax.text(1240,101,'若 T//2=1：\nsqueeze(1) → (B, 2C)',ha='center',va='center',fontsize=11,color=C['orange'],weight='bold')
fig.suptitle('FlattenConsecutive(2)：把“相邻两行”并成“一行更宽” = 通道拼接',fontsize=11.5,weight='bold',y=1.03)
fig.tight_layout(); fig.savefig(OUT/'day10-flatten.png',bbox_inches='tight',facecolor='white'); fig.savefig(OUT/'day10-flatten.svg',bbox_inches='tight',facecolor='white'); plt.close(fig)

# ==================== 3. 真实 notebook 打印点的 loss 图 ====================
steps=np.arange(0,200000,10000); loss=np.array([3.3167,2.0576,2.0723,2.5134,2.1476,1.7836,2.2592,1.9331,1.6875,2.0395,1.7736,1.9569,1.7465,1.8126,1.7406,1.7466,1.8806,1.6266,1.6476,1.8555])
fig,ax=plt.subplots(figsize=(12.5,6.5),dpi=160)
ax.plot(steps,loss,color=C['blue'],lw=2.4,marker='o',ms=5.5,mfc='white',mew=1.8,mec=C['blue'],label='batch loss（每 10000 步采样）')
ax.axvline(150000,color=C['orange'],lw=1.8,ls='--',label='lr: 0.1 → 0.01')
ax.annotate('学习率衰减点\n150k steps',xy=(150000,1.7466),xytext=(158000,2.28),arrowprops=dict(arrowstyle='->',color=C['orange'],lw=1.4),fontsize=11,color=C['orange'],weight='bold')
ax.axhline(3.3,color='#94a3b8',ls=':',lw=1.2); ax.text(2500,3.34,'均匀预测基线 ln(27)≈3.30',fontsize=10.5,color=C['muted'])
ax.set_title('WaveNet 字符模型训练轨迹（真实 notebook 打印点）',fontsize=17,weight='bold',color=C['ink'],pad=14)
ax.set_xlabel('训练步数',fontsize=12); ax.set_ylabel('loss（每 10000 步记录）',fontsize=12); ax.set_xlim(-5000,205000); ax.set_ylim(1.45,3.55); ax.set_xticks(np.arange(0,200001,25000)); ax.grid(True,color=C['grid'],lw=.8,alpha=.8); ax.legend(frameon=False,fontsize=11,loc='upper right')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT/'day10-loss-curve.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
print('redraw complete',OUT)
