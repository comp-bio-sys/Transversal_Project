"""Vector overview of the data preparation, fitting and evaluation procedure."""

import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle, FancyArrowPatch, Polygon
from matplotlib.colors import to_rgb

INK = "#202A31"
MUTED = "#586A75"
BLUE, TEAL, PURPLE, ORANGE = "#2878A5", "#078478", "#9268A1", "#BE6C37"


def draw_workflow(ax, scale=1.0, title=True):
    """Draw on a 180 by 66 canvas; all illustrative marks are schematic."""
    ax.set(xlim=(0, 180), ylim=(66, 0), aspect="equal")
    ax.axis("off")

    def text(x, y, s, size=10, color=INK, bold=False, ha="left", va="top"):
        return ax.text(x, y, s, fontsize=size * scale, color=color,
                       weight="semibold" if bold else "normal", ha=ha, va=va,
                       linespacing=1.25)

    def line(points, c=MUTED, lw=.85, style="-"):
        xy = np.array(points)
        ax.plot(xy[:, 0], xy[:, 1], color=c, lw=lw * scale, ls=style, zorder=2)

    def arrow(a, b, c=MUTED, lw=1, style="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                     mutation_scale=9 * scale, lw=lw * scale, color=c,
                     linestyle=style, shrinkA=1, shrinkB=1, zorder=3))

    def rect(x, y, w, h, fc="white", ec="none", lw=.8):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                              lw=lw * scale))

    def step(n, x, y, label):
        ax.add_patch(Circle((x + 1.5, y + 1.5), 1.5, fc=INK))
        text(x + 1.5, y + 1.5, str(n), 10, "white", True, "center", "center")
        text(x + 4.5, y + .15, label, 12, bold=True)

    def cells(x, y, c, s=1):
        ax.add_patch(Ellipse((x, y + .5*s), 9*s, 5.5*s, fc="#F6F9FA", ec=c,
                             lw=.8 * scale))
        for dx, dy in [(-2,-.3),(-.2,1),(1.6,-.6),(.1,-1.2),(2,1.1)]:
            ax.add_patch(Ellipse((x+dx*s,y+dy*s),2.4*s,1.8*s,angle=dx*18,
                                 fc=c, ec="white", alpha=.55, lw=.5*scale))
            ax.add_patch(Circle((x+dx*s,y+dy*s),.32*s,fc=c))

    def molecule(x, y, c=INK, s=1):
        pts = [(x+1.9*s*np.cos(t), y+1.9*s*np.sin(t))
               for t in np.linspace(0,2*np.pi,7)]
        line(pts,c,1)
        line([(x+1.9*s,y),(x+3.2*s,y-.8*s),(x+4.2*s,y+.1*s)],c,1)
        line([(x-1.9*s,y),(x-3*s,y-1.1*s)],c,1)
        text(x-4*s,y-2.2*s,"N",7,c)

    def matrix(x, y, w=10, h=4, c=TEAL, rows=3, binary=False):
        rgb = np.array(to_rgb(c))
        for i in range(rows):
            for j in range(10):
                value = ((j*3+i*5+2)%11)/10
                if binary:
                    value = float(value > .5)
                col = rgb*value + (1-value)*np.ones(3)
                rect(x+j*w/10, y+i*h/rows, w/10-.13, h/rows-.13, col)

    def network(x,y,c):
        layers = [[(x,y-.9),(x,y+.9)],
                  [(x+2.4,y-1.5),(x+2.4,y),(x+2.4,y+1.5)],
                  [(x+4.8,y-.9),(x+4.8,y+.9)]]
        for a,b in zip(layers,layers[1:]):
            for aa in a:
                for bb in b:
                    line([aa,bb],c,.45)
        for layer in layers:
            for p in layer:
                ax.add_patch(Circle(p,.36,fc="white",ec=c,lw=.7*scale,zorder=4))

    # Preparation: each displayed expression row represents a bulk profile.
    rect(0,0,55,21,"#F4F7F9")
    rect(58,0,60,21,"#F1F7F5")
    rect(121,0,59,21,"#F7F5F0")
    step(1,2,1,"Assemble observed experiments")
    cells(8,9,TEAL)
    molecule(19,8,BLUE,.8)
    text(14,8,"+",13,ha="center")
    arrow((24,9),(29,9))
    matrix(31,6,20,5,BLUE,4)
    text(41,11.7,"978 measured genes",8.5,ha="center")
    text(3,14.5,"LINCS Level 3 · 10 µM · 24 h",10,bold=True)
    text(3,17.6,"298,470 selected treatment profiles + fixed split IDs",8.5)
    arrow((55.2,10),(57.8,10))

    step(2,60,1,"Pair each treatment with controls")
    cells(65,8.4,TEAL,.8)
    text(65,11.7,"DMSO",8,ha="center")
    arrow((70,8.5),(74,8.5),TEAL)
    matrix(75,6,9,5,TEAL,3)
    arrow((85,8.5),(89,8.5),TEAL)
    matrix(90,7,9,3,TEAL,1)
    text(104,6.8,"Average replicates\n→ control vector c",8.4)
    text(61,14.5,"Match detection plate + cell + time + bead batch",9,bold=True)
    text(61,17.6,"269,364 paired profiles · 47,995 measured controls",8.5)
    arrow((118.2,10),(120.8,10))

    step(3,123,1,"Screen, then preserve the splits")
    for x,w,c,label,num in [(124,16,BLUE,"Train","210,438"),
                            (142.5,16,TEAL,"Validation","25,190"),
                            (161,16,ORANGE,"Test","27,765")]:
        rect(x,6,w,2,c)
        text(x+w/2,8.8,label,8.7,c,True,"center")
        text(x+w/2,11.4,num + " profiles",8.4,INK,True,"center")
    text(124,14.5,"Exclude SKBR3 and close external analogues",9,bold=True)
    text(124,17.6,"Compound-disjoint · scale controls using train only",8.5)

    line([(151,21.1),(151,22.1),(4,22.1),(4,23)],lw=.7)
    arrow((4,23),(4,24),lw=.7)
    step(4,2,24,"Encode the same molecule; fit separate predictors")
    text(28,28.4,"Chemical input",8.7,MUTED,True)
    text(64,28.4,"Network",8.7,MUTED,True)
    text(93,28.4,"978 predicted genes",8.7,MUTED,True)
    molecule(9,36.5,INK,1.3)
    text(9,41,"Structure\n(SMILES)",9,ha="center")
    line([(17,31.7),(17,46.7)],BLUE,.75)
    arrow((14,36.5),(17,36.5))
    text(49,25.8,"Measured control c → every model",8.5,TEAL)
    line([(60,28.5),(60,47.2)],TEAL,.9)
    rows = [(31.7,BLUE,"FCFP4","1,024 feature bits","MLP"),
            (36.7,TEAL,"ECFP4","1,024 atom bits","MLP"),
            (41.7,PURPLE,"ChemicalVAE","196 frozen values","MLP"),
            (46.7,ORANGE,"FCFP4 × dose","Dose-scaled bits","PRnet")]
    for y,c,name,detail,model in rows:
        arrow((17,y),(22,y),c)
        matrix(22,y-1.2,6.5,2.4,c,2,binary=model=="MLP" and name!="ChemicalVAE")
        text(30,y-1.7,name,10,c,True)
        text(30,y+.3,detail,7.8)
        arrow((52,y),(63,y),c)
        ax.add_patch(Circle((60,y),.37,fc=TEAL,zorder=5))
        if model == "PRnet":
            ax.add_patch(Polygon([(64,y-1.4),(67,y-.6),(67,y+.6),(64,y+1.4)],
                                 fc="#F4E6DC",ec=c,lw=.7*scale))
            ax.add_patch(Polygon([(68,y-.6),(71,y-1.4),(71,y+1.4),(68,y+.6)],
                                 fc="#F4E6DC",ec=c,lw=.7*scale))
            arrow((67,y),(68,y),c,.6)
        else:
            network(65,y,c)
        text(72,y-1.1,model,10,c,True)
        arrow((82,y),(90,y),c)
        matrix(92,y-1.2,7,2.4,c,1)
        text(101,y-1.15,"c + Δ̂" if model=="MLP" else "μ̂",10,c)
        if model == "PRnet":
            text(101,y+.8,"Test: 32 draws",7.8,c)
    text(23,49.1,"MLPs: 512 → 256 units; mean change + variance     |     PRnet: published generative architecture",8.2)

    line([(119,24),(119,50.3)],"#D0DADF",.8)
    arrow((113,36.7),(122,36.7),INK)
    step(5,123,24,"Select using validation")
    rect(125,30,18,5,"#EAF2F7")
    text(134,30.7,"Train batches\nGaussian NLL loss",8.5,BLUE,True,"center")
    arrow((144,32.5),(149,32.5))
    rect(150,30,27,5,"#EDF5F2")
    text(163.5,31.5,"Validation NLL",9.5,TEAL,True,"center")
    line([(164,29.5),(164,28.3),(134,28.3),(134,29.7)],MUTED,.7)
    arrow((134,29.2),(134,30),lw=.7)
    arrow((163.5,35.2),(163.5,38.2))
    text(125,36.3,"Fit to measured treated expression y",8.3)
    rect(139,39,38,5,"white",TEAL)
    text(158,40.5,"Save best checkpoint",9.5,TEAL,True,"center")
    text(125,45,"AdamW · equal 3-setting grid · 3 seeds",9,bold=True)
    text(125,48,"Control-only MLP + mean / linear baselines",8.8)
    text(125,50.2,"Diagnostics: chemical shuffle · control noise · coverage",7.8)
    text(2,51.4,"Cells, structures and expression strips are schematic; no measured values are drawn here.",7.8,MUTED)
    arrow((163.5,50),(163.5,52.5))

    # Separate test settings: GEO uses a LINCS reference, not GEO as model input.
    rect(0,53,180,13,"#F4F7F9")
    step(6,2,54,"Compare predictions with held-out measurements")
    text(177,54.4,"MSE · response correlation · direction agreement",9,MUTED,ha="right")
    for x in [60,120]:
        line([(x,58),(x,65)],"#CDD8DD",.7)
    molecule(7,61.3,BLUE,.7)
    text(13,58.5,"New compounds",10,BLUE,True)
    text(13,61.2,"474 distant compounds / 16,531 profiles\nNearest-train ECFP4 Tanimoto ≤0.7",8.5)
    cells(66,61.3,TEAL,.65)
    text(72,58.5,"New cell context: SKBR3",10,TEAL,True)
    text(72,61.2,"574 profiles / 137 compounds\nMeasured SKBR3 DMSO controls",8.5)
    for j,dy in enumerate([-1,1,-1]):
        arrow((125+j*2,62),(125+j*2,62+dy*2),PURPLE,.8)
    text(134,58.5,"Independent GEO measurements",10,PURPLE,True)
    text(134,61.2,"7 studies × 6 genes · dose/time matched\nPredict with cell-mean LINCS controls",8.2)
    if title:
        text(0,-2.3,"How the benchmark was carried out",15,bold=True)

