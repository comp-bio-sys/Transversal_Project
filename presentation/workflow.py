"""Vector overview of the data preparation, fitting and evaluation procedure."""

import numpy as np
import json
from pathlib import Path
import matplotlib.image as mimage
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon
from matplotlib.colors import to_rgb

INK = "#202A31"
MUTED = "#586A75"
BLUE, TEAL, PURPLE, ORANGE = "#2878A5", "#078478", "#9268A1", "#BE6C37"
DATA = Path(__file__).resolve().parent / "data"


def draw_workflow(ax, scale=1.0, title=True, schematic_note=True):
    """Draw on a 180 by 66 canvas; all illustrative marks are schematic."""
    ax.set(xlim=(0, 180), ylim=(66, 0), aspect="equal")
    ax.axis("off")
    culture_image = mimage.imread(DATA / "cultures.png")
    compound = json.loads((DATA / "caffeine.json").read_text())

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
        h,w = culture_image.shape[:2]
        control = culture_image[int(h*.15):int(h*.97),:w//2]
        width = 9*s
        height = width * control.shape[0] / control.shape[1]
        ax.imshow(control,extent=(x-width/2,x+width/2,y+height/2,y-height/2),
                  interpolation="antialiased",zorder=3)

    def molecule(x, y, c=INK, s=1):
        coords = np.asarray(compound["xy"])*[s,-s]+[x,y]
        for start,end,order in compound["bonds"]:
            a,b = coords[start],coords[end]
            direction = (b-a)/np.linalg.norm(b-a)
            offset = .06*s*np.array([-direction[1],direction[0]])
            for shift in ([-offset,offset] if order==2 else [np.zeros(2)]):
                line([a+shift,b+shift],c,.8)
        for (xx,yy),symbol in zip(coords,compound["symbols"]):
            if symbol in ("N","O"):
                ax.text(xx,yy,symbol,fontsize=6.5*scale,color=c,ha="center",va="center",
                        bbox={"fc":"white","ec":"none","pad":.1},zorder=5)

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
    ax.imshow(culture_image,extent=(1.5,25.5,16.7,4.7),interpolation="antialiased",zorder=3)
    text(7.5,17.2,"DMSO control",8.5,BLUE,True,ha="center")
    text(7.5,19.1,"Baseline input",7.4,MUTED,ha="center")
    text(19.5,17.2,"Treated cells",8.5,ORANGE,True,ha="center")
    text(19.5,19.1,"Measured target",7.4,MUTED,ha="center")
    text(40,4.9,"Compound structure / SMILES",8.1,INK,True,ha="center")
    molecule(40,9.5,INK,.93)
    text(40,12.8,"Caffeine shown as an example",7.1,MUTED,ha="center")
    text(28,15.4,"10 µM · 24 h · 978 genes",8.6,bold=True)
    text(28,17.8,"298,470 selected profiles",7.8)
    text(28,19.2,"LINCS Level 3 · fixed split IDs",7.8)
    arrow((55.2,10),(57.8,10))

    step(2,60,1,"Pair each treatment with controls")
    cells(65,8.4,TEAL,.85)
    text(65,11.7,"DMSO",8,ha="center")
    arrow((70,8.5),(73,8.5),TEAL)
    text(82,4.6,"Replicate profiles",8.2,TEAL,True,ha="center")
    values = np.array([[.25,.75,.5,.3],[.35,.65,.6,.45],[.3,.7,.4,.3]])
    rgb = np.array(to_rgb(TEAL))
    for i,row in enumerate(values):
        text(74.5,6.5+i*1.7,f"r{i+1}",6.8,MUTED,ha="right")
        for j,v in enumerate(row):
            rect(76+j*3.2,6.3+i*1.7,2.8,1.35,rgb*v+(1-v)*np.ones(3))
    for x in [76,101]:
        for j,label in enumerate(["g₁","g₂","…","g₉₇₈"]):
            text(x+j*3.2+1.4,11.7,label,6.8,MUTED,ha="center")
    arrow((90,8.5),(99,8.5),TEAL)
    text(94.5,6.3,"Mean",7.8,TEAL,True,ha="center")
    text(107,4.0,"Mean DMSO\nexpression",8.2,TEAL,True,ha="center")
    for j,v in enumerate(values.mean(axis=0)):
        rect(101+j*3.2,7.6,2.8,1.8,rgb*v+(1-v)*np.ones(3))
    text(61,14.5,"Match detection plate + cell + time + bead batch",9,bold=True)
    text(61,17.6,"269,364 paired profiles · 47,995 measured controls",8.5)
    text(61,19.3,"DMSO reference: 978-gene replicate mean; not a network",7.4,MUTED)
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
    step(4,2,24,"Fit three chemical MLPs and a PRnet comparator")
    text(28,28.4,"Chemical input",8.7,MUTED,True)
    text(64,28.4,"Network",8.7,MUTED,True)
    text(101,27.1,"Predicted treated expression",8.3,MUTED,True,ha="center")
    text(101,28.8,"One value for each of 978 genes",7.4,MUTED,ha="center")
    molecule(9,36.5,INK,1.3)
    text(9,41,"Compound SMILES",8.2,ha="center")
    text(9,43,"Example: caffeine",7,MUTED,ha="center")
    line([(17,31.7),(17,46.7)],BLUE,.75)
    arrow((14,36.5),(17,36.5))
    text(49,26.4,"DMSO expression → every model",8.1,TEAL)
    line([(60,28.5),(60,47.2)],TEAL,.9)
    rows = [(31.7,BLUE,"FCFP4","1,024 feature bits","MLP"),
            (36.7,TEAL,"ECFP4","1,024 atom bits","MLP"),
            (41.7,PURPLE,"ChemicalVAE","196 frozen values","MLP"),
            (46.7,ORANGE,"FCFP4 × dose","Dose-scaled bits","PRnet")]
    rect(21.2,34.8,91,4.2,"#F0F7F5",TEAL,.8)
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
        arrow((82,y),(88,y),c)
        for j,value in enumerate(["ŷ₁","ŷ₂","…","ŷ₉₇₈"]):
            rect(90.5+j*5.4,y-1.1,4.9,2.4,"white",c,.7)
            text(92.95+j*5.4,y+.1,value,8.8,c,ha="center",va="center")
        if model == "PRnet":
            text(101,y+1.8,"Mean of 32 test draws",7.5,c,ha="center")
    text(23,49.6,"MLPs: DMSO expression + predicted change     |     512 → 256 units; mean change + variance",8.2)

    line([(119,24),(119,50.3)],"#D0DADF",.8)
    arrow((113,36.7),(122,36.7),INK)
    step(5,123,24,"Select from recorded validation")
    rect(125,30,18,5,"#EAF2F7")
    text(134,30.7,"Train batches\nGaussian NLL loss",8.5,BLUE,True,"center")
    arrow((144,32.5),(149,32.5))
    rect(150,30,27,5,"#EDF5F2")
    text(163.5,31.5,"Validation NLL",9.5,TEAL,True,"center")
    line([(164,29.5),(164,28.3),(134,28.3),(134,29.7)],MUTED,.7)
    arrow((134,29.2),(134,30),lw=.7)
    arrow((163.5,35.2),(163.5,38.2))
    text(125,36.3,"Fit to measured treated expression y",8.3)
    rect(125,39,52,6,"white",TEAL)
    text(151,39.8,"ECFP4: lowest recorded validation NLL",8.8,TEAL,True,"center")
    text(151,42.1,"Lower learning rate: 0.001 → 0.0003",8.5,TEAL,False,"center")
    text(125,46,"ECFP4 validation NLL: 0.0846 → 0.0837",8.7,bold=True)
    text(125,48.1,"Selected ECFP4 vs PRnet + measured DMSO reference",8.1)
    text(125,50.2,"Same cohort · best validation checkpoints · 3 seeds",7.8)
    if schematic_note:
        text(2,51.4,"Cells, structures and expression strips are schematic; no measured values are drawn here.",7.8,MUTED)
    arrow((163.5,50),(163.5,52.5))

    # Separate test settings: GEO uses a LINCS reference, not GEO as model input.
    rect(0,53,180,13,"#F4F7F9")
    step(6,2,54,"Evaluate selected ECFP4 and retrained PRnet")
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
    text(134,61.2,"7 studies × 6 genes · dose/time matched\nPredict with cell-mean LINCS DMSO expression",8.2)
    if title:
        text(0,-2.3,"How the benchmark was carried out",15,bold=True)

