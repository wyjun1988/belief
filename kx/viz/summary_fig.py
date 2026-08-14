"""decoration 시퀀스 요약 그림 — 방/구역 지도 + 액자 이동 궤적/높이.

    $P kx/viz/summary_fig.py    (docs/img/scenegraph_decoration.png 생성)
"""
import json, os, sys, numpy as np, matplotlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
matplotlib.use("Agg")
matplotlib.rcParams["font.family"]="AppleGothic"
matplotlib.rcParams["axes.unicode_minus"]=False
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

d="data/seq/Apartment_release_decoration_seq137_M1292"
g=json.load(open(d+"/graph_gtdepth.json"))
gt=json.load(open(d+"/gt/objects.json"))["instances"]
z=np.load(d+"/regions_gtdepth.npz")
zones,rooms,lo,res=z["zones"],z["rooms"],z["lo"],float(z["res"])
names=g["regions"]["zone_names"]
poses=np.loadtxt(d+"/pose/poses.txt").reshape(-1,4,4)
COL={"kitchen":"#f0aa3c","living":"#4aa0f0","dining":"#78c878","bedroom":"#c86ec8"}

fig,axes=plt.subplots(1,2,figsize=(15,6.2))
ax=axes[0]
from kx.graph.frames import floor_basis
# ⚠️ 구역 래스터의 축은 **중력정렬 바닥좌표 (u,v)** 다 — world 의 (x,z) 가 아니다.
# 이 시퀀스는 e2 = -z 라서 그냥 (x,z) 로 찍으면 바닥 지도만 z축으로 뒤집힌 채
# 물체 마커 위에 얹힌다(2026-08-14 에 잡은 그림 버그).
E1,E2,_=floor_basis(np.array(g["regions"]["up"],float))
U=np.arange(zones.shape[0])*res+lo[0]; V=np.arange(zones.shape[1])*res+lo[1]
img=np.ones(zones.shape+(3,))
for zi,zn in enumerate(names):
    c=np.array(matplotlib.colors.to_rgb(COL.get(zn,"#999999")))
    img[zones==zi]=0.45*c+0.55

def to_world(A):
    """(u,v) 배열 → world (x,z) 축으로 정렬된 배열 + extent. 축정렬 basis 전제."""
    assert abs(abs(E1[0])+abs(E1[2])-1)<1e-6, "basis 가 축정렬이 아니다"
    if abs(E1[0])>0.5:                       # axis0=u→x, axis1=v→z
        Ax, Az, out = U*E1[0], V*E2[2], np.swapaxes(A,0,1)
    else:                                    # axis0=u→z, axis1=v→x
        Ax, Az, out = V*E2[0], U*E1[2], A
    if Ax[0]>Ax[-1]: out, Ax = out[:,::-1], Ax[::-1]
    if Az[0]>Az[-1]: out, Az = out[::-1], Az[::-1]
    return out, [Ax[0],Ax[-1],Az[0],Az[-1]]

im, ext = to_world(img)
ax.imshow(im, origin="lower", extent=ext, interpolation="nearest")
XX = np.linspace(ext[0],ext[1],im.shape[1]); ZZ = np.linspace(ext[2],ext[3],im.shape[0])
for r in range(1,int(rooms.max())+1):
    rm, _ = to_world((rooms==r).astype(float)[...,None])
    ax.contour(XX,ZZ,rm[...,0],levels=[0.5],colors="k",linewidths=2.2)
t=poses[:,:3,3]
ax.plot(t[:,0],t[:,2],lw=0.7,color="0.35",alpha=0.85)
o=[v for v in g["objects"].values() if v["name"]=="BlackSquarePictureFrame"][0]
st=[p for p in o["placements"] if p["stable"]]
P=np.array([p["position"] for p in st]); allp=np.array([p["position"] for p in o["placements"]])
ax.plot(allp[:,0],allp[:,2],"-",color="crimson",lw=1.0,alpha=0.5)
ax.plot(P[:,0],P[:,2],"o-",color="crimson",ms=13,lw=2.5,mec="k",zorder=5)
for i,(p,pl) in enumerate(zip(P,st)):
    ax.annotate("%d. %s"%(i+1,pl["support"]),(p[0],p[2]),textcoords="offset points",
                xytext=(12,10),fontsize=10,weight="bold",
                bbox=dict(fc="white",alpha=0.9,ec="crimson",lw=1))
ax.set_title("방(굵은 선 = 벽으로 갈림) x 구역(색) + 액자 이동",fontsize=13)
ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal")
ax.legend(handles=[Patch(fc=COL[n],label="%s  %.0f m2"%(n,g["regions"]["summary"][n]["area_m2"]))
                   for n in names]
          +[plt.Line2D([],[],color="0.35",lw=1,label="관찰자 궤적"),
            plt.Line2D([],[],color="crimson",marker="o",lw=2,label="액자 정지 위치")],
          loc="lower left",fontsize=9,framealpha=0.9)

ax=axes[1]
gtP=np.array(gt[str(o["instance_id"])]["positions"])
ax.plot(np.arange(len(gtP)),gtP[:,1],color="0.5",lw=2.5,label="GT 높이")
obs_f,obs_y=[],[]
for p in o["placements"]:
    obs_f+=[p["start_frame"],p["end_frame"]]; obs_y+=[p["position"][1]]*2
ax.plot(obs_f,obs_y,color="crimson",lw=1.7,alpha=0.85,label="그래프 belief")
for i,p in enumerate(st):
    ax.axvspan(p["start_frame"],p["end_frame"],color="crimson",alpha=0.16)
    ax.text((p["start_frame"]+p["end_frame"])/2,2.30,"%d. %s"%(i+1,p["support"]),
            ha="center",fontsize=9,weight="bold")
for m in gt[str(o["instance_id"])]["moves"]:
    ax.axvline(m["end_idx"],color="k",ls="--",lw=1.2)
ax.set_ylim(0,2.55); ax.set_xlabel("프레임 (10Hz)"); ax.set_ylabel("높이 y [m]")
ax.set_title("액자 높이: 선반 1.86m -> 탁자 0.47m -> 선반 1.95m",fontsize=13)
ax.legend(loc="lower left",fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("docs/img/scenegraph_decoration.png",dpi=130)
print("saved docs/img/scenegraph_decoration.png")
