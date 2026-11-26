import math
from PIL import Image,ImageDraw,ImageFont
W,H=1600,900
BG=(14,22,40); WH=(240,244,250); CY=(70,200,240); OR=(255,150,50); RD=(255,90,90); GR=(110,220,140); GY=(150,160,180)
FB="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
def F(s): return ImageFont.truetype(FB,s)
OUT="episodes/ep-hubble/raw/schematics/"
def new(title,sub):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im); d.text((W//2,60),title,font=F(60),fill=WH,anchor="mm"); d.text((W//2,128),sub,font=F(40),fill=GY,anchor="mm"); return im,d
def txt(d,xy,s,size=44,fill=WH,anchor="mm"): d.text(xy,s,font=F(size),fill=fill,anchor=anchor)
def lab(d,xy,s,size,fill):
    b=d.textbbox(xy,s,font=F(size),anchor='mm'); d.rectangle((b[0]-12,b[1]-8,b[2]+12,b[3]+8),fill=BG); d.text(xy,s,font=F(size),fill=fill,anchor='mm')
def arrow(d,p,q,fill,w=6):
    d.line([p,q],fill=fill,width=w); a=math.atan2(q[1]-p[1],q[0]-p[0])
    for s in (-0.4,0.4): d.line([q,(q[0]-24*math.cos(a+s),q[1]-24*math.sin(a+s))],fill=fill,width=w)
def box(d,x0,y0,x1,y1,label,col=CY,size=42):
    d.rounded_rectangle((x0,y0,x1,y1),radius=14,outline=col,width=6,fill=(26,38,66))
    if label: txt(d,((x0+x1)//2,(y0+y1)//2),label,size,col)
def scene(title,sub,flat):
    im,d=new(title,sub); cx=W//2; yv=800; f=340; xm=380
    prof=lambda x:x*x/(4*f)-flat*(abs(x/xm)**4)*f
    d.line([(cx+x,yv-prof(x)) for x in range(-xm,xm+1,4)],fill=CY,width=14)
    d.rectangle((cx-xm-10,yv+14,cx+xm+10,yv+54),fill=(40,52,80))
    for xo in [-340,-250,-150,-60,60,150,250,340]:
        col=OR if abs(xo)>200 else WH; y=prof(xo); h=1e-3
        sl=(prof(xo+h)-prof(xo-h))/(2*h); n=(-sl,1); l=math.hypot(*n); n=(n[0]/l,n[1]/l)
        r=(2*n[1]*n[0],-1+2*n[1]*n[1]) # reflect (0,-1)
        t=-xo/r[0]; fy=y+t*r[1]
        d.line([(cx+xo,170),(cx+xo,yv-y)],fill=col,width=4); d.line([(cx+xo,yv-y),(cx,yv-fy)],fill=col,width=4)
    return im,d,cx,yv
im,d,cx,yv=scene("Perfect mirror","every ray, edge or centre, meets at one focus",0)
d.ellipse((cx-16,yv-340-16,cx+16,yv-340+16),fill=GR); lab(d,(cx+420,300),"ONE focus",56,GR); lab(d,(cx+420,370),"= sharp star",46,GR)
im.save(OUT+"perfect_mirror_all_rays_one_focus_diagram.png")
im,d,cx,yv=scene("Hubble's mirror","edge rays focus farther away: spherical aberration",0.09)
lab(d,(cx+430,260),"Edge rays focus higher",42,OR); lab(d,(cx+430,330),"Centre rays focus lower",42,WH); lab(d,(cx-430,260),"= blurry star",56,RD)
im.save(OUT+"spherical_aberration_edge_rays_miss_focus_diagram.png")
im,d=new("Mirror shape error","edge ground 2.2 micrometres too flat (not to scale)")
cx=W//2; yv=720; f=420; xm=600
d.line([(cx+x,yv-x*x/(4*f)) for x in range(-xm,xm+1,4)],fill=GR,width=10); d.line([(cx+x,yv-(x*x/(4*f)-90*(abs(x/xm)**3))) for x in range(-xm,xm+1,4)],fill=RD,width=10)
d.line([(280,300),(320,300)],fill=GR,width=10); d.line([(280,360),(320,360)],fill=RD,width=10)
txt(d,(335,300),"Intended shape",48,GR,"lm"); txt(d,(335,360),"Actual shape",48,RD,"lm")
ya=yv-xm*xm/(4*f); lab(d,(cx+xm-230,ya-110),"2.2 µm at the edge",50,OR); txt(d,(cx,840),"A human hair is ~50x thicker",46,WH)
im.save(OUT+"mirror_shape_error_2p2_micrometres_diagram.png")
im,d=new("Null corrector test","a rig that reports what the mirror SHOULD look like")
box(d,80,330,380,470,"Laser",OR,50); box(d,540,330,840,470,"",CY); txt(d,(690,290),"Null corrector",46,CY); box(d,1100,260,1480,540,"Hubble mirror",GR,38)
arrow(d,(380,400),(540,400),OR); arrow(d,(840,400),(1100,400),OR); d.rectangle((700,345,716,455),fill=RD)
txt(d,(690,545),"field lens: 1.3 mm out of place",44,RD); txt(d,(W//2,680),"A worn spot on a measuring rod fooled the laser",44,WH)
txt(d,(W//2,765),"Result: the test says \"PERFECT\"",54,OR); txt(d,(W//2,835),"for the wrong shape",48,RD)
im.save(OUT+"null_corrector_field_lens_misplaced_1p3_mm_diagram.png")
im,d=new("Three tests, one believed","two other tests showed the flaw")
for i,(n,v,c,t) in enumerate([("Null corrector","PERFECT",GR,"trusted"),("Test 2","FLAW",RD,"dismissed"),("Test 3","FLAW",RD,"dismissed")]):
    y=230+i*190; box(d,120,y,1480,y+150,"",c); txt(d,(170,y+75),n,54,WH,"lm"); txt(d,(760,y+75),v,60,c,"lm"); txt(d,(1430,y+75),t,48,GY,"rm")
txt(d,(W//2,830),"Lesson: investigate disagreement",54,OR)
im.save(OUT+"three_tests_two_dissent_trusted_one_wrong_diagram.png")
im,d=new("Glasses for a telescope","corrective mirrors cancel the error")
box(d,60,330,430,570,"Main mirror",RD,46); txt(d,(245,620),"edge too flat",40,RD); txt(d,(530,450),"+",110,WH)
box(d,630,330,1000,570,"Corrector",OR,46); txt(d,(815,620),"opposite error",40,OR); txt(d,(1085,450),"=",110,WH)
box(d,1170,330,1540,570,"Sharp image",GR,44); txt(d,(1355,620),"error cancelled",40,GR); txt(d,(W//2,780),"WFPC2 and COSTAR, 1993",56,CY)
im.save(OUT+"corrective_optics_cancel_mirror_error_diagram.png")
im,d=new("Same star, before and after","blurry halo vs sharp core")
for cx,lab_,col,core,halo,k in [(430,"Before: flawed mirror",RD,45,260,0.35),(1170,"After: corrected",GR,38,60,0.25)]:
    cy=480
    for r in range(halo,0,-4):
        a=int(255*(1-r/halo)**2.2*k); d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(min(255,14+a),min(255,22+a),min(255,40+a)))
    for r in range(core,0,-3):
        v=int(255*(1-r/core)**1.2); d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(max(14,v),max(22,v),max(40,v)))
    txt(d,(cx,820),lab_,48,col)
im.save(OUT+"star_blur_halo_before_after_diagram.png")
# montage
fs=sorted(__import__("glob").glob(OUT+"*.png")); S=Image.new("RGB",(1600,1350),(0,0,0))
for i,f in enumerate(fs[:6]): im=Image.open(f).resize((800,450)); S.paste(im,((i%2)*800,(i//2)*450))
S.save("scratch/hubble/diag_check.jpg",quality=80)
