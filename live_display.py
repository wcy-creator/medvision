#!/usr/bin/env python3
"""MedVision Live Display with HUD - pygame"""
import os
os.environ["SDL_VIDEODRIVER"] = "x11"
import pygame, cv2, numpy as np, time, json, subprocess, tempfile, threading
from datetime import datetime

pygame.init()
W, H = 800, 600
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("MedVision Live")
fs = pygame.font.SysFont("monospace", 18)
fm = pygame.font.SysFont("monospace", 22, bold=True)
fl = pygame.font.SysFont("monospace", 28, bold=True)
clk = pygame.time.Clock()
SNAP = "/opt/medvision/snapshots"
os.makedirs(SNAP, exist_ok=True)
BG=(20,20,35); GRN=(0,255,100); RED=(255,60,60); YEL=(255,255,0)
WHT=(255,255,255); BLU=(80,160,255); PNL=(30,30,50)

class LD:
  def __init__(s):
    s.fps=0; s.fc=0; s.t0=time.time(); s.pt=time.time()
    s.msg="Ready"; s.mc=GRN; s.az=False; s.u2=False; s.p2=None; s.ok=True; s.ai=""
    try:
      from picamera2 import Picamera2
      s.p2=Picamera2()
      s.p2.configure(s.p2.create_preview_configuration(main={"size":(640,480),"format":"RGB888"}))
      s.p2.start(); time.sleep(0.5); s.u2=True; print("Camera: picamera2")
    except Exception as e: print("picamera2 fail:", e)

  def cap(s):
    if s.u2 and s.p2:
      try: return s.p2.capture_array()
      except: return None
    t=tempfile.mktemp(suffix=".jpg")
    try:
      subprocess.run(["rpicam-still","-o",t,"--width","640","--height","480","--nopreview","-t","200","--immediate"],capture_output=True,timeout=5)
      return cv2.imread(t)
    except: return None
    finally:
      if os.path.exists(t): os.remove(t)

  def ana(s,fr):
    g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY); h,w=g.shape
    br=float(np.mean(g)); sd=float(np.std(g))
    ed=cv2.Canny(g,50,150); ep=np.count_nonzero(ed)/(h*w)*100
    ct,_=cv2.findContours(ed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bg=[c for c in ct if cv2.contourArea(c)>500]
    hsv=cv2.cvtColor(fr,cv2.COLOR_BGR2HSV)
    mp=np.count_nonzero(cv2.inRange(hsv,(0,0,180),(180,50,255)))/(h*w)*100
    return {"br":br,"sd":sd,"ep":ep,"cnt":len(bg),"mp":mp,"big":bg}

  def hud(s,scr,fr,a):
    fh,fw=fr.shape[:2]
    sc=min((W-220)/fw,(H-80)/fh); nw,nh=int(fw*sc),int(fh*sc)
    ox=210+(W-220-nw)//2; oy=40+(H-80-nh)//2
    if a and a["big"]:
      fb=cv2.cvtColor(fr,cv2.COLOR_RGB2BGR)
      for c in a["big"][:10]:
        x,y,cw,ch=cv2.boundingRect(c)
        cv2.rectangle(fb,(x,y),(x+cw,y+ch),(0,255,255),2)
      fr=cv2.cvtColor(fb,cv2.COLOR_BGR2RGB)
    sf=pygame.surfarray.make_surface(np.rot90(fr))
    sf=pygame.transform.scale(sf,(nw,nh)); scr.blit(sf,(ox,oy))
    pygame.draw.rect(scr,GRN,(ox-2,oy-2,nw+4,nh+4),2)
    pygame.draw.rect(scr,PNL,(0,0,W,36))
    scr.blit(fl.render("MedVision",True,GRN),(10,4))
    scr.blit(fm.render(datetime.now().strftime("%H:%M:%S"),True,WHT),(W-120,6))
    fc=GRN if s.fps>5 else YEL if s.fps>1 else RED
    scr.blit(fm.render("%.1f FPS"%s.fps,True,fc),(W-280,6))
    if a:
      pygame.draw.rect(scr,PNL,(5,42,200,250))
      pygame.draw.rect(scr,GRN,(5,42,200,250),1)
      its=[("Brightness","%.0f"%a["br"],WHT),("StdDev","%.0f"%a["sd"],WHT),
           ("Edge","%.1f%%"%a["ep"],BLU),("Contours","%d"%a["cnt"],YEL),
           ("Metal","%.1f%%"%a["mp"],BLU)]
      y=50
      for lb,vl,co in its:
        scr.blit(fs.render(lb,True,(150,150,150)),(12,y))
        scr.blit(fs.render(vl,True,co),(130,y)); y+=28
      by=y+10; bv=min(a["br"]/255,1.0)
      pygame.draw.rect(scr,(50,50,50),(12,by,180,10))
      bc=GRN if 40<a["br"]<200 else RED
      pygame.draw.rect(scr,bc,(12,by,int(180*bv),10))
      bl="OK" if 40<a["br"]<200 else ("DARK" if a["br"]<40 else "BRIGHT")
      scr.blit(fs.render(bl,True,GRN if bl=="OK" else RED),(12,by+18))
    if s.ai:
      pygame.draw.rect(scr,PNL,(5,310,200,200))
      pygame.draw.rect(scr,BLU,(5,310,200,200),1)
      scr.blit(fs.render("AI Analysis",True,BLU),(12,316))
      ls=[s.ai[i:i+24] for i in range(0,min(len(s.ai),120),24)]
      yy=340
      for ln in ls[:6]: scr.blit(fs.render(ln,True,WHT),(12,yy)); yy+=22
    pygame.draw.rect(scr,PNL,(0,H-32,W,32))
    scr.blit(fs.render(s.msg,True,s.mc),(10,H-26))
    scr.blit(fs.render("S:Snap A:AI Q:Quit",True,(120,120,120)),(W-280,H-26))

  def aia(s,fr):
    if s.az: return
    s.az=True; s.msg="AI analyzing..."; s.mc=YEL
    def run():
      t=None
      try:
        t=tempfile.mktemp(suffix=".jpg"); cv2.imwrite(t,fr)
        r=subprocess.run(["python3","/opt/medvision/vision_analyzer.py"],capture_output=True,text=True,timeout=30)
        if r.returncode==0:
          o=r.stdout; j=o.find("{")
          if j>=0:
            d=json.loads(o[j:]); cl=d.get("cloud_analysis")
            if cl and cl.get("success"): s.ai=cl["analysis"][:120]; s.msg="AI complete"
            else: s.ai=str(d.get("local_analysis",{}).get("summary","")); s.msg="AI local only"
            s.mc=GRN
          else: s.msg="AI done"; s.mc=GRN
        else: s.msg="AI fail: "+r.stderr[:40]; s.mc=RED
      except Exception as e: s.msg="Error: "+str(e)[:40]; s.mc=RED
      finally:
        if t and os.path.exists(t): os.remove(t)
        s.az=False
    threading.Thread(target=run,daemon=True).start()

  def run(s):
    print("MedVision Live (pygame)")
    print("S=snap A=AI Q/ESC=quit")
    while s.ok:
      for ev in pygame.event.get():
        if ev.type==pygame.QUIT: s.ok=False
        elif ev.type==pygame.KEYDOWN:
          if ev.key in(pygame.K_q,pygame.K_ESCAPE): s.ok=False
          elif ev.key==pygame.K_s:
            fr=s.cap()
            if fr is not None:
              p=os.path.join(SNAP,"live_%s.jpg"%datetime.now().strftime("%Y%m%d_%H%M%S"))
              cv2.imwrite(p,fr); s.msg="Saved: "+p; s.mc=GRN
          elif ev.key==pygame.K_a:
            fr=s.cap()
            if fr is not None: s.aia(fr.copy())
      screen.fill(BG); fr=s.cap()
      if fr is not None:
        s.fc+=1; n=time.time(); dt=n-s.pt
        if dt>0: s.fps=1.0/dt
        s.pt=n; a=s.ana(fr)
        s.hud(screen,cv2.cvtColor(fr,cv2.COLOR_BGR2RGB),a)
      else: screen.blit(fl.render("Camera Error",True,RED),(W//2-100,H//2))
      pygame.display.flip(); clk.tick(30)
    pygame.quit()
    if s.p2: s.p2.stop(); s.p2.close()
    print("Done: %d frames, %.0fs"%(s.fc,time.time()-s.t0))

if __name__=="__main__": LD().run()
