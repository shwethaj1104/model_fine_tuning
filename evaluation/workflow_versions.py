"""Five portfolio workflow concepts using verified project facts and one palette."""
from pathlib import Path
import textwrap

NAVY, BLUE, PALE, BG, WHITE, MUTED, BORDER = '#102F62','#88A8D2','#DFEAFE','#F3F7FC','#FFFFFF','#536B87','#D9E5F3'
OUT = Path(__file__).resolve().parents[1]/'artifacts/portfolio/workflow_versions'
STEPS = [
 ('Define the need','Consistent labels, context, reported impact and team recommendations.','A repeatable support workflow'),
 ('Choose Qwen3-0.6B','Focused classification and fact extraction; small enough for free-GPU training.','Model fit + compute budget'),
 ('Prepare examples','228 AI-authored examples: 128 training, 34 validation and 66 held-out test tickets.','Separate learning from testing'),
 ('Measure the base','On validation: 17/34 structured answers and 4/34 correct team recommendations.','Record the starting gaps'),
 ('Fine-tune with LoRA','Train for 3 epochs on a free Colab T4 GPU. Update the adapter, not the base weights.','Task-specific training'),
 ('Select epoch 2 / 3','Validation loss: E1 0.2943 · E2 0.2563 · E3 0.2620. Select the lowest.','Epoch 2 selected before final test'),
 ('Verify on unseen tickets','On 66 test tickets: correct issue labels 25 → 58; structured answers 35 → 65.','Measured gain on authored test data'),
 ('Publish on Hugging Face','Share the adapter, loading instructions, evaluation results and limitations.','Reusable model + evidence'),
]


class Canvas:
 def __init__(self, eyebrow, title, subtitle, dark=False):
  import matplotlib.pyplot as plt
  from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
  self.plt=plt;self.Round=FancyBboxPatch;self.Arrow=FancyArrowPatch
  self.fig=plt.figure(figsize=(16,10),facecolor=BG)
  self.box(.035,.80,.93,.165,NAVY,NAVY,.022)
  self.text(.055,.937,eyebrow,10,PALE,True)
  self.text(.055,.889,title,26,WHITE,True)
  self.text(.055,.857,subtitle,11,PALE)
  labels=[('Qwen3-0.6B',.14),('Python',.10),('LoRA',.09),('Free Colab T4',.15),('Hugging Face',.15)]
  x=.055
  for label,w in labels:
   self.box(x,.811,w,.027,PALE,PALE,.006);self.text(x+w/2,.819,label,9,NAVY,True,ha='center');x+=w+.012
 def text(self,x,y,s,size=11,color=NAVY,bold=False,**kwargs):
  return self.fig.text(x,y,s,fontsize=size,fontfamily='DejaVu Sans',color=color,weight='bold' if bold else 'normal',zorder=5,**kwargs)
 def box(self,x,y,w,h,fill=WHITE,edge=BORDER,r=.012):
  self.fig.add_artist(self.Round((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',transform=self.fig.transFigure,facecolor=fill,edgecolor=edge,linewidth=1,zorder=2))
 def arrow(self,a,b):
  self.fig.add_artist(self.Arrow(a,b,transform=self.fig.transFigure,arrowstyle='-|>',mutation_scale=18,color=BLUE,linewidth=2,zorder=1))
 def lines(self,x,y,s,width=44,size=10,leading=.022,color=MUTED,bold=False):
  lines=textwrap.wrap(s,width=width,break_long_words=False)
  for j,line in enumerate(lines):self.text(x,y-j*leading,line,size,color,bold)
  return len(lines)
 def badge(self,x,y,s,w=.031):
  self.box(x,y,w,.033,NAVY,NAVY,.006);self.text(x+w/2,y+.011,s,10,WHITE,True,ha='center')
 def step(self,i,x,y,w,h,wrap=34,title_size=12,body_size=9):
  title,body,tag=STEPS[i]
  self.box(x,y,w,h);self.badge(x+.012,y+h-.050,f'{i+1:02}')
  self.lines(x+.053,y+h-.037,title,width=max(16,int(w*120)),size=title_size,leading=.024,color=NAVY,bold=True)
  self.lines(x+.014,y+h-.104,body,width=wrap,size=body_size,leading=.023)
  self.box(x+.012,y+.016,w-.024,.035,PALE,PALE,.006)
  self.text(x+w/2,y+.029,tag,7.5,NAVY,True,ha='center')
 def save(self,name):
  OUT.mkdir(parents=True,exist_ok=True)
  for ext in ('png',):self.fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor=BG)
  self.plt.close(self.fig)


def journey():
 c=Canvas('01 / THE JOURNEY','Eight steps. One measured model improvement.','Read the numbered path to follow the project from support need to published adapter.')
 xs=[.045,.287,.529,.771];w=.184
 positions=[(x,.45) for x in xs]+[(x,.065) for x in reversed(xs)]
 for x,xx in zip(xs,xs[1:]):
  c.arrow((x+w+.004,.60),(xx-.006,.60));c.arrow((xx-.006,.215),(x+w+.004,.215))
 c.arrow((.863,.438),(.863,.375))
 for i,(x,y) in enumerate(positions):c.step(i,x,y,w,.29,wrap=32,title_size=11,body_size=9)
 c.save('01_numbered_journey')


def phases():
 c=Canvas('02 / PHASE OVERVIEW','A clear path from task fit to tested release.','Four phases group the work by purpose, with the engineering actions and evidence kept together.')
 groups=[('FRAME', 'Choose a useful, affordable task.',[0,1]),('PREPARE','Create examples and a baseline.',[2,3]),('ADAPT','Train and choose the best version.',[4,5]),('PROVE & SHARE','Test fairly and publish the evidence.',[6,7])]
 for col,(title,purpose,indices) in enumerate(groups):
  x=.045+col*.24;c.box(x,.06,.19,.68)
  c.box(x,.665,.19,.075,NAVY,NAVY,.012);c.text(x+.015,.708,title,13,WHITE,True)
  c.lines(x+.014,.638,purpose,29,10,.022,NAVY,True)
  for row,i in enumerate(indices):
   y=.37-row*.245; c.badge(x+.014,y+.145,str(i+1))
   short_titles=['Define scope','Choose Qwen3-0.6B','Prepare examples','Measure baseline','Train with LoRA','Select epoch 2 / 3','Test unseen tickets','Publish adapter']
   c.lines(x+.055,y+.164,short_titles[i],20,10.5,.023,NAVY,True)
   c.lines(x+.014,y+.105,STEPS[i][1],31,9,.023)
  if col<3:c.arrow((x+.195,.405),(x+.234,.405))
 c.save('02_four_phase_overview')


def evidence_first():
 c=Canvas('03 / RESULTS FIRST','What changed—and the work behind it.','Start with the final held-out evidence, then trace the decisions that produced it.')
 for x,number,label,context in [(.045,'25 → 58','Correct issue labels','of 66 test tickets'),(.365,'35 → 65','Structured answers','of 66 test tickets'),(.685,'2 / 3','Selected training epoch','lowest validation loss')]:
  c.box(x,.586,.27,.154);c.text(x+.017,.706,label,12,NAVY,True);c.text(x+.017,.649,number,29,NAVY,True);c.text(x+.017,.606,context,10,MUTED)
 c.text(.045,.545,'HOW THE RESULT WAS BUILT',11,NAVY,True)
 for i in range(8):
  col=i%2;row=i//2;x=.045+col*.475;y=.405-row*.103
  c.box(x,y,.435,.089);c.badge(x+.012,y+.042,str(i+1))
  c.text(x+.055,y+.063,STEPS[i][0],11,NAVY,True)
  c.lines(x+.055,y+.036,STEPS[i][1],68,8.5,.018)
 c.save('03_results_first_story')


def gates():
 c=Canvas('04 / DECISION GATES','Four decisions that keep the experiment credible.','A decision-led view: what was checked, what was chosen, and what evidence moved the project forward.')
 decisions=[('Is this a useful fit?','DEFINE + SELECT MODEL','Support labels, context, impact and team. Qwen3-0.6B fits a focused task and free-GPU budget.','Decision: use a compact, task-specific model'),
 ('What needs to improve?','PREPARE DATA + BASELINE','228 authored examples with separate splits. Base validation: 17/34 formatted answers; 4/34 correct teams.','Decision: target structure and support-specific labels'),
 ('Which trained version wins?','TRAIN + SELECT CHECKPOINT','3 LoRA epochs on a free T4. Validation losses: 0.2943 → 0.2563 → 0.2620.','Decision: select epoch 2 before final testing'),
 ('Does the result hold up?','TEST + PUBLISH','66 held-out authored tickets. Correct labels: 25 → 58; structured answers: 35 → 65. Adapter published.','Evidence: saved predictions, scores and model card')]
 for i,(question,label,body,decision) in enumerate(decisions):
  y=.585-i*.17
  c.box(.045,y,.91,.145)
  c.badge(.061,y+.092,str(i+1),.035)
  c.text(.11,y+.112,question,16,NAVY,True)
  c.text(.061,y+.052,label,8.5,MUTED,True)
  c.lines(.395,y+.110,body,83,10,.023)
  c.box(.386,y+.015,.55,.033,PALE,PALE,.006)
  c.text(.398,y+.027,decision,9,NAVY,True)
  if i<3:c.arrow((.253,y-.006),(.253,y-.022))
 c.save('04_decision_gates')


def handoffs():
 c=Canvas('05 / ARTIFACT HANDOFFS','From raw request to reusable model.','Follow the outputs: each stage hands a concrete artifact to the next stage.')
 stages=[('INPUTS','Task & model choice','Issue labels, context, impact and team. Qwen3-0.6B chosen for a narrow task and free GPU.','OUTPUT: TASK CONTRACT',0,1),
 ('DATA','Examples & baseline','128 train / 34 validation / 66 test. Base validation: 17/34 structured answers; 4/34 correct teams.','OUTPUT: DATA + BASELINE',2,3),
 ('MODEL','Training & selection','LoRA, 3 epochs, free T4. Validation loss E1 .2943 / E2 .2563 / E3 .2620; epoch 2 selected.','OUTPUT: SELECTED ADAPTER',4,5),
 ('EVIDENCE','Final test & publishing','66 unseen authored tickets: correct labels 25 → 58; structured answers 35 → 65. Shared on Hugging Face.','OUTPUT: MODEL + RESULTS',6,7)]
 for i,(label,title,body,out,a,b) in enumerate(stages):
  y=.59-i*.165
  c.box(.045,y,.91,.141)
  c.box(.045,y,.145,.141,NAVY,NAVY,.012)
  c.text(.06,y+.095,label,12,WHITE,True)
  c.text(.06,y+.063,f'STEPS {a+1} + {b+1}',9,PALE)
  c.text(.212,y+.105,title,15,NAVY,True)
  c.lines(.212,y+.071,body,64,9.5,.022)
  c.box(.747,y+.028,.188,.077,PALE,PALE,.009)
  c.lines(.759,y+.073,out,24,9,.020,NAVY,True)
  if i<3:c.arrow((.118,y-.005),(.118,y-.020))
 c.save('05_artifact_handoffs')


def main():
 for concept in (journey,phases,evidence_first,gates,handoffs):concept()
 # A contact sheet makes reviewing all five practical.
 from PIL import Image,ImageOps,ImageDraw
 paths=sorted(OUT.glob('0*.png'))
 sheet=Image.new('RGB',(1920,850),'#F3F7FC');draw=ImageDraw.Draw(sheet)
 for i,path in enumerate(paths):
  thumb=Image.open(path).convert('RGB');thumb.thumbnail((620,388))
  x=15+(i%3)*640;y=30+(i//3)*420
  draw.text((x,y-18),path.stem.replace('_',' '),fill=NAVY)
  sheet.paste(thumb,(x,y))
 sheet.save(OUT/'all_five_versions.png')
 (OUT/'README.md').write_text('''# Workflow design options

All five concepts use the same verified project facts and navy/white palette. Each has a PNG export. The existing workflow is preserved.

1. Numbered journey — follow eight actions in chronological order.
2. Four-phase overview — scan the purpose of each phase, then its two actions.
3. Results-first story — begin with held-out outcomes, then inspect the supporting work.
4. Decision gates — understand what was checked and why a choice was made.
5. Artifact handoffs — follow the outputs from task contract through model and evidence.

The baseline counts are from 34 validation tickets; final outcome counts are from 66 held-out AI-authored test tickets. LoRA checkpoint selection uses validation loss, not final-test performance. No operational savings or production quality is claimed.
''')
 print(OUT)


if __name__=='__main__':main()
