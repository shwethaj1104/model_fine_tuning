"""Export a portfolio process diagram for the completed fine-tuning project."""
from pathlib import Path


def render(output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    navy, blue, light = '#102F62', '#88A8D2', '#DFEAFE'
    bg, white, muted, border = '#F3F7FC', '#FFFFFF', '#536B87', '#D9E5F3'
    fig = plt.figure(figsize=(16, 10), facecolor=bg)

    def box(x,y,w,h,fill=white,edge=border,r=.015):
        fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
            transform=fig.transFigure,facecolor=fill,edgecolor=edge,linewidth=1,zorder=2))
    def text(x,y,value,size=12,color=navy,weight='normal',**kwargs):
        return fig.text(x,y,value,fontsize=size,color=color,weight=weight,zorder=4,**kwargs)
    def arrow(a,b):
        fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle='-|>',
            mutation_scale=19,color=blue,linewidth=2.2,zorder=1))

    box(.04,.785,.92,.18,navy,navy,.022)
    text(.065,.926,'AI ENGINEERING  /  PROJECT WORKFLOW',11,light,'bold')
    text(.065,.868,'Model Fine-Tuning for Support Triage',27,white,'bold')
    chips = [('Qwen3-0.6B', .152), ('Python', .112), ('LoRA Fine-Tuning', .172),
             ('Colab T4 GPU', .152), ('Hugging Face', .152)]
    chip_x = .065
    for label, width in chips:
        box(chip_x,.807,width,.037,light,light,.009)
        text(chip_x+width/2,.819,label,10,navy,'bold',ha='center')
        chip_x += width + .017

    # A clear eight-step path separates base-model choice from checkpoint choice.
    xs=[.045,.287,.529,.771]
    positions=[(x,.445) for x in xs]+[(x,.065) for x in reversed(xs)]
    w,h=.184,.285
    for left,right in zip(xs,xs[1:]):
        arrow((left+w+.005,.587),(right-.005,.587))
        arrow((right-.005,.208),(left+w+.005,.208))
    arrow((xs[-1]+w/2,.432),(xs[-1]+w/2,.363))
    steps=[
        ('01','Define the need','Consistent ticket decisions',
         ['Set clear issue and team labels.','Capture context and reported impact.'],
         'A repeatable support workflow'),
        ('02','Choose the model','Match the model to the task',
         ['Target ticket labels and fact extraction.','0.6B size fits free-GPU fine-tuning.'],
         'Qwen3-0.6B · Focused, low-cost pilot'),
        ('03','Prepare examples','Teach support-specific patterns',
         ['Create and review 228 examples.','Separate learning from final testing.'],
         '128 train · 34 validation · 66 test'),
        ('04','Measure the base','Find the starting gaps',
         ['17/34 answers met the format.','4/34 team suggestions were correct.'],
         'Starting validation benchmark'),
        ('05','Fine-tune model','Adapt it to the support task',
         ['Train a LoRA adapter for 3 epochs.','Use a free Google Colab T4 GPU.'],
         'Task-specific training · Free GPU'),
        ('06','Select best run','Lower validation loss is better',
         ['Epoch 1: 0.2943','Epoch 2: 0.2563  ← selected','Epoch 3: 0.2620'],
         'Lowest validation loss · Epoch 2'),
        ('07','Verify the gain','Test on unseen requests',
         ['Correct issue labels: 25 → 58 of 66.','Valid answers: 35 → 65 of 66.'],
         '66 AI-authored held-out tickets'),
        ('08','Publish model','Make the work reusable',
         ['Share the adapter on Hugging Face.','Include usage steps and test results.'],
         'Published model + measured evidence'),
    ]
    for (x,y),(number,title,subtitle,lines,tag) in zip(positions,steps):
        box(x,y,w,h)
        box(x+.011,y+h-.062,.029,.039,navy,navy,.008)
        text(x+.0255,y+h-.050,number,11,white,'bold',ha='center')
        text(x+.049,y+h-.050,title,12.5,navy,'bold')
        text(x+.011,y+.170,subtitle,10,navy,'bold')
        for j,line in enumerate(lines):
            text(x+.011,y+.132-j*.026,line,8.5,
                 navy if number=='06' and j==1 else muted,
                 'bold' if number=='06' and j==1 else 'normal')
        box(x+.009,y+.023,w-.018,.044,light,light,.007)
        text(x+w/2,y+.041,tag,7.5,navy,'bold',ha='center')

    output.mkdir(parents=True,exist_ok=True)
    for suffix in ('png',):
        path=output/f'fine_tuning_workflow.{suffix}'
        fig.savefig(path,dpi=200,facecolor=bg)
        print(path)
    plt.close(fig)


if __name__=='__main__':
    render(Path(__file__).resolve().parents[1]/'artifacts/portfolio')
