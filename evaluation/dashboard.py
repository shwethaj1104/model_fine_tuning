"""Create a portfolio figure from a completed, verified final-test run."""
import argparse
import json
from pathlib import Path


def load_results(folder):
    manifest=json.loads((folder/'final_test_manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Complete both final-test evaluations first; no placeholder scores are rendered.')
    reference=manifest['frozen_reference']
    base=json.loads((folder/'base-test/metrics.json').read_text())
    adapted=json.loads((folder/'adapted-test/metrics.json').read_text())
    if base['examples'] != adapted['examples'] or base['examples'] != reference['test_examples']:
        raise ValueError('Both models must have been evaluated on the same number of test examples.')
    return reference,base,adapted


def render(folder):
    reference, base, adapted = load_results(folder)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Circle
    from matplotlib.colors import LinearSegmentedColormap

    bg, paper, ink, muted = '#F3F7FC', '#FFFFFF', '#102D50', '#536B87'
    primary, medium_blue, light_blue = '#102F62', '#88A8D2', '#D8E5F5'
    pale, line = '#EAF1FC', '#D9E5F3'
    n = base['examples']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':ink})
    fig = plt.figure(figsize=(16,13.5), facecolor=bg)

    def text(x,y,value,size=10,color=ink,weight='normal',**kwargs):
        return fig.text(x,y,value,fontsize=size,color=color,weight=weight,**kwargs)
    def box(x,y,w,h,fill=paper,border=line,r=.012,hatch=None):
        fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
            transform=fig.transFigure,facecolor=fill,edgecolor=border,linewidth=.8,hatch=hatch,zorder=-1))
    def chip(x,y,w,label,fill='#DFEAFE',color='#163E78'):
        box(x,y,w,.037,fill,fill,.009)
        text(x+w/2,y+.013,label,10,color,'bold',ha='center')
    def outcomes(result):
        m=result['category_confusion_matrix']
        correct=sum(row.get(category,0) for category,row in m.items())
        unusable=sum(row.get('invalid_output',0) for row in m.values())
        return correct,n-correct-unusable,unusable
    def save(figure,name):
        for suffix in ('png','pdf'):
            figure.savefig(folder/f'{name}.{suffix}',dpi=200,facecolor=figure.get_facecolor())

    # A distinct model identity strip; all labels are derived from saved artifacts.
    box(.035,.812,.93,.163,'#102F62','#102F62',.02)
    text(.055,.950,'AI ENGINEERING   /   BEFORE & AFTER',10,'#C5D9FC','bold')
    text(.055,.910,'Model Fine-Tuning for Support Triage',26,'#FFFFFF','bold')
    chip(.055,.838,.205,'MODEL  ·  Qwen3-0.6B')
    chip(.279,.838,.205,'Task-specific fine-tuning','#DFEAFE','#163E78')
    chip(.503,.838,.205,f'{n} unseen test tickets','#DFEAFE','#163E78')
    chip(.727,.838,.217,f"Selected: Epoch {reference['selected_epoch']} / 3",'#DFEAFE','#163E78')

    specs=[('schema_validity_rate','STRUCTURED ANSWERS','Responses match the format','a support tool expects.'),
           ('category_macro_f1','ISSUE UNDERSTANDING','Balances missed and incorrect issue','labels using precision and recall.'),
           ('environment_accuracy','CONTEXT DETECTION','Finds the affected environment;','keeps it unknown when not stated.'),
           ('team_accuracy','TEAM RECOMMENDATION','Suggests the expected support','team for each incoming request.')]
    for i,(key,title,d1,d2) in enumerate(specs):
        x=.035+i*.2375
        box(x,.612,.2175,.171)
        text(x+.014,.759,title,10,primary,'bold')
        text(x+.014,.738,d1,9,muted);text(x+.014,.722,d2,9,muted)
        value=adapted[key]
        delta=value-base[key]
        change=f'{delta:+.3f}' if key=='category_macro_f1' else f'{delta*100:+.1f}%'
        delta_color=primary
        text(x+.014,.675,change,28,delta_color,'bold')
        if key=='category_macro_f1':
            text(x+.137,.687,'F1 improvement',7.5,delta_color,'bold')
            text(x+.137,.674,'Scale 0–1',7.5,muted)
        box(x+.011,.621,.195,.025,'#EFF4FD','#EFF4FD',.006)
        start=f'{base[key]:.3f}' if key=='category_macro_f1' else f'{base[key]:.1%}'
        end=f'{value:.3f}' if key=='category_macro_f1' else f'{value:.1%}'
        text(x+.019,.630,f'Base {start}  →  Fine-tuned {end}',9,ink,'bold')

    box(.035,.265,.593,.317)
    text(.052,.556,'More requests classified correctly',16,ink,'bold')
    text(.052,.536,'Base model → Fine-tuned. Each filled dot is one correctly classified test ticket.',9,muted)
    friendly={'authentication':'Login & access','billing':'Bills & payments','rate_limit':'Usage limits',
              'integration':'Connecting tools','bug':'Product defects','other':'Other requests'}
    categories=list(base['category_per_class'])
    for i,c in enumerate(categories):
        col=i%2;row=i//2;x=.052+col*.285;y=.442-row*.079
        box(x,y,.269,.070,'#F5F8FD','#E6EEF9',.008)
        correct=adapted['category_confusion_matrix'][c][c]
        before=base['category_confusion_matrix'][c][c]
        support=sum(adapted['category_confusion_matrix'][c].values())
        text(x+.009,y+.051,friendly.get(c,c.title()),11,ink,'bold')
        text(x+.258,y+.051,f'{before}/{support} → {correct}/{support}',10,primary,'bold',ha='right')
        # One mark per actual ticket; dots reflect recall counts, not F1.
        shown=min(support,20)
        for dot in range(shown):
            xx=x+.013+dot*.010
            fig.add_artist(Circle((xx,y+.030),.0025,transform=fig.transFigure,
                           facecolor=primary if dot<round(correct/support*shown) else '#D4E1F3',edgecolor='none'))
        f0=base['category_per_class'][c]['f1'];f1=adapted['category_per_class'][c]['f1']
        text(x+.009,y+.010,f"{c.replace('_',' ').title()} · F1 {f0:.2f} → {f1:.2f}",8,muted)

    box(.651,.265,.314,.317)
    text(.669,.556,'Project approach',15,ink,'bold')
    steps=[('01','MODEL ADAPTATION','Adapted an existing AI model','to classify support requests.'),
           ('02','VERSION SELECTION','Compared saved versions before','the final test; selected epoch '+str(reference['selected_epoch'])+'.'),
           ('03','UNSEEN-TICKET EVALUATION',f'Tested both models on {n} new tickets','with the same instructions and settings.')]
    for i,(num,title,l1,l2) in enumerate(steps):
        yy=.496-i*.068
        box(.669,yy-.004,.030,.030,pale,pale,.008)
        text(.684,yy+.006,num,10,primary,'bold',ha='center')
        text(.710,yy+.015,title,9,primary,'bold')
        text(.710,yy-.002,l1,10,ink);text(.710,yy-.019,l2,9,muted)
    b0,a0=outcomes(base)[0],outcomes(adapted)[0]
    box(.669,.279,.278,.036,'#EAF1FC','#EAF1FC',.007)
    text(.683,.293,f'{a0-b0:+d} more tickets correctly classified',11,primary,'bold')

    # An understandable alternative to a confusion matrix on the main poster.
    box(.035,.085,.93,.153)
    text(.052,.212,'Fewer unusable answers',16,ink,'bold')
    text(.052,.193,f'{outcomes(base)[2]} → {outcomes(adapted)[2]} answers outside the required format. More consistent outputs for downstream support tools.',9,muted)
    for xx,label,color in ((.55,'Correct issue',primary),(.704,'Wrong issue',medium_blue),(.847,'Unusable answer',light_blue)):
        box(xx,.210,.012,.012,color,line,.002)
        text(xx+.017,.212,label,9,ink,'bold')
    for yy,result,label in ((.153,base,'Base model'),(.113,adapted,'Fine-tuned')):
        text(.052,yy,label,11,ink,'bold')
        cursor=.157
        for count,color in zip(outcomes(result),(primary,medium_blue,light_blue)):
            width=.64*count/n
            if count:
                box(cursor,yy-.008,width,.025,color,color,.003)
                if width>.027:text(cursor+width/2,yy+.000,f'{count}',10,'white' if color==primary else ink,'bold',ha='center')
                cursor+=width
        correct,wrong,invalid=outcomes(result)
        text(.817,yy,f'{correct} right · {wrong} wrong · {invalid} unusable',9,muted)

    text(.035,.061,'EVIDENCE & SCOPE',9,primary,'bold')
    text(.035,.044,'Improvements are against the base model. Rate changes shown with % are percentage-point differences, not relative growth. F1 combines precision and recall across 6 issue types.',8.5,muted)
    text(.035,.029,f'{n} AI-authored test tickets; not live customer traffic. No handling-time or cost savings measured. Team follows issue type; summaries need human review.',8.5,muted)
    text(.035,.013,'Base model version  '+reference['resolved_revision'],7.5,muted)
    text(.965,.013,'Saved test results · no illustrative values',8,muted,ha='right')
    save(fig,'final_test_dashboard');plt.close(fig)

    # Keep the rigorous engineering view as a readable second artifact.
    detail,axes=plt.subplots(1,2,figsize=(16,7),facecolor=bg)
    detail.subplots_adjust(left=.11,right=.97,bottom=.23,top=.73,wspace=.38)
    detail.text(.055,.92,'Where did each category go?',fontsize=25,color=ink,weight='bold')
    detail.text(.055,.865,'Choose an actual issue on the left, then read across to see the model’s predicted label.',fontsize=12,color=muted)
    detail.text(.055,.82,'Diagonal cells = correct labels. Off-diagonal cells = mix-ups. Invalid = no usable schema-valid response.',fontsize=11,color=muted)
    columns=categories+['invalid_output']
    vmax=max(sum(r['category_confusion_matrix'][c].values()) for r in (base,adapted) for c in categories)
    cmap=LinearSegmentedColormap.from_list('triage',['#EFF5FE','#88A8D2','#102F62'])
    for ax,result,title in zip(axes,(base,adapted),('Base model','Fine-tuned model')):
        matrix=np.array([[result['category_confusion_matrix'][c][p] for p in columns] for c in categories])
        ax.imshow(matrix,cmap=cmap,vmin=0,vmax=max(1,vmax),aspect='auto')
        for (r,c),value in np.ndenumerate(matrix):
            ax.text(c,r,str(value),ha='center',va='center',color='white' if value>vmax*.6 else ink,fontsize=11)
        short={'authentication':'Login/access','billing':'Billing','rate_limit':'Usage limits','integration':'Integration','bug':'Defects','other':'Other','invalid_output':'Invalid'}
        ax.set_xticks(range(len(columns)),[short[c] for c in columns],rotation=30,ha='right',fontsize=9)
        ax.set_yticks(range(len(categories)),[short[c] for c in categories],fontsize=10)
        ax.set_xlabel('Model predicted',labelpad=10);ax.set_ylabel('Actual issue',labelpad=10)
        ax.set_title(title,fontsize=15,weight='bold',pad=15)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.tick_params(length=0)
    detail.text(.055,.06,f'{n} AI-authored held-out tickets · both plots use the same count/color scale · no model retraining',fontsize=10,color=muted)
    save(detail,'final_test_confusion_details');plt.close(detail)
    return folder/'final_test_dashboard.png',folder/'final_test_dashboard.pdf',folder/'final_test_confusion_details.png',folder/'final_test_confusion_details.pdf'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    args=parser.parse_args()
    for path in render(args.results_dir):print(path)


if __name__=='__main__':main()
