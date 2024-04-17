from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

def calculate_metrics(outputs, captions_list):
    bleu = lambda ref, out, w: sentence_bleu([ref.split()], out.split(), weights=w, smoothing_function=SmoothingFunction().method1)
    avg_bleu = lambda weights: sum([bleu(ref, out, weights) for ref, out in zip(captions_list, outputs)]) / len(outputs)
    weights = [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)]
    bleu_scores = [avg_bleu(w) for w in weights]
    print("Avg. BLEU-1 to BLEU-4 scores: ", bleu_scores)

    rougeL_scores = [rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(ref, out)['rougeL'].fmeasure for ref, out in zip(captions_list, outputs)]
    print("Avg. ROUGE-L: ", sum(rougeL_scores)/len(rougeL_scores))

    print("accuracy: ", sum(x==y for x, y in zip(outputs, captions_list)) / len(outputs))
    # assuming captions_list contains at least one of each example
    print("number of non-matching captions: ", sum(x not in captions_list for x in outputs))

def calculate_metrics_new(outputs, captions_list):
    metrics = {
        "bleu": Bleu(),
        "rouge": Rouge(),
        "meteor": Meteor(),
        "cider": Cider(),
        "spice": Spice()
    }
    
    scores = {}
    gts = {i: [cap] for i, cap in enumerate(captions_list)}
    res = {i: [out] for i, out in enumerate(outputs)}
    
    for name, metric in metrics.items():
        score = metric.compute_score(gts, res)[0]
        if name == 'bleu':
            scores.update({f'BLEU-{i}': score[i-1] for i in range(1, 5)})
        else:
            scores[name.upper()] = score
    
    print(scores)
    
    print("accuracy: ", sum(x==y for x, y in zip(outputs, captions_list)))
    # assuming captions_list contains at least one of each example
    print("number of non-matching captions: ", sum(x not in captions_list for x in outputs))