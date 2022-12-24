from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as ms
import nltk
nltk.download('omw-1.4')
from metrics import cal_score


## bleu 1, 2, 3, 4, cider and m

def bleu_score(reference, prediction, n_gram, smoothing_function=None):
    if n_gram == 1:
        return sentence_bleu(reference, prediction, weights=(1, 0, 0, 0))*100
    if n_gram == 2:
        return sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0))*100
    if n_gram == 3:
        return sentence_bleu(reference, prediction, weights=(0.33333, 0.33333, 0.33333, 0))*100
    if n_gram == 4:
        return sentence_bleu(reference, prediction, weights=(0.25, 0.25, 0.25, 0.25))*100

def meteor_score(reference,prediction):
    return ms(reference,prediction)*100

# def cider:


if __name__ == '__main__':
    ## example
    # reference = [
    # 'this is a dog with bad tail, which is not only bad but failed.'.split(),
    # ]
    # prediction = 'this is a dog with good tail'.split()

    # for i in range(1,5):
    #     print(f'bleu{i}', bleu_score(reference, prediction, i))
    # meteor = meteor_score(reference,prediction)
    # print('meteor', meteor)

    reference = {
        136:["this is a dog with bad tail, which is not only bad but failed."],
    } 

    prediction = {
        136:["this is a dog with good tail"],
    }
    # a = calc_scores(reference, prediction)
    a = cal_score(reference, prediction)
    # print(type(a))
    # print(len(a))
    print('result', a)