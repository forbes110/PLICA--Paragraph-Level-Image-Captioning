from pycocoevalcap.meteor.meteor import Meteor 

def meteor_score(ref, sample):
    # ref and sample are both dict
    scorer = Meteor()
    final_score = {}
    score, _ = scorer.compute_score(ref, sample)
    final_score["METEOR"] = score*100
    return final_score

if __name__ == "__main__":
    reference = {
        136:["this is a dog with good tail"],
        # 136:["this is a dog with bad tail, which is not only bad but failed."], 
        100:["I love you"],
        # 12:["what the fuck"]
    } 

    prediction = {
        136:["this is a dog with good tail"],
        100:["I love you"],
        # 12:["what the fuck"]
    }
    # a = calc_scores(reference, prediction)
    a = meteor_score(reference, prediction)
    # print(type(a))
    # print(len(a))
    print('result', a)
