from pycocoevalcap.meteor.meteor import Meteor 

def meteor_score(ref, sample):
    # ref and sample are both dict
    scorer = (Meteor(),"METEOR")
    final_score = {}
    score, _ = scorer.compute_score(ref, sample)
    final_score["METEOR"] = score*100
    return final_score
