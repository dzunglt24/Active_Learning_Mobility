import math

def calc_information_density(bert_tags, crf_tags, density):
    ''' Calculate information density of a sentence.
    This implementation follows the vote entropy formulation from 'https://www.biostat.wisc.edu/~craven/papers/settles.emnlp08.pdf' 
    
    Args:
        label_list: list of tags
        bert_tags: output tags from BERT model
        crf_tags: output tag from Stanford model
        density: density of a sentence in corpus
    
    Returns:
        Information density of a sentence
    '''
    
    assert len(bert_tags) == len(crf_tags)
    T = len(bert_tags)
    tag_list = list(set(bert_tags + crf_tags))
    M = len(tag_list)
    C = 2
    
    vote_entropy = 0
    if M == 1 or T == 0:
        return 0
    else:
        diff = 0
        for i in range(T):
            bert_tag = bert_tags[i]
            crf_tag =  crf_tags[i]
            if bert_tag != crf_tag: diff += 1
            for j in range(M):
                V = 0
                if bert_tag == tag_list[j]:
                    V += 1
                if crf_tag == tag_list[j]:
                    V += 1

                if V != 0:
                    vote_entropy -= (V/C) * math.log(V/C)
        
        if diff == 0:
            vote_entropy = 0
        else:
            vote_entropy = vote_entropy / math.sqrt(T)
            # vote_entropy = vote_entropy / diff
        return vote_entropy * density

def calc_agree_density(bert_tags, crf_tags, density):
    count = 0
    if bert_tags == crf_tags:
        for tag in bert_tags:
            if tag in ["B-Action", "B-Mobility"]:
                count +=1

    return count * density
