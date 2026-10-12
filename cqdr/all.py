#!/usr/bin/env python
# coding: utf-8
import copy

def predict(loader, qa, queries, last=False):
        
    # queries --> list(tuple)
    # queries_tail = [((ind2ent[query[0]], ind2rel[query[1]]), id) for id, query in enumerate(queries) if not ind2rel[query[1]].endswith("_reverse")]
    # queries_head = [((ind2ent[query[0]], ind2rel[query[1]][:-len("_reverse")]), id) for id, query in enumerate(queries) if ind2rel[query[1]].endswith("_reverse")]
    if "FB15k-237" in dataset or "FB15k" in dataset:
        queries_tail = [((ind2ent[query[0]], ind2rel[query[1]][len("+"):]), id) for id, query in enumerate(queries) if ind2rel[query[1]].startswith("+")]
        queries_head = [((ind2ent[query[0]], ind2rel[query[1]][len("-"):]), id) for id, query in enumerate(queries) if ind2rel[query[1]].startswith("-")]
    elif "NELL" in dataset:
        queries_tail = [((ind2ent[query[0]], ind2rel[query[1]]), id) for id, query in enumerate(queries) if not ind2rel[query[1]].endswith("_reverse")]
        queries_head = [((ind2ent[query[0]], ind2rel[query[1]][:-len("_reverse")]), id) for id, query in enumerate(queries) if ind2rel[query[1]].endswith("_reverse")]
        
        # queries_tail = [((ind2ent[query[0]], ind2rel[query[1]]), id) for id, query in enumerate(queries)]
        # queries_head = []
    else:
        raise
    
    
    qa.calculate_answers(queries=[x[0] for x in queries_tail], loader=loader, direction="tail")
    tail_answers = qa.get_answers(as_string=False)
    
    qa.calculate_answers(queries=[x[0] for x in queries_head], loader=loader, direction="head")
    head_answers = qa.get_answers(as_string=False)
    
    answers = [None] * len(queries)
    for ix, answer in enumerate(tail_answers):
        answers[queries_tail[ix][1]] = [(ent2ind[pyclauseid2ent[x[0]]], x[1]) for x in answer]
    for ix, answer in enumerate(head_answers):
        answers[queries_head[ix][1]] = [(ent2ind[pyclauseid2ent[x[0]]], x[1]) for x in answer]
        
    for query, id in queries_tail:
        gold = hrt[query[0]][query[1]]
        scores = dict()
        for g in gold:
            scores[ent2ind[g]] = 1.0
        for answer, score in answers[id]:
            if answer not in scores:
                scores[answer] = score
        answers[id] = list(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        
    for query, id in queries_head:
        gold = trh[query[0]][query[1]]
        scores = dict()
        for g in gold:
            scores[ent2ind[g]] = 1.0
        for answer, score in answers[id]:
            if answer not in scores:
                scores[answer] = score
        answers[id] = list(sorted(scores.items(), key=lambda x: x[1], reverse=True))   
        
    return answers

# new fast next queries
# test also on 2p
from tqdm import tqdm

def next_queries(predicted_answers, queries, last=False):

    second_queries = []
    for answer, query in zip(predicted_answers, queries):
        second_queries += [(x[0], query) for x in answer]
    second_queries = list(set(second_queries))
    second_queries_map = {v: k for k, v in enumerate(second_queries)}
    print(len(second_queries))
    
    
    results = []
    second_predicted_answers = []
    batch_size = 100000
    batch_size = 50000
    for ix in tqdm(range(0, len(second_queries), batch_size)):
        second_predicted_answers += maxplus(predict(loader, qa, second_queries[ix:ix+batch_size], last))
        
    for answer, query in tqdm(zip(predicted_answers, queries), total=len(queries)):
        scores_dict = defaultdict(float)
        for predicted_answer, score in answer:
            key = (predicted_answer, query)
            for second_predicted_answer, second_score in second_predicted_answers[second_queries_map[key]]:
                scores_dict[second_predicted_answer] = max(scores_dict[second_predicted_answer], score * second_score if fn == "prod" else min(score, second_score))
                # scores_dict[second_predicted_answer] = 1. - (1. - scores_dict[second_predicted_answer]) * (1. - score * second_score)
        results.append(list(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)))
    return results

import collections

def evaluation(scores, queries, test_ans, test_ans_hard):
    nentity = len(scores[0])    
    logs = []
    
    argsort = torch.argsort(scores, dim=1, descending=True)
    ranking = argsort.clone().to(torch.float)
    
    ranking = ranking.scatter_(1, 
                                argsort, 
                                torch.arange(nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                    1)
                                ) # achieve the ranking of all entities
    
    
    rankings = []
    for idx, (i, query) in enumerate(zip(argsort[:, 0], queries)):
        hard_answer = test_ans_hard[query]
        easy_answer = test_ans[query]
        num_hard = len(hard_answer)
        num_easy = len(easy_answer)
        assert len(hard_answer.intersection(easy_answer)) == 0
        cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
        cur_ranking, indices = torch.sort(cur_ranking)
        masks = indices >= num_easy
        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
        cur_ranking = cur_ranking - answer_list + 1 # filtered setting
        cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers
        
        rankings += cur_ranking.tolist()

        mrr = torch.mean(1./cur_ranking).item()
        h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
        h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
        h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

        logs.append({
            'MRR': mrr,
            'HITS1': h1,
            'HITS3': h3,
            'HITS10': h10,
            'num_hard_answer': num_hard,
        })
        
    print((1./torch.tensor(rankings)).mean())

    metrics = collections.defaultdict(int)
    
    for metric in logs[0].keys():
        if metric in ['num_hard_answer']:
            continue
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    metrics['num_queries'] = len(logs)

    return metrics

def maxplus(predicted_answers):
    if not agg_maxplus:
        return predicted_answers
    new_answers = []
    for ix in range(len(predicted_answers)):
        scores = defaultdict(list)
        for answer, score in predicted_answers[ix]:
            scores[score].append(answer)
            
        new_scores = []
        for score, values in scores.items():
            for ix, value in enumerate(values, 0):
                if score == 1.0:
                    new_scores.append((value, score))
                else:
                    assert (score - 0.00001 * 1/(len(values) - ix)) >= 0.0, f"{score}"
                    new_scores.append((value, score - 0.00001 * 1/(len(values) - ix)))
        new_answers.append(list(sorted(new_scores, key=lambda x: x[1], reverse=True))) # changed reverse
    return new_answers

def intersect(answers, fn="prod"):
    inter = None
    for answer_ in answers:
        if inter is None:
            inter = set([x[0] for x in answer_])
        else:
            inter = inter.intersection([x[0] for x in answer_])
            
    if fn == "prod":
        scores_dict = defaultdict(lambda: 1.0)
        for answers_ in answers:
            for entity, score in answers_:
                if entity in inter:
                    scores_dict[entity] *= score
                else:
                    scores_dict[entity] = 0.0001 * score
        return scores_dict
    elif fn == "avg":
        scores_dict = defaultdict(lambda: 0.0)
        for answers_ in answers:
            for entity, score in answers_:
                scores_dict[entity] += score / (2 if type_ == "2i" else 3)
        return scores_dict
    elif fn == "godel":
        scores_dict = defaultdict(lambda: 1.0)
        for answers_ in answers:
            for entity, score in answers_:
                if entity in inter:
                    scores_dict[entity] = min(scores_dict[entity], score)
        return scores_dict
    elif fn == "luk":
        scores_dict = defaultdict(lambda: 0.0)
        
        if len(answers) == 2:
            for answers_ in answers:
                for entity, score in answers_:
                    if entity in inter:
                        scores_dict[entity] += score
            for entity, score in scores_dict.items():
                scores_dict[entity] = max(0, score - 1)
        elif len(answers) == 3:
            left, middle, right = answers
            scores_dict_lm = defaultdict(lambda: 0.0)
            for entity, score in left:
                if entity in inter:
                    scores_dict_lm[entity] += score
            for entity, score in middle:
                if entity in inter:
                    scores_dict_lm[entity] += score
            for entity, score in scores_dict_lm.items():
                scores_dict[entity] += max(0, score - 1)
                
            ####
            for entity, score in right:
                if entity in inter:
                    scores_dict[entity] += score
            for entity, score in scores_dict.items():
                scores_dict[entity] = max(0, score - 1)
        return scores_dict
    else:
        raise
    
def intersect_left(left, right, fn="prod"):
    assert fn == "prod"
    scores_dict = defaultdict(lambda: 1.0)
    for entity, score in left:
        scores_dict[entity] = score
    for entity, score in right:
        if entity in scores_dict:
            scores_dict[entity] = min(scores_dict[entity], score)
            # scores_dict[entity] *= score
        else:
            scores_dict[entity] = (score * 0.0001)
        # else *= 1
    return scores_dict

def invert(left, fn="prod"):
    assert fn == "prod"
    scores_dict = dict()
    for entity, score in left:
        scores_dict[entity] = 1 - score
        assert (1-score) >= 0
        assert (1-score) <= 1, f"{score}"
    return scores_dict
    
    
def unionize(answers, fn="prod"):
    if fn == "prod":
        scores_dict = defaultdict(lambda: 1.0)
        for answers_ in answers:
            for entity, score in answers_:
                scores_dict[entity] *= (1-score)
        return scores_dict
    elif fn == "godel":
        scores_dict = defaultdict(lambda: 1.0)
        for answers_ in answers:
            for entity, score in answers_:
                scores_dict[entity] = min(scores_dict[entity], (1-score))
        return scores_dict
    elif fn == "avg":
        scores_dict = defaultdict(lambda: 0.0)
        for answers_ in answers:
            for entity, score in answers_:
                scores_dict[entity] += ((1 - score) / 2)
        return scores_dict
    elif fn == "luk":
        scores_dict = defaultdict(lambda: 0.0)   
        scores_dict_c = defaultdict(int)      
        assert len(answers) == 2 
        for answers_ in answers:
            for entity, score in answers_:
                scores_dict[entity] += (1 - score)
                scores_dict_c[entity] += 1
        
        for entity, count in scores_dict_c.items():
            scores_dict[entity] += 2 - count
                
        for entity, score in scores_dict.items():
            scores_dict[entity] = max(0, score - 1)
        return scores_dict
    elif fn == "mrrjoin":
        scores_dict = defaultdict(lambda: 0.0)
        for answers_ in answers:
            for ix, (entity, score) in enumerate(answers_):
                scores_dict[entity] = max(scores_dict[entity], 1 / (ix + 1))
        for e in scores_dict:
            scores_dict[e] = 1. - scores_dict[e]
        return scores_dict
    else:
        raise

from collections import defaultdict
import torch
from tqdm import tqdm
import sys

if __name__ == "__main__":
    
    """ run = "FB15k-237"
    type_ = "3p"
    k=15000 """
    
    run = sys.argv[1]
    type_ = sys.argv[2]
    k = int(sys.argv[3])
    agg_maxplus = bool(int(sys.argv[4]))
    fn = sys.argv[5] # prod, godel, avg
    split = sys.argv[6] # test, valid
    print(run, type_, k, agg_maxplus, fn)

    if run == "nell":
        dataset = "NELL-betae"
        rules_path = "../rules/nellrules-1000"
    elif run == "FB15k":
        dataset = "FB15k-betae"
        rules_path = "../rules/fb15krules-1000"
    elif run == "FB15k-237":
        dataset = "FB15k-237-betae"
        rules_path = "../rules/fb15k237rules-1000"
    train_path = f"../data/betae/{dataset}/train_symb.txt"

    import pickle

    with open(f"../data/betae/{dataset}/id2rel.pkl", "rb") as infile:
        ind2rel = pickle.load(infile)

    with open(f"../data/betae/{dataset}/id2ent.pkl", "rb") as infile:
        ind2ent = pickle.load(infile)

    with open(f"../data/betae/{dataset}/test-easy-answers.pkl", "rb") as infile:
        test_ans_all = pickle.load(infile)
        
    with open(f"../data/betae/{dataset}/test-hard-answers.pkl", "rb") as infile:
        test_ans_hard_all = pickle.load(infile)
        
    with open(f"../data/betae/{dataset}/test-queries.pkl", "rb") as infile:
        test_queries = pickle.load(infile)
        
    with open(f"../data/betae/{dataset}/valid-easy-answers.pkl", "rb") as infile:
        valid_ans_all = pickle.load(infile)
        
    with open(f"../data/betae/{dataset}/valid-hard-answers.pkl", "rb") as infile:
        valid_ans_hard_all = pickle.load(infile)
        
    with open(f"../data/betae/{dataset}/valid-queries.pkl", "rb") as infile:
        valid_queries = pickle.load(infile)

    if type_ == "1p":
        structure = ("e", ("r",))
    elif type_ == "2p":
        structure = ("e", ("r","r"))
    elif type_ == "3p":
        structure = ("e", ("r","r","r"))
    elif type_ == "2u":
        structure = (("e", ("r",)), ("e", ("r",)), ("u",))
    elif type_ == "2i":
        structure = (("e", ("r",)), ("e", ("r",)))
    elif type_ == "3i":
        structure = (("e", ("r",)), ("e", ("r",)), ("e", ("r",)))
    elif type_ == "ip":
        structure = ((('e', ('r',)), ('e', ('r',))), ('r',))
    elif type_ == "pi":
        structure = (('e', ('r', 'r')), ('e', ('r',)))
    elif type_ == "up":
        structure = ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))
    elif type_ == "2in":
        structure = (("e", ("r",)), ("e", ("r", "n")))
    elif type_ == "3in":
        structure = (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n")))
    elif type_ == "inp":
        structure = ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    elif type_ == "pin":
        structure = (('e', ('r', 'r')), ('e', ('r', 'n')))
    elif type_ == "pni":
        structure = (('e', ('r', 'r', 'n')), ('e', ('r',)))

    if split == "test":
        test_ans = {x: test_ans_all[x] for x in test_queries[structure]}
        test_ans_hard = {x: test_ans_hard_all[x] for x in test_queries[structure]}
        queries = list(test_ans_hard.keys())
    elif split == "valid":
        test_ans = {x: valid_ans_all[x] for x in valid_queries[structure]}
        test_ans_hard = {x: valid_ans_hard_all[x] for x in valid_queries[structure]}
        queries = list(test_ans_hard.keys())
    else:
        raise    
    
        
    ent2ind = {v: k for k,v in ind2ent.items()}
    print(len(queries))

    from collections import defaultdict

    with open(train_path, "r") as infile:
        train = infile.readlines()
        train = [x.strip().split("\t") for x in train]
    hrt = defaultdict(lambda: defaultdict(set))
    trh = defaultdict(lambda: defaultdict(set))

    for (h,r,t) in train:
        # h,r,t = ind2ent[h], ind2rel[r], ind2end[t]
        hrt[h][r].add(t)
        trh[t][r].add(h)

    from c_clause import QAHandler, Loader
    from clause import Options

    opts = Options()

    loader = Loader(options=opts.get("loader"))
    loader.load_data(train_path)
    loader.load_rules(rules=rules_path)

    opts.set("qa_handler.filter_w_data", False)
    opts.set("qa_handler.aggregation_function", "maxplus")
    opts.set("qa_handler.topk", k)
    opts.set("qa_handler.num_threads", 31)
    opts.set("qa_handler.num_top_rules", -1)
    qa = QAHandler(options=opts.get("qa_handler"))
    
    ent2pyclauseid = loader.get_entity_index()
    pyclauseid2ent = {v:k for k,v in ent2pyclauseid.items()}
        
    if type_ in ["1p", "2p", "3p"]:
        #1p
        pyclause_queries = [(x[0], x[1][0]) for x in queries]
        predicted_answers = maxplus(predict(loader, qa, pyclause_queries))

        predicted_answers_next = predicted_answers

        get_p_len = len(queries[0][1])
        # if get_p_len > 1:
        for p in range(1, get_p_len):
            queries_next = [x[1][p] for x in queries]
            predicted_answers_next = next_queries(predicted_answers_next, queries_next, True)
            
        scores = torch.zeros(len(queries), len(ind2ent), device="cpu")
        for i, (answer) in tqdm(enumerate(predicted_answers_next), total=len(predicted_answers_next)):
            for id, (predicted_answer, score) in enumerate(answer):
                scores[i][predicted_answer] = score
                
    elif type_ in ["2i", "3i"]:
        intersections = list(zip(*queries))

        predicted_answers = []
        for i in range(len(intersections)):
            pyclause_queries = [(x[0], x[1][0]) for x in intersections[i]]
            predicted_answers.append(maxplus(predict(loader, qa, pyclause_queries)))
        
        scores = torch.zeros(len(queries), len(ind2ent))
        for i, answers in enumerate(zip(*predicted_answers)):
            scores_dict = intersect(answers, fn=fn)         
            scores[i][list(scores_dict.keys())] = torch.tensor(list(scores_dict.values())).float()
            
    elif type_ in ["2in"]:
        intersections = list(zip(*queries))

        predicted_answers = []
        for i in range(len(intersections)):
            pyclause_queries = [(x[0], x[1][0]) for x in intersections[i]]
            predicted_answers.append(maxplus(predict(loader, qa, pyclause_queries)))
            
        scores = torch.zeros(len(queries), len(ind2ent))
        for i, answers in enumerate(zip(*predicted_answers)):
            scores_dict = intersect_left(answers[0], invert(answers[1]).items())
            scores[i][list(scores_dict.keys())] = torch.tensor(list(scores_dict.values())).float()
            
    elif type_ in ["3in"]:
        intersections = list(zip(*queries))

        predicted_answers = []
        for i in range(len(intersections)):
            pyclause_queries = [(x[0], x[1][0]) for x in intersections[i]]
            predicted_answers.append(maxplus(predict(loader, qa, pyclause_queries)))
            
        scores = torch.zeros(len(queries), len(ind2ent))
        for i, answers in enumerate(zip(*predicted_answers)):
            scores_dict = intersect_left(intersect([answers[0], answers[1]], fn).items(), invert(answers[2]).items())
            scores[i][list(scores_dict.keys())] = torch.tensor(list(scores_dict.values())).float()
            
    elif type_ == "inp":
        intersections = list(zip(*[x[0] for x in queries]))

        predicted_answers = []
        for i in range(len(intersections)):
            pyclause_queries = [(x[0], x[1][0]) for x in intersections[i]]
            predicted_answers.append(maxplus(predict(loader, qa, pyclause_queries)))
            
        next_answers = []
        for i, answers in enumerate(zip(*predicted_answers)):
            scores_dict = intersect_left(answers[0], invert(answers[1]).items())
            next_answers.append(list(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)))
            
        del predicted_answers

        queries_next = [x[1][0] for x in queries]
        next_answers = next_queries(next_answers, queries_next, True)
            
        scores = torch.zeros(len(queries), len(ind2ent))
        for i, (answer) in tqdm(enumerate(next_answers), total=len(next_answers)):
            for id, (predicted_answer, score) in enumerate(answer):
                scores[i][predicted_answer] = score
                
    elif type_ == "pin":
        p_queries_left = [x[0] for x in queries]

        pyclause_queries = [(x[0], x[1][0]) for x in p_queries_left]
        predicted_answers_p_left = maxplus(predict(loader, qa, pyclause_queries))

        queries_next = [x[1][1] for x in p_queries_left]
        predicted_answers_p_left = next_queries(predicted_answers_p_left, queries_next, True)

        p_queries_right = [x[1] for x in queries]
        pyclause_queries = [(x[0], x[1][0]) for x in p_queries_right]
        predicted_answers_p_right = maxplus(predict(loader, qa, pyclause_queries))

        next_answers = []
        for i, answers in enumerate(zip(predicted_answers_p_left, predicted_answers_p_right)):
            scores_dict = intersect_left(answers[0], invert(answers[1]).items())
            next_answers.append(list(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)))

        scores = torch.zeros(len(queries), len(ind2ent))
        for i, (answer) in tqdm(enumerate(next_answers), total=len(next_answers)):
            for id, (predicted_answer, score) in enumerate(answer):
                scores[i][predicted_answer] = score
                
    elif type_ == "pni":
        p_queries_left = [x[0] for x in queries]

        pyclause_queries = [(x[0], x[1][0]) for x in p_queries_left]
        predicted_answers_p_left = maxplus(predict(loader, qa, pyclause_queries))

        queries_next = [x[1][1] for x in p_queries_left]
        predicted_answers_p_left = next_queries(predicted_answers_p_left, queries_next, True)

        p_queries_right = [x[1] for x in queries]
        pyclause_queries = [(x[0], x[1][0]) for x in p_queries_right]
        predicted_answers_p_right = maxplus(predict(loader, qa, pyclause_queries))

        next_answers = []
        for i, answers in enumerate(zip(predicted_answers_p_left, predicted_answers_p_right)):
            scores_dict = intersect_left(answers[1], invert(answers[0]).items())
            next_answers.append(list(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)))

        scores = torch.zeros(len(queries), len(ind2ent))
        for i, (answer) in tqdm(enumerate(next_answers), total=len(next_answers)):
            for id, (predicted_answer, score) in enumerate(answer):
                scores[i][predicted_answer] = score
            
    elif type_ == "2u":
        intersections = list(zip(*[x[:-1] for x in queries]))

        predicted_answers = []
        for i in range(len(intersections)):
            pyclause_queries = [(x[0], x[1][0]) for x in intersections[i]]
            predicted_answers.append(maxplus(predict(loader, qa, pyclause_queries)))

        scores = torch.zeros(len(queries), len(ind2ent))
        for i, answers in enumerate(zip(*predicted_answers)):                 
            scores_dict = unionize(answers, fn)
            scores[i][list(scores_dict.keys())] = 1 - torch.tensor(list(scores_dict.values())).float()
            
    elif type_ == "ip":
        intersections = list(zip(*[x[0] for x in queries]))
        predicted_answers = []
        for i in range(len(intersections)):
            pyclause_queries = [(x[0], x[1][0]) for x in intersections[i]]
            predicted_answers.append(maxplus(predict(loader, qa, pyclause_queries)))
            
        next_answers = []
        for i, answers in enumerate(zip(*predicted_answers)):
            scores_dict = intersect(answers, fn)
            next_answers.append(list(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)))
            
        next_answers = next_queries(next_answers, [x[1][0] for x in queries])
        
        scores = torch.zeros(len(queries), len(ind2ent))
        for i, (answer) in tqdm(enumerate(next_answers), total=len(next_answers)):
            for id, (predicted_answer, score) in enumerate(answer):
                scores[i][predicted_answer] = score
                
    elif type_ == "up":
        print("Calculating union...")
        intersections = list(zip(*[x[0][:-1] for x in queries]))
        predicted_answers = []
        for i in range(len(intersections)):
            pyclause_queries = [(x[0], x[1][0]) for x in intersections[i]]
            predicted_answers.append(maxplus(predict(loader, qa, pyclause_queries)))
            
        print("Merging union...")
        next_answers = []
        for i, answers in enumerate(zip(*predicted_answers)):
            scores_dict = unionize(answers, fn)
            for k,v in scores_dict.items():
                scores_dict[k] = 1.0 - v
            next_answers.append(list(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)))
        del predicted_answers
        
        print("Calculating projection...")
        next_answers = next_queries(next_answers, [x[1][0] for x in queries])
        
        print("Creating scores tensor...")
        scores = torch.zeros(len(queries), len(ind2ent))
        for i, (answer) in tqdm(enumerate(next_answers), total=len(next_answers)):
            for id, (predicted_answer, score) in enumerate(answer):
                scores[i][predicted_answer] = score
                
    elif type_ == "pi":
        pyclause_queries = [(x[0][0], x[0][1][0]) for x in queries]
        predicted_answers_left = maxplus(predict(loader, qa, pyclause_queries))
        queries_next = [x[0][1][1] for x in queries]
        predicted_answers_left = next_queries(predicted_answers_left, queries_next)
        pyclause_queries = [(x[1][0], x[1][1][0]) for x in queries]
        predicted_answers_right = maxplus(predict(loader, qa, pyclause_queries))
        
        """ scores = torch.zeros(len(queries), len(ind2ent))
        for i, answers in enumerate(zip(predicted_answers_left, predicted_answers_right)):
            scores_dict = intersect(answers, fn)
            scores[i][list(scores_dict.keys())] = torch.tensor(list(scores_dict.values())).float() """
            
        next_answers = []
        for i, answers in enumerate(zip(predicted_answers_left, predicted_answers_right)):
            scores_dict = intersect(answers, fn)
            next_answers.append(list(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)))

        scores = torch.zeros(len(queries), len(ind2ent))
        for i, (answer) in tqdm(enumerate(next_answers), total=len(next_answers)):
            for id, (predicted_answer, score) in enumerate(answer):
                scores[i][predicted_answer] = score

    print("evaluate")
    print(evaluation(scores, queries, test_ans, test_ans_hard))




