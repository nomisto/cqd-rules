
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import collections
import pickle
import sys
from collections import defaultdict
import copy


# kvsall
def get_train_dataset_2(left, right, queries):
    left = torch.load(left)
    right = torch.load(right)

    target = torch.zeros((len(queries), len(ent2ind)))
    entities_left = torch.empty((len(queries)))
    relations_left = torch.empty((len(queries)))
    entities_right = torch.empty((len(queries)))
    relations_right = torch.empty((len(queries)))
    for ix, q in enumerate(queries):
        target[ix, list(train_ans[q])] = 1
        entities_left[ix] = q[0][0]
        entities_right[ix] = q[1][0]
        
        relations_left[ix] = q[0][1][0]
        relations_right[ix] = q[1][1][0]
        
    
    entities_left = entities_left.long()
    entities_right = entities_right.long()
    relations_left = relations_left.long()
    relations_right = relations_right.long()
    return TensorDataset(left, right, target, entities_left, relations_left, entities_right, relations_right)
    
    
def get_train_dataset_3(left, middle, right, queries):
    left = torch.load(left)
    middle = torch.load(middle)
    right = torch.load(right)

    target = torch.zeros((len(queries), len(ent2ind)))
    relations_left = torch.empty((len(queries)))
    relations_middle = torch.empty((len(queries)))
    relations_right = torch.empty((len(queries)))
    entities_left = torch.empty((len(queries)))
    entities_middle = torch.empty((len(queries)))
    entities_right = torch.empty((len(queries)))
    for ix, q in enumerate(queries):
        target[ix, list(train_ans[q])] = 1
        entities_left[ix] = q[0][0]
        entities_middle[ix] = q[1][0]
        entities_right[ix] = q[2][0]
        
        relations_left[ix] = q[0][1][0]
        relations_middle[ix] = q[1][1][0]
        relations_right[ix] = q[2][1][0]
            
    relations_left = relations_left.long()
    relations_middle = relations_middle.long()
    relations_right = relations_right.long()
    
    entities_left = entities_left.long()
    entities_middle = entities_middle.long()
    entities_right = entities_right.long()

    return TensorDataset(left, middle, right, target, entities_left, relations_left, entities_middle, relations_middle, entities_right, relations_right)


# 1vsAll
def get_train_dataset_2_1vsAll(left, right, queries):
    left = torch.load(left)
    right = torch.load(right)
    
    left_data = []
    right_data = []
    target_data = []
    entities_left_data = []
    entities_right_data = []
    relations_left_data = []
    relations_right_data = []
    
    for ix, q in enumerate(queries):
        positive_indices = list(train_ans[q])
        for positive_index in positive_indices:
            left_data.append(left[ix])
            right_data.append(right[ix])
            target_data.append(positive_index)
            entities_left_data.append(q[0][0])
            entities_right_data.append(q[1][0])
            relations_left_data.append(q[0][1][0])
            relations_right_data.append(q[1][1][0])
            
    left_tensor = torch.stack(left_data)
    right_tensor = torch.stack(right_data)
    target_tensor = torch.tensor(target_data, dtype=torch.long)
    relations_left_tensor = torch.tensor(relations_left_data, dtype=torch.long)
    relations_right_tensor = torch.tensor(relations_right_data, dtype=torch.long)
    entities_left_tensor = torch.tensor(entities_left_data, dtype=torch.long)
    entities_right_tensor = torch.tensor(entities_right_data, dtype=torch.long)
    
    return TensorDataset(left_tensor, right_tensor, target_tensor, entities_left_tensor, relations_left_tensor, entities_right_tensor, relations_right_tensor)
    
    
def get_train_dataset_3_1vsAll(left, middle, right, queries):
    left = torch.load(left)
    middle = torch.load(middle)
    right = torch.load(right)

    left_data = []
    middle_data = []
    right_data = []
    target_data = []
    entities_left_data = []
    entities_middle_data = []
    entities_right_data = []
    relations_left_data = []
    relations_middle_data = []
    relations_right_data = []
    
    for ix, q in enumerate(queries):
        positive_indices = list(train_ans[q])
        for positive_index in positive_indices:
            left_data.append(left[ix])
            middle_data.append(middle[ix])
            right_data.append(right[ix])
            target_data.append(positive_index)
            entities_left_data.append(q[0][0])
            entities_middle_data.append(q[1][0])
            entities_right_data.append(q[2][0])
            relations_left_data.append(q[0][1][0])
            relations_middle_data.append(q[1][1][0])
            relations_right_data.append(q[2][1][0])
            
    left_tensor = torch.stack(left_data)
    middle_tensor = torch.stack(middle_data)
    right_tensor = torch.stack(right_data)
    target_tensor = torch.tensor(target_data, dtype=torch.long)
    relations_left_tensor = torch.tensor(relations_left_data, dtype=torch.long)
    relations_middle_tensor = torch.tensor(relations_middle_data, dtype=torch.long)
    relations_right_tensor = torch.tensor(relations_right_data, dtype=torch.long)
    entities_left_tensor = torch.tensor(entities_left_data, dtype=torch.long)
    entities_middle_tensor = torch.tensor(entities_middle_data, dtype=torch.long)
    entities_right_tensor = torch.tensor(entities_right_data, dtype=torch.long)

    return TensorDataset(left_tensor, middle_tensor, right_tensor, target_tensor, entities_left_tensor, relations_left_tensor, entities_middle_tensor, relations_middle_tensor, entities_right_tensor, relations_right_tensor)


import torch

def get_test_dataset_2(left, right, test_queries):
    left_test = torch.load(left)
    right_test = torch.load(right)

    entities_left_test = torch.empty((len(test_queries)))
    entities_right_test = torch.empty((len(test_queries)))
    relations_left_test = torch.empty((len(test_queries)))
    relations_right_test = torch.empty((len(test_queries)))
    for ix, q in enumerate(test_queries):
        entities_left_test[ix] = q[0][0]
        entities_right_test[ix] = q[1][0]
        relations_left_test[ix] = q[0][1][0]
        relations_right_test[ix] = q[1][1][0]
        
        if q[0] in train_ans_1p:
            left_test[ix, list(train_ans_1p[q[0]])] = 1.0
        if (q[1][0], (q[1][1][0],)) in train_ans_1p:
            if len(q[1][1]) == 2 and q[1][1][1] == -2:
                right_test[ix, list(train_ans_1p[(q[1][0], (q[1][1][0],))])] = 0.0
            else:
                right_test[ix, list(train_ans_1p[(q[1][0], (q[1][1][0],))])] = 1.0
            
    entities_left_test = entities_left_test.long()
    entities_right_test = entities_right_test.long()
        
    relations_left_test = relations_left_test.long()
    relations_right_test = relations_right_test.long()

    return TensorDataset(left_test, right_test, entities_left_test, relations_left_test, entities_right_test, relations_right_test)

def get_test_dataset_3(left, middle, right, test_queries):
    left_test = torch.load(left)
    middle_test = torch.load(middle)
    right_test = torch.load(right)

    entities_left_test = torch.empty((len(test_queries)))
    entities_middle_test = torch.empty((len(test_queries)))
    entities_right_test = torch.empty((len(test_queries)))
    relations_left_test = torch.empty((len(test_queries)))
    relations_middle_test = torch.empty((len(test_queries)))
    relations_right_test = torch.empty((len(test_queries)))
    for ix, q in enumerate(test_queries):
        entities_left_test[ix] = q[0][0]
        entities_middle_test[ix] = q[1][0]
        entities_right_test[ix] = q[2][0]
        relations_left_test[ix] = q[0][1][0]
        relations_middle_test[ix] = q[1][1][0]
        relations_right_test[ix] = q[2][1][0]
        
        
        if q[0] in train_ans_1p:
            left_test[ix, list(train_ans_1p[q[0]])] = 1.0
        if q[1] in train_ans_1p:
            middle_test[ix, list(train_ans_1p[q[1]])] = 1.0
        if (q[2][0], (q[2][1][0],)) in train_ans_1p:
            if len(q[2][1]) == 2 and q[2][1][1] == -2:
                right_test[ix, list(train_ans_1p[(q[2][0], (q[2][1][0],))])] = 0.0
            else:
                right_test[ix, list(train_ans_1p[(q[2][0], (q[2][1][0],))])] = 1.0
    
    entities_left_test = entities_left_test.long()
    entities_middle_test = entities_middle_test.long()
    entities_right_test = entities_right_test.long()
    relations_left_test = relations_left_test.long()
    relations_middle_test = relations_middle_test.long()
    relations_right_test = relations_right_test.long()

    return TensorDataset(left_test, middle_test, right_test, entities_left_test, relations_left_test, entities_middle_test, relations_middle_test, entities_right_test, relations_right_test)


def to_logit(x):
    return torch.log(x / (1 - x))

def to_sigmoid(x):
    return torch.nn.functional.sigmoid(x)

to_sigmoid(to_logit(torch.tensor([0.5]))), to_logit(to_sigmoid(torch.tensor([10])))


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

class BCELoss:
    
    def __init__(self, reduction="sum", **kwargs):
        self._bceloss = torch.nn.BCELoss()

    def __call__(self, scores, labels, **kwargs):
        if labels.dim() == 1: # Labels are indexes of positive classes
            labels_ = torch.zeros_like(scores)
            labels_[torch.arange(labels.shape[0]), labels] = 1.
            return self._bceloss(scores, labels_)
        else:
            # label matrix
            return self._bceloss(scores, labels)

class KLDivWithSoftmaxKgeLoss:
    
    def __init__(self, reduction="sum", **kwargs):
        self._celoss = torch.nn.CrossEntropyLoss(reduction=reduction, **kwargs)
        self._klloss = torch.nn.KLDivLoss(reduction=reduction, **kwargs)

    def __call__(self, scores, labels, **kwargs):
        if labels.dim() == 1:
            # Labels are indexes of positive classes, i.e., we are in a multiclass
            # setting. Then kl divergence can be computed more efficiently using
            # pytorch's CrossEntropyLoss. (since the entropy of the data distribution is
            # then 0 so kl divergence equals cross entropy)
            #
            # Gives same result as:
            #   labels = self._labels_as_matrix(scores, labels)
            # followed by using _klloss as below.
            return self._celoss(scores, labels)
        else:
            # label matrix; use KlDivLoss
            return self._klloss(
                F.log_softmax(scores, dim=1), F.normalize(labels.float(), p=1, dim=1)
            )


# Define the neural network class
from tqdm import tqdm
class Adaptive(nn.Module):
    def __init__(self, num_rels, approach, deep):
        super(Adaptive, self).__init__()
        self.approach = approach
        self.deep = deep
        if approach == "er":
            self.alpha = nn.Linear(rel_embeddings.shape[1], 1 if not deep else 500)
            self.beta = nn.Linear(rel_embeddings.shape[1], 1 if not deep else 500)
        elif approach == "eser":
            self.alpha = nn.Linear(ent_embeddings.shape[1] + rel_embeddings.shape[1], 1 if not deep else 500)
            self.beta = nn.Linear(ent_embeddings.shape[1] + rel_embeddings.shape[1], 1 if not deep else 500)
        elif approach == "r":
            self.alpha = nn.Embedding(num_rels, 1 if not deep else 500)
            self.beta = nn.Embedding(num_rels, 1 if not deep else 500)
            
        if deep:
            self.deep_fn_alpha = nn.Linear(500,1)
            self.deep_fn_beta = nn.Linear(500,1)
            
    def get_alpha_beta(self, entity, relation):
        if self.approach == "er":
            alpha = self.alpha(rel_embeddings[relation])
            beta = self.beta(rel_embeddings[relation])
        elif self.approach == "eser":
            embs = torch.cat([ent_embeddings[entity], rel_embeddings[relation]], dim=1)
            alpha = self.alpha(embs)
            beta = self.beta(embs)
        elif self.approach == "r":
            alpha = self.alpha(relation)
            beta = self.beta(relation)
            
        if self.deep:
            alpha = self.deep_fn_alpha(torch.nn.functional.relu(alpha))
            beta = self.deep_fn_beta(torch.nn.functional.relu(beta))
            
        return alpha, beta
        

    def forward(self, data_left, data_right, entity_left, relation_left, entity_right, relation_right):
        
        alpha_left, beta_left = self.get_alpha_beta(entity_left, relation_left)
        alpha_right, beta_right = self.get_alpha_beta(entity_right, relation_right)
        
        mask = ~((data_left != 0.0) & (data_right != 0.0))
        data_left = torch.clamp(data_left, 1e-7, 1 - 1e-7)
        data_right = torch.clamp(data_right, 1e-7, 1 - 1e-7)
        adapt_left = to_logit(data_left) * (1 + alpha_left) + beta_left
        adapt_right = to_logit(data_right) * (1 + alpha_right) + beta_right
        x = to_sigmoid(adapt_left) * to_sigmoid(adapt_right)
        x[mask] = 0.0
        return x
    
    def forward_2in(self, data_left, data_right, entity_left, relation_left, entity_right, relation_right):
        
        alpha_left, beta_left = self.get_alpha_beta(entity_left, relation_left)
        alpha_right, beta_right = self.get_alpha_beta(entity_right, relation_right)
        
        mask = (data_right == 0.0)
        data_left = torch.clamp(data_left, 1e-7, 1 - 1e-7)
        data_right = torch.clamp(data_right, 1e-7, 1 - 1e-7)
        adapt_left = to_logit(data_left) * (1 + alpha_left) + beta_left
        adapt_right = to_logit(data_right) * (1 + alpha_right) + beta_right
        x = torch.minimum(to_sigmoid(adapt_left), to_sigmoid(adapt_right))
        x[mask] = 0.0
        return x
    
    def forward_2u(self, data_left, data_right, entity_left, relation_left, entity_right, relation_right):
        
        alpha_left, beta_left = self.get_alpha_beta(entity_left, relation_left)
        alpha_right, beta_right = self.get_alpha_beta(entity_right, relation_right)
        
        data_left = torch.clamp(data_left, 1e-7, 1 - 1e-7)
        data_right = torch.clamp(data_right, 1e-7, 1 - 1e-7)
        adapt_left = to_logit(data_left) * (1 + alpha_left) + beta_left
        adapt_right = to_logit(data_right) * (1 + alpha_right) + beta_right
        x = torch.maximum(to_sigmoid(adapt_left), to_sigmoid(adapt_right))
        return x
    
    def forward_3i(self, data_left, data_middle, data_right, entity_left, relation_left, entity_middle, relation_middle, entity_right, relation_right):
        
        alpha_left, beta_left = self.get_alpha_beta(entity_left, relation_left)
        alpha_middle, beta_middle = self.get_alpha_beta(entity_middle, relation_middle)
        alpha_right, beta_right = self.get_alpha_beta(entity_right, relation_right)
        
        mask = ~((data_left != 0.0) & (data_middle != 0.0) & (data_right != 0.0))
        data_left = torch.clamp(data_left, 1e-7, 1 - 1e-7)
        data_middle = torch.clamp(data_middle, 1e-7, 1 - 1e-7)
        data_right = torch.clamp(data_right, 1e-7, 1 - 1e-7)
        adapt_left = to_logit(data_left) * (1 + alpha_left) + beta_left
        adapt_middle = to_logit(data_middle) * (1 + alpha_middle) + beta_middle
        adapt_right = to_logit(data_right) * (1 + alpha_right) + beta_right
        x = to_sigmoid(adapt_left) * to_sigmoid(adapt_middle) * to_sigmoid(adapt_right)
        x[mask] = 0.0
        return x
    
    def forward_3in(self, data_left, data_middle, data_right, entity_left, relation_left, entity_middle, relation_middle, entity_right, relation_right):
        
        alpha_left, beta_left = self.get_alpha_beta(entity_left, relation_left)
        alpha_middle, beta_middle = self.get_alpha_beta(entity_middle, relation_middle)
        alpha_right, beta_right = self.get_alpha_beta(entity_right, relation_right)
        
        mask = (data_right == 0.0)
        mask1 = ~((data_left != 0.0) & (data_middle != 0.0))
        data_left = torch.clamp(data_left, 1e-7, 1 - 1e-7)
        data_middle = torch.clamp(data_middle, 1e-7, 1 - 1e-7)
        data_right = torch.clamp(data_right, 1e-7, 1 - 1e-7)
        adapt_left = to_logit(data_left) * (1 + alpha_left) + beta_left
        adapt_middle = to_logit(data_middle) * (1 + alpha_middle) + beta_middle
        adapt_right = to_logit(data_right) * (1 + alpha_right) + beta_right
        # x = to_sigmoid(adapt_left) * to_sigmoid(adapt_middle) * to_sigmoid(adapt_right)
        # x = torch.minimum(torch.minimum(to_sigmoid(adapt_left),to_sigmoid(adapt_middle)),to_sigmoid(adapt_right))
        x = torch.minimum(to_sigmoid(adapt_left)*to_sigmoid(adapt_middle),to_sigmoid(adapt_right))
        x[mask] = 0.0
        x[mask1] = 0.0
        return x
    
    def test(self):
        obj_ = {}
        with torch.no_grad():
            """ scores = []
            for batch_idx, (data_left_2, data_right_2, target_2, entity_left_2, relation_left_2, entity_right_2, relation_right_2) in tqdm(enumerate(test_train_loader_2i)):
                scores.append(model(data_left_2.cuda(), data_right_2.cuda(), entity_left_2.cuda(), relation_left_2.cuda(), entity_right_2.cuda(), relation_right_2.cuda()).cpu())
            scores = torch.vstack(scores)
            mrr_2i = evaluation(scores, queries_2i, defaultdict(set), train_ans_2i)
            print("train 2i", mrr_2i["MRR"])
            obj_["train_2i"] = mrr_2i["MRR"] """
            
            scores = []
            for batch_idx, (data_left, data_right, entity_left, relation_left, entity_right, relation_right) in enumerate(test_loader_2i):
                scores.append(self.forward(data_left.cuda(), data_right.cuda(), entity_left.cuda(), relation_left.cuda(), entity_right.cuda(), relation_right.cuda()).cpu())
            scores = torch.vstack(scores)
            mrr_2i = evaluation(scores, test_queries_2i, test_ans_2i, test_ans_hard_2i)
            print("2i", mrr_2i["MRR"])
            obj_["2i"] = mrr_2i["MRR"]
            
            scores = []
            for batch_idx, (data_left, data_right, entity_left, relation_left, entity_right, relation_right) in enumerate(test_loader_2u):
                scores.append(self.forward_2u(data_left.cuda(), data_right.cuda(), entity_left.cuda(), relation_left.cuda(), entity_right.cuda(), relation_right.cuda()).cpu())
            scores = torch.vstack(scores)
            mrr_2u = evaluation(scores, test_queries_2u, test_ans_2u, test_ans_hard_2u)
            print("2u", mrr_2u["MRR"])
            obj_["2u"] = mrr_2u["MRR"]
            
            scores = []
            for batch_idx, (data_left, data_right, entity_left, relation_left, entity_right, relation_right) in enumerate(test_loader_2in):
                scores.append(self.forward_2in(data_left.cuda(), data_right.cuda(), entity_left.cuda(), relation_left.cuda(), entity_right.cuda(), relation_right.cuda()).cpu())
            scores = torch.vstack(scores)
            mrr_2in = evaluation(scores, test_queries_2in, test_ans_2in, test_ans_hard_2in)
            print("2in", mrr_2in["MRR"])
            obj_["2in"] = mrr_2in["MRR"]
            
            scores = []
            for batch_idx, (data_left, data_middle, data_right, entity_left, relation_left, entity_middle, relation_middle, entity_right, relation_right) in enumerate(test_loader_3i):
                scores.append(self.forward_3i(data_left.cuda(), data_middle.cuda(), data_right.cuda(), entity_left.cuda(), relation_left.cuda(), entity_middle.cuda(), relation_middle.cuda(), entity_right.cuda(), relation_right.cuda()).cpu())
            scores = torch.vstack(scores)
            mrr_3i = evaluation(scores, test_queries_3i, test_ans_3i, test_ans_hard_3i)
            print("3i", mrr_3i["MRR"])
            obj_["3i"] = mrr_3i["MRR"]
            
            scores = []
            for batch_idx, (data_left, data_middle, data_right, entity_left, relation_left, entity_middle, relation_middle, entity_right, relation_right) in enumerate(test_loader_3in):
                scores.append(self.forward_3in(data_left.cuda(), data_middle.cuda(), data_right.cuda(), entity_left.cuda(), relation_left.cuda(), entity_middle.cuda(), relation_middle.cuda(), entity_right.cuda(), relation_right.cuda()).cpu())
            scores = torch.vstack(scores)
            mrr_3in = evaluation(scores, test_queries_3in, test_ans_3in, test_ans_hard_3in)
            print("3in", mrr_3in["MRR"])
            obj_["3in"] = mrr_3in["MRR"]
        return obj_
    
def objective(trial):
    # Hyperparameters
    # learning_rate = 0.001
    # epochs = 1000
    # batch_size = 64
    epochs = 100
    patience = 10
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2)
    opt = trial.suggest_categorical("opt", ["adam", "adagrad"])
    loss = trial.suggest_categorical("loss", ["bce", "ce"])
    approach = trial.suggest_categorical("approach", ["r", "er", "eser"])
    strat = trial.suggest_categorical("strat", ["kvsall", "1vsall"])
    deep = trial.suggest_categorical("deep", [True, False])
    batch_size = trial.suggest_categorical("batch_size", [64,128,256,512])
    
    print(learning_rate, weight_decay, opt, loss, approach, strat, batch_size, deep)
    
    if strat == "kvsall":
        dataset_2i = get_train_dataset_2("scores_left.pt", "scores_right.pt", queries_2i)
        dataset_2in = get_train_dataset_2("scores_left_2in.pt", "scores_right_2in.pt", queries_2in)
        dataset_2 = ConcatDataset([dataset_2i, dataset_2in])
        dataset_3i =  get_train_dataset_3("scores_left_3i.pt", "scores_middle_3i.pt", "scores_right_3i.pt", queries_3i)
        dataset_3in = get_train_dataset_3("scores_left_3in.pt", "scores_middle_3in.pt", "scores_right_3in.pt", queries_3in)
        dataset_3 = ConcatDataset([dataset_3i, dataset_3in])
    elif strat == "1vsall":
        dataset_2i = get_train_dataset_2_1vsAll("scores_left.pt", "scores_right.pt", queries_2i)
        dataset_2in = get_train_dataset_2_1vsAll("scores_left_2in.pt", "scores_right_2in.pt", queries_2in)
        dataset_2 = ConcatDataset([dataset_2i, dataset_2in])
        dataset_3i =  get_train_dataset_3_1vsAll("scores_left_3i.pt", "scores_middle_3i.pt", "scores_right_3i.pt", queries_3i)
        dataset_3in = get_train_dataset_3_1vsAll("scores_left_3in.pt", "scores_middle_3in.pt", "scores_right_3in.pt", queries_3in)
        dataset_3 = ConcatDataset([dataset_3i, dataset_3in])
    
    train_loader_2 = DataLoader(dataset_2, batch_size=batch_size, shuffle=True)
    train_loader_3 = DataLoader(dataset_3, batch_size=batch_size, shuffle=True)
    # test_train_loader_2i = DataLoader(dataset_2i, batch_size=test_batch_size, shuffle=False)
        
    # Model, loss function, optimizer
    model = Adaptive(len(ind2rel), approach, deep).cuda()
    
    if loss == "bce":
        criterion = BCELoss()
    else:
        criterion = KLDivWithSoftmaxKgeLoss()

    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise
    
    max_ = 0.0
    evals = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, ((data_left_2, data_right_2, target_2, entity_left_2, relation_left_2, entity_right_2, relation_right_2), (data_left_3, data_middle_3, data_right_3, target_3, entity_left_3, relation_left_3, entity_middle_3, relation_middle_3, entity_right_3, relation_right_3)) in tqdm(enumerate(zip(train_loader_2, train_loader_3)), total=len(train_loader_2)):
        
            # Forward pass
            outputs = model(data_left_2.cuda(), data_right_2.cuda(), entity_left_2.cuda(), relation_left_2.cuda(), entity_right_2.cuda(), relation_right_2.cuda())
            loss1 = criterion(outputs, target_2.cuda())
            loss1.backward(retain_graph=True)
            running_loss += loss1.item()
            
            outputs = model.forward_3i(data_left_3.cuda(), data_middle_3.cuda(), data_right_3.cuda(), entity_left_3.cuda(), relation_left_3.cuda(), entity_middle_3.cuda(), relation_middle_3.cuda(), entity_right_3.cuda(), relation_right_3.cuda())
            loss2 = criterion(outputs, target_3.cuda())
            loss2.backward()
            running_loss += loss2.item()
            
            # Backward pass and optimization
            optimizer.step()
            optimizer.zero_grad()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / (len(train_loader_2) * 2):.4f}")
        eval = model.test()
        evals.append(eval)
        avg_ = sum(eval.values()) / len(eval)
        if avg_ > max_:
            max_ = avg_
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                return max_
            
    global glob_hp
    glob_hp.append((learning_rate, weight_decay, opt, loss, approach, strat, batch_size, deep))
    global glob_trace
    glob_trace.append(evals)
    global glob_models
    glob_models.append(copy.deepcopy(model))
    return max_



if __name__ == "__main__":
    
    path_model = "../models/fb15k-237-betae/checkpoint"
    datasets_folder = "../data/betae"
    dataset = "FB15k-237-betae"
    train_path = f"{datasets_folder}/{dataset}/train_symb.txt"
    rules_path = "../rules/fb15k237rules-1000"

    checkpoint = torch.load(path_model)
    ent_embeddings = checkpoint["model_state_dict"]['embeddings.0.weight']
    rel_embeddings = checkpoint["model_state_dict"]['embeddings.1.weight']
    ent_embeddings.requires_grad_ = False
    rel_embeddings.requires_grad_ = False

    with open(f"{datasets_folder}/{dataset}/id2rel.pkl", "rb") as infile:
        ind2rel = pickle.load(infile)

    with open(f"{datasets_folder}/{dataset}/id2ent.pkl", "rb") as infile:
        ind2ent = pickle.load(infile)
        
    with open(f"{datasets_folder}/{dataset}/train-answers.pkl", "rb") as infile:
        train_ans = pickle.load(infile)
        
    with open(f"{datasets_folder}/{dataset}/train-queries.pkl", "rb") as infile:
        train_queries = pickle.load(infile)

    with open(f"{datasets_folder}/{dataset}/test-easy-answers.pkl", "rb") as infile:
        test_ans_all = pickle.load(infile)
        
    with open(f"{datasets_folder}/{dataset}/test-hard-answers.pkl", "rb") as infile:
        test_ans_hard_all = pickle.load(infile)
        
    with open(f"{datasets_folder}/{dataset}/test-queries.pkl", "rb") as infile:
        test_queries = pickle.load(infile)
        
    with open(f"{datasets_folder}/{dataset}/valid-easy-answers.pkl", "rb") as infile:
        valid_ans_all = pickle.load(infile)
        
    with open(f"{datasets_folder}/{dataset}/valid-hard-answers.pkl", "rb") as infile:
        valid_ans_hard_all = pickle.load(infile)

    structure_1p = (("e", ("r",)))
    train_ans_1p = {x: train_ans[x] for x in train_queries[structure_1p]}
        
    structure_2i = (("e", ("r",)), ("e", ("r",)))
    train_ans_2i = {x: train_ans[x] for x in train_queries[structure_2i]}
    queries_2i = list(train_ans_2i.keys())
    print(len(queries_2i))

    structure_2in = (("e", ("r",)), ("e", ("r", "n")))
    train_ans_2in = {x: train_ans[x] for x in train_queries[structure_2in]}
    queries_2in = list(train_ans_2in.keys())
    print(len(queries_2in))

    structure_3i = (("e", ("r",)), ("e", ("r",)), ("e", ("r",)))
    train_ans_3i = {x: train_ans[x] for x in train_queries[structure_3i]}
    queries_3i = list(train_ans_3i.keys())
    print(len(queries_3i))

    structure_3in = (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n")))
    train_ans_3in = {x: train_ans[x] for x in train_queries[structure_3in]}
    queries_3in = list(train_ans_3in.keys())
    print(len(queries_3in))

    structure_2i = (("e", ("r",)), ("e", ("r",)))
    test_ans_2i = {x: test_ans_all[x] for x in test_queries[structure_2i]}
    test_ans_hard_2i = {x: test_ans_hard_all[x] for x in test_queries[structure_2i]}
    test_queries_2i = list(test_ans_hard_2i.keys())

    structure_3i = (("e", ("r",)), ("e", ("r",)), ("e", ("r",)))
    test_ans_3i = {x: test_ans_all[x] for x in test_queries[structure_3i]}
    test_ans_hard_3i = {x: test_ans_hard_all[x] for x in test_queries[structure_3i]}
    test_queries_3i = list(test_ans_hard_3i.keys())

    structure_2u = (("e", ("r",)), ("e", ("r",)), ("u",))
    test_ans_2u = {x: test_ans_all[x] for x in test_queries[structure_2u]}
    test_ans_hard_2u = {x: test_ans_hard_all[x] for x in test_queries[structure_2u]}
    test_queries_2u = list(test_ans_hard_2u.keys())

    structure_2in = (("e", ("r",)), ("e", ("r", "n")))
    test_ans_2in = {x: test_ans_all[x] for x in test_queries[structure_2in]}
    test_ans_hard_2in = {x: test_ans_hard_all[x] for x in test_queries[structure_2in]}
    test_queries_2in = list(test_ans_hard_2in.keys())

    structure_3in = (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n")))
    test_ans_3in = {x: test_ans_all[x] for x in test_queries[structure_3in]}
    test_ans_hard_3in = {x: test_ans_hard_all[x] for x in test_queries[structure_3in]}
    test_queries_3in = list(test_ans_hard_3in.keys())

    ent2ind = {v: k for k,v in ind2ent.items()}

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
    
    test_dataset_2i = get_test_dataset_2("scores_left_test.pt", "scores_right_test.pt", test_queries_2i)
    test_dataset_2u = get_test_dataset_2("scores_left_test_2u.pt", "scores_right_test_2u.pt", test_queries_2u)
    test_dataset_2in = get_test_dataset_2("scores_left_test_2in.pt", "scores_right_test_2in.pt", test_queries_2in)
    test_dataset_3in = get_test_dataset_3("scores_left_test_3in.pt", "scores_middle_test_3in.pt", "scores_right_test_3in.pt", test_queries_3in)
    test_dataset_3i = get_test_dataset_3("scores_left_test_3i.pt", "scores_middle_test_3i.pt", "scores_right_test_3i.pt", test_queries_3i)
    
    test_batch_size = 64
    test_loader_2i = DataLoader(test_dataset_2i, batch_size=test_batch_size, shuffle=False)
    test_loader_2u = DataLoader(test_dataset_2u, batch_size=test_batch_size, shuffle=False)
    test_loader_2in = DataLoader(test_dataset_2in, batch_size=test_batch_size, shuffle=False)
    test_loader_3i = DataLoader(test_dataset_3i, batch_size=test_batch_size, shuffle=False)
    test_loader_3in = DataLoader(test_dataset_3in, batch_size=test_batch_size, shuffle=False)
    
    import optuna
    import pickle

    glob_hp = []
    glob_trace = []
    glob_models = []
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    with open("glob_trace.pkl", "wb") as outfile:
        pickle.dump(glob_trace, outfile)
        
    with open("glob_hp.pkl", "wb") as outfile:
        pickle.dump(glob_hp, outfile)
        
    with open("study.pkl", "wb") as outfile:
        pickle.dump(study, outfile)

    # Print best hyperparameters
    print("Best hyperparameters:", study.best_params)
    

    


