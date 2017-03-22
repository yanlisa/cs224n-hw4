""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


import numpy as np
q_types = ['who',
        'whom',
        'whose',
        'what is',
        'what are',
        'what was',
        'what were',
        'what can',
        'what type',
        'what kind',
        'what do',
        'what did',
        'what does',
        'what has',
        'whats',
        'in what',
        'in which',
        'which',
        'what year',
        'when',
        'how long',
        'how much',
        'how many',
        'where',
        'why',
        'how did']
q_prefixes = {'what':['in'], 'which': ['in']}
q_suffixes = {'what': ['is', 'are', 'was', 'were', 'can', 'type', 'kind', 'do', 'did', 'does', 'has', 'year'],
            'how': ['long', 'much', 'many']
            }
# put 'which' at the end because it might get confused?
qwords = ['who', 'whom', 'whose', 'what', 'whats','when', 'how', 'where', 'why', 'how', 'which']

import string
def process_question(question):
    question = question.encode('ascii', 'ignore').lower()
    exclude = set(string.punctuation)
    question = ''.join(ch for ch in question if ch not in exclude)
    return question

def get_question_type(question):
    words = question.split(' ')
    q_inds = filter(lambda i: words[i] in qwords, range(len(words)))
    for i in q_inds:
        qword = words[i]
        if qword in q_prefixes and i != 0:
            if words[i-1] in q_prefixes[qword]:
                return ' '.join([words[i-1], qword])
        elif qword in q_suffixes and i != len(words)-1:
            if words[i+1] in q_suffixes[qword]:
                return ' '.join([qword, words[i+1]])
        return words[i]
    return ''
   
def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    q_success = {}
    ans_success = {}
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                question = process_question(qa['question'])
                q_type = get_question_type(question)
                if q_type not in q_success:
                    q_success[q_type] = []
                ans_len = int(np.average(map(lambda ans: 
                    len(ans.encode('ascii', 'ignore').split(' ')),
                    ground_truths))) # round down?
                if ans_len > 10:
                    ans_len = 10
                    #print("answer length", ans_len, "ans", ground_truths)
                if ans_len not in ans_success:
                    ans_success[ans_len] = []
                em_curr = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1_curr = metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

                q_success[q_type].append((f1_curr, em_curr))
                ans_success[ans_len].append((f1_curr, em_curr))
                exact_match += em_curr
                f1 += f1_curr

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    for q_type in q_success:
        f1_all, em_all = zip(*q_success[q_type])
        all_len = len(q_success[q_type])
        if q_type == '':
            q_type = 'UNK'
        print("{},{},{},{}".format(q_type, all_len,
            100.0 * np.sum(f1_all)/float(all_len),
            100.0 * np.sum(em_all)/float(all_len)))
    print()
    for ans_len in sorted(ans_success.keys()):
        f1_all, em_all = zip(*ans_success[ans_len])
        all_len = len(ans_success[ans_len])
        print("{},{},{},{}".format(ans_len, all_len,
            100.0 * np.sum(f1_all)/float(all_len),
            100.0 * np.sum(em_all)/float(all_len)))


    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))
