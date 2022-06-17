# Extract chunks of local context for doc-level relation prediction
# Based on the original paper https://www.aclweb.org/anthology/2020.argmining-1.8.pdf
# the argumentativeness nature for each sentence is assumed known during
# relation prediction. The system need to predict an undirected pariwise relation
# for all clause pairs that are no more than 5 sentences apart.
# We aim to transform the original json format into local context blocks for
# prediction.
# ------- PROCEDURE ---------
# 1. remove the part outside `THE LAW`
# 2. split the document into sentences (check whether clauses are broken by this)
# 3. store data as list of sentences with the following field
#    - sentence_id
#    - is_argument
# 4. store relation pairs, e.g. "[[4, 5], [6, 5]]"

import json
import numpy as np
from nltk.tokenize import sent_tokenize

class Document:

    def __init__(self, name, text, clauses, arguments):
        self.name = name
        self.text = text
        self.clauses = clauses
        self.arguments = arguments

    def remove_boilerplate(self):
        """Remove anything outside `THE LAW`. Adjust clause positions."""

        if 'AS TO THE LAW' in self.text:
            start_str = 'AS TO THE LAW'
        elif 'THE LAW' in self.text:
            start_str = 'THE LAW'
        else:
            raise ValueError

        real_start_index = self.text.index(start_str) + len(start_str)
        self.text = self.text[real_start_index:]
        print(f'size reduction from {len(self.text) + real_start_index} to {len(self.text)}')
        kept_clauses = []
        self.clause_ids = set()
        for cl in self.clauses:
            if cl['start'] < real_start_index:
                continue

            # shift from removing boilerplate
            cl_start_shifted = cl['start'] - real_start_index
            cl_end_shifted = cl['end'] - real_start_index

            cl_text = self.text[cl_start_shifted:cl_end_shifted]
            cl_text_trimmed = cl_text.strip()
            # shift by preceding or trailing space for clause
            if len(cl_text_trimmed) < len(cl_text):
                cl_shift = cl_text.index(cl_text_trimmed)
                cl_start_shifted += cl_shift
                cl_end_shifted = cl_start_shifted + len(cl_text_trimmed)

            kept_clauses.append({'_id': cl['_id'],
                                 'start': cl_start_shifted,
                                 'end': cl_end_shifted})
            self.clause_ids.add(cl['_id'])

        print(f'{len(kept_clauses)} clauses kept from {len(self.clauses)}')
        self.clauses = kept_clauses

    def sentence_split_and_clause_matching(self):
        """Split document into sentences. First take out clauses, then split
        the parts outside clauses."""

        clause_positions = []
        for i, cl in enumerate(self.clauses):
            clause_positions.append((cl['start'], cl['end'], cl['_id']))
        clause_positions_sorted = sorted(clause_positions, key=lambda x: x[0])
        self.ssplit = []
        sent_len_dist = []
        cur_ptr = 0
        for (cl_start, cl_end, cl_id) in clause_positions_sorted:
            cur_outside = self.text[cur_ptr: cl_start]
            cur_split = sent_tokenize(cur_outside)
            for s in cur_split:
                self.ssplit.append((s, False, None))
                sent_len_dist.append(len(s.split()))

            clause_text = self.text[cl_start: cl_end]
            self.ssplit.append((clause_text, True, cl_id))
            sent_len_dist.append(len(clause_text.split()))
            cur_ptr = cl_end
        print(f'{len(self.ssplit)} sentences found, sentence length: {min(sent_len_dist)} - {max(sent_len_dist)} (mean: {np.mean(sent_len_dist)})')


    def write_to_disk(self, fout):
        cur_doc = {
            'sentences': [item[0] for item in self.ssplit],
            'clause_id': [item[2] for item in self.ssplit],
            'is_argument': [],
            'relations': [],
        }

        arg_cl_ids = set()
        for arg in self.arguments:
            premises = arg['premises']
            conclusion = arg['conclusion']
            if conclusion not in self.clause_ids:
                continue

            for p in premises:
                if p not in self.clause_ids:
                    continue
                cur_doc['relations'].append((p, conclusion))
                arg_cl_ids.add(conclusion)
                arg_cl_ids.add(p)

        for cl_id in cur_doc['clause_id']:
            if cl_id in arg_cl_ids:
                cur_doc['is_argument'].append(True)
            else:
                cur_doc['is_argument'].append(False)


        fout.write(json.dumps(cur_doc) + "\n")
        return len(self.clauses), len(arg_cl_ids), len(self.ssplit), len(cur_doc['relations'])

if __name__=='__main__':
    with open('raw/echr_corpus/ECHR_Corpus.json') as jf:
        data = json.load(jf)

    num_args, num_cl, num_rel, num_sent = 0, 0, 0, 0
    fout = open('raw/echr_corpus/ECHR_sentences.jsonl', 'w')
    for item in data:
        doc = Document(name=item['name'], text=item['text'],
                       clauses=item['clauses'], arguments=item['arguments'])
        doc.remove_boilerplate()
        doc.sentence_split_and_clause_matching()
        cur_num_cl, cur_num_args, cur_num_sents, cur_rels = doc.write_to_disk(fout)
        num_args += cur_num_args
        num_cl += cur_num_cl
        num_rel += cur_rels
        num_sent += cur_num_sents
    fout.close()
    print(f'{num_sent} sentences in total, {num_cl} clauses, '
          f'{num_args} are arguments, {num_rel} pairs of relation found')
