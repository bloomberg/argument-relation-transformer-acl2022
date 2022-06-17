import json
import os
import glob
from itertools import permutations
import numpy as np


class Statistics:

    def __init__(self, dataset):
        self.dataset = dataset
        self.stats_keys = [
            'num_doc', 'num_prop', 'num_segments', 'num_unique_target', 
            'num_unique_support_target', 'num_unique_attack_target',
            'num_link', 'num_supports', 'num_attacks', 'avg_prop_len'
        ]
        self.stats = {
            _split: {
                k: 0 if k.startswith('num') else [] for k in self.stats_keys
            }
            for _split in ['train', 'val', 'test']
        }

    def update(self, doc, _split):
        self.stats[_split]['num_doc'] += 1
        self.stats[_split]['num_prop'] += doc.num_args
        self.stats[_split]['num_segments'] += len(doc.segments)
        self.stats[_split]['num_unique_target'] += len(doc.target_prop_ids["union"])
        self.stats[_split]['num_link'] += (len(doc.relations))
        self.stats[_split]['num_supports'] += len(doc.supports)
        self.stats[_split]['num_attacks'] += len(doc.attacks)
        self.stats[_split]['num_unique_support_target'] += len(doc.target_prop_ids["support"])
        self.stats[_split]['num_unique_attack_target'] += len(doc.target_prop_ids["attack"])
        self.stats[_split]['avg_prop_len'].extend(doc.segment_lengths)

    def print(self):
        print(f'---------- {self.dataset} -----------')
        # print title
        print(f'{"split":<6s} |', end='')
        for k in self.stats_keys:
            print(f'{k:<10s} |', end='')
        print()
        for _split in ['train', 'val', 'test']:
            print(f'{_split:<6s} |', end='')
            for k in self.stats_keys:
                cur_s = self.stats[_split][k]
                if k.startswith('num'):
                    print(f'{cur_s:<10} |', end='')
                else:
                    print(f'{np.mean(cur_s):<10.1f}', end='')
            print()

        print(f'{"total":<6s} |', end='')
        for k in self.stats_keys:
            if k.startswith('num'):
                cur_s = sum([item[k] for item in self.stats.values()])
                print(f'{cur_s:<10} |', end='')
            else:
                cur_s = []
                for _split in self.stats:
                    cur_s.extend(self.stats[_split][k])
                cur_s = np.mean(cur_s)
                print(f'{cur_s:<10.1f} |', end='')
            
        print()


class ArgumentDocument:

    dataset_path_prefix = None
    dataset_type = None
    dataset_glob_expression = None
    dataset_train_ids = []
    dataset_val_ids = []
    dataset_test_ids = []

    def __init__(self, file_id):
        self.file_id = file_id
        self.num_args = 0
        # raw segmentation, might not be arguments
        self.segments = []
        # for UKP, non-args are not considered for relations
        self.types = []
        # target proposition ids, for UKP, this is NOT their segment ids
        self.target_prop_ids = {"support" : set(), "attack" : set(), "union" : set()}

        # indicate the target information for each segment
        self.target_labels = {"support" : [], "attack" : []}

        self.supports = []
        self.attacks = []
        self.relations = []

    def load_document(self):
        """Load raw data from disk, and store in the following fields:
        self.num_args
        self.segments = []
        self.types = []
        self.target_prop_ids = set() # use proposition ids, not segment ids
        self.supports = [] # use segment ids
        self.supports_in_prop_ids = [] # use proposition ids
        self.attacks = []
        ...
        """
        raise NotImplementedError

    def generate_relation_candidates(self):
        """Create possible link prediction candidate pairs."""
        for (src, tgt) in permutations(range(self.num_args), 2):
            if (src, tgt) in self.supports:
                label = "support"
            elif (src, tgt) in self.attacks:
                label = "attack"
            else:
                label = "none"
            src_text = self.segments[src]
            tgt_text = self.segments[tgt]
            yield ((src, src_text), (tgt, tgt_text), label)

    @classmethod
    def find_split(cls, file_id):
        raise NotImplementedError


    @classmethod
    def make_all_data(cls):
        docs = {'train': [], 'val': [], 'test': []}
        stats = Statistics(cls.dataset_type)
        for path in glob.glob(cls.dataset_path_prefix + cls.dataset_glob_expression):
            if 'ids' in path:
                continue
            
            file_id = os.path.basename(path).split('.')[0]
            cur_doc = cls(file_id)
            cur_doc.load_document()
            _split = cls.find_split(file_id)

            docs[_split].append(cur_doc)
            stats.update(cur_doc, _split)

        stats.print()
        cls.make_relation_doc_level_data(docs)


    @classmethod
    def make_relation_doc_level_data(cls, docs):
        fout_list = {
            s: open(f'trainable/{cls.dataset_type}_{s}.jsonl', 'w')
            for s in docs.keys()
        }
        for s in docs:
            for doc in docs[s]:
                output_obj = {
                    'doc_id': doc.file_id,
                    'text': doc.segments,
                    'relations': [{"head": item[1], "tail": item[0], "type": item[2]} for item in doc.relations],
                }
                fout_list[s].write(json.dumps(output_obj) + "\n")
            fout_list[s].close()


class CDCPDocument(ArgumentDocument):

    dataset_path_prefix = "raw/cdcp/"
    dataset_glob_expression = "*.txt"
    dataset_type = "cdcp"
    raw_train_ids = [ln.strip() for ln in open('cdcp_train_ids.txt')]

    dataset_train_ids = raw_train_ids[:-80]
    dataset_val_ids = raw_train_ids[-80:]
    dataset_test_ids = [ln.strip() for ln in open('cdcp_test_ids.txt')]

    def __init__(self, file_id):
        # file_id: "01418"
        super().__init__(file_id)
        self.text_path = CDCPDocument.dataset_path_prefix + f'{file_id}.txt'
        self.ann_path = CDCPDocument.dataset_path_prefix + f'{file_id}.ann.json'

    def load_document(self):
        raw_text = open(self.text_path).read()
        ann_obj = json.loads(open(self.ann_path).read())

        for (ch_start, ch_end) in ann_obj['prop_offsets']:
            self.segments.append(raw_text[ch_start:ch_end])
        self.segment_lengths = [len(sent.split()) for sent in self.segments]
        self.num_args = len(self.segments)
        self.proposition_ids = [i for i in range(self.num_args)]

        self.target_prop_ids = {"support" : set(), "attack" : set(), "union" : set()}
        for ((src_start, src_end), tgt) in ann_obj['reasons'] + ann_obj['evidences']:
            self.target_prop_ids["support"].add(tgt)
            self.target_prop_ids["union"].add(tgt)
            for i in range(src_start, src_end + 1):
                self.relations.append((i, tgt, "support"))
        self.relations_in_prop_ids = self.relations
        self.target_labels["support"] = [(i in self.target_prop_ids) for i in range(self.num_args)]
        self.target_labels["attack"] = [False for i in range(self.num_args)]


    @classmethod
    def find_split(cls, file_id):
        if file_id in cls.dataset_train_ids:
            return 'train'
        elif file_id in cls.dataset_val_ids:
            return 'val'
        else:
            return 'test'


class UKPDocument(ArgumentDocument):

    dataset_path_prefix = "raw/ArgumentAnnotatedEssays-2.0/brat-project-final/"
    dataset_glob_expression = "*.ann"
    dataset_type = "essays"
    dataset_train_ids = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                         37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53,
                         54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 69, 70,
                         73, 74, 75, 76, 78, 79, 80, 81, 83, 84, 85, 87, 88, 89, 90,
                         92, 93, 94, 95, 96, 99, 100, 101, 102, 105, 106, 107, 109,
                         110, 111, 112, 113, 114, 115, 116, 118, 120, 121, 122, 123,
                         124, 125, 127, 128, 130, 131, 132, 133, 134, 135, 137, 138,
                         140, 141, 143, 144, 145, 146, 147, 148, 150, 151, 152, 153,
                         155, 156, 157, 158, 159, 161, 162, 164, 165, 166, 167, 168,
                         170, 171, 173, 174, 175, 176, 177, 178, 179, 181, 183, 184,
                         185, 186, 188, 189, 190, 191, 194, 195, 196, 197, 198, 200,
                         201, 203, 205, 206, 207, 208, 209, 210, 213, 214, 215, 216,
                         217, 219, 222, 223, 224, 225, 226, 228, 230, 231, 232, 233,
                         235, 236, 237, 238, 239, 242, 244, 246, 247, 248, 249, 250,
                         251, 253, 254, 256, 257, 258, 260, 261, 262, 263, 264, 267,
                         268, 269, 270, 271, 272, 273, 274, 275, 276, 279, 280, 281,
                         282, 283, 284, 285, 286, 288, 290, 291, 292, 293, 294, 295,
                         296, 297, 298, 299, 300, 302, 303, 304, 305, 307, 308, 309,
                         311, 312, 313, 314, 315, 317, 318, 319, 320, 321, 323, 324,
                         325, 326, 327, 329, 330, 332, 333, 334, 336, 337, 338, 339,
                         340, 342, 343, 344, 345, 346, 347, 349, 350, 351, 353, 354,
                         ]
    dataset_val_ids = [
        356, 357, 358, 360, 361, 362, 363, 365, 366, 367, 368, 369,
        370, 371, 372, 374, 375, 376, 377, 378, 379, 380, 381, 383,
        384, 385, 387, 388, 389, 390, 391, 392, 394, 395, 396, 397,
        399, 400, 401, 402
    ]
    dataset_test_ids = [
        4, 5, 6, 21, 42, 52, 61, 68, 71, 72, 77, 82, 86, 91, 97, 98,
        103, 104, 108, 117, 119, 126, 129, 136, 139, 142, 149, 154,
        160, 163, 169, 172, 180, 182, 187, 192, 193, 199, 202, 204,
        211, 212, 218, 220, 221, 227, 229, 234, 240, 241, 243, 245,
        252, 255, 259, 265, 266, 277, 278, 287, 289, 301, 306, 310,
        316, 322, 328, 331, 335, 341, 348, 352, 355, 359, 364, 373,
        382, 386, 393, 398
    ]


    def __init__(self, file_id):
        # file_id: "essay012"
        super().__init__(file_id)
        self.adjusted_prop_para = []
        self.text_path = UKPDocument.dataset_path_prefix + f'{file_id}.txt'
        self.ann_path = UKPDocument.dataset_path_prefix + f'{file_id}.ann'


    def generate_relation_candidates(self):
        """For all argumentative segments, create possible link prediction
        candidate pairs. Note that only arguments within the same paragraph can
        be candidates.
        """
        for (src, tgt) in permutations(range(self.num_args), 2):
            if self.prop_para[src] != self.prop_para[tgt]:
                continue
            
            if (src, tgt) in self.supports:
                label = "support"
            elif (src, tgt) in self.attacks:
                label = "attack"
            else:
                label = "none"
            src_text = self.prop_id_to_text[src]
            tgt_text = self.prop_id_to_text[tgt]
            yield ((src, src_text), (tgt, tgt_text), label)


    def load_document(self):
        raw_text = open(self.text_path).read()

        props_info = dict() # T1 -> (begin, end)
        raw_rels = {'supports': [], 'attacks': []}

        para_offsets = []
        ix = 0
        while True:
            para_offsets.append(ix)
            try:
                ix = raw_text.index("\n", ix + 1)
            except ValueError:
                break
        para_offsets = np.array(para_offsets)
        for line in open(self.ann_path):
            if line[0] == 'T':
                fields = line.split('\t')
                position = fields[1].split()
                char_start = int(position[1])
                char_end = int(position[2])
                props_info[fields[0]] = (char_start, char_end)

            elif line[0] == 'R':
                fields = line.split('\t')
                rel_info = fields[1].split()
                rel_type = rel_info[0]
                src_id = rel_info[1].split(':')[1]
                tgt_id = rel_info[2].split(':')[1]
                raw_rels[rel_type].append((src_id, tgt_id))

        # old_ix: (T2, T1, T3)
        # sorted_prop_ids: [(5, 10), (20, 30), (35, 40)]
        old_ix, sorted_prop_ids = zip(*sorted(props_info.items(), key=lambda x: x[1]))
        inv_idx = {k: v for v, k in enumerate(old_ix)}
        self.supports = [(inv_idx[src], inv_idx[tgt]) for (src, tgt) in raw_rels['supports']]
        self.attacks = [(inv_idx[src], inv_idx[tgt]) for (src, tgt) in raw_rels['attacks']]
        self.target_prop_ids = {"support": set(), "attack": set(), "union": set()}
        for (s, t) in self.supports:
            self.target_prop_ids["support"].add(t)
            self.target_prop_ids["union"].add(t)

        for (s, t) in self.attacks:
            self.target_prop_ids["attack"].add(t)
            self.target_prop_ids["union"].add(t)

        self.prop_para = [int(np.searchsorted(para_offsets, start)) - 1
                          for start, _ in sorted_prop_ids]

        self.segments = []
        self.is_argument = []
        self.target_labels = {"support": [], "attack": []}
        self.num_args = len(sorted_prop_ids)
        self.prop_id_to_text = []

        prop_id_to_seg_id = dict()
        cur_char_ptr = 0

        self.proposition_ids = []
        self.adjusted_prop_para = []
        for ix, item in enumerate(sorted_prop_ids):

            # first add previous segment
            if cur_char_ptr < item[0]:
                prev_seg = raw_text[cur_char_ptr:item[0]]
                self.is_argument.append(False)
                self.segments.append(prev_seg)
                self.proposition_ids.append(-1)
                self.target_labels["support"].append(None)
                self.target_labels["attack"].append(None)
                self.adjusted_prop_para.append(int(np.searchsorted(para_offsets, cur_char_ptr)))

            # add self
            cur_seg = raw_text[item[0]:item[1]]
            prop_id_to_seg_id[ix] = len(self.segments)
            self.segments.append(cur_seg)
            self.is_argument.append(True)
            self.proposition_ids.append(ix)
            self.target_labels["support"].append(ix in self.target_prop_ids["support"])
            self.target_labels["attack"].append(ix in self.target_prop_ids["attack"])
            self.prop_id_to_text.append(cur_seg)
            cur_char_ptr = item[1]
            self.adjusted_prop_para.append(int(np.searchsorted(para_offsets, item[0])))

        if cur_char_ptr < len(raw_text) - 1:
            self.segments.append(raw_text[cur_char_ptr:])
            self.is_argument.append(False)
            self.target_labels["support"].append(None)
            self.target_labels["attack"].append(None)
            self.proposition_ids.append(-1)
            self.adjusted_prop_para.append(int(np.searchsorted(para_offsets, cur_char_ptr)))

        self.segment_lengths = [len(sent.split()) for sent in self.segments]

        self.relations = [] # using global ids
        self.relations_in_prop_ids = [] # using prop ids

        tgt_to_ids = dict()
        rel_tgt_to_ids = dict()
        

        for ix, relation_type in enumerate([self.supports, self.attacks]):
            for (src, tgt) in relation_type:
                src_real_id = prop_id_to_seg_id[src]
                tgt_real_id = prop_id_to_seg_id[tgt]

                if tgt_real_id not in rel_tgt_to_ids:
                    rel_tgt_to_ids[tgt_real_id] = [[], []]
                # support -> [src], attack -> [src]
                rel_tgt_to_ids[tgt_real_id][ix].append(src_real_id)

                if tgt not in tgt_to_ids:
                    tgt_to_ids[tgt] = [[], []]
                tgt_to_ids[tgt][ix].append(src)

        for t, (sup, att) in rel_tgt_to_ids.items():
            for s in sup:
                self.relations.append((s, t, "support"))
            for s in att:
                self.relations.append((s, t, "attack"))

        for t, (sup, att) in tgt_to_ids.items():
            for s in sup:
                self.relations_in_prop_ids.append((s, t, "support"))
            for s in att:
                self.relations_in_prop_ids.append((s, t, "attack"))


    @classmethod
    def find_split(cls, file_id):
        doc_num_id = int(file_id[5:])
        if doc_num_id in cls.dataset_train_ids:
            return 'train'
        elif doc_num_id in cls.dataset_val_ids:
            return 'val'
        else:
            return 'test'


class ECHRDocument(ArgumentDocument):

    dataset_type = 'ECHR'
    dataset_path_prefix = 'raw/echr_corpus/'
    train_ids = list(range(27))
    val_ids = list(range(27, 27+7))
    test_ids = list(range(27 + 7, 42))

    def __init__(self, file_id, doc_items):
        super().__init__(file_id)
        self.doc_items = doc_items

    def load_document(self):
        self.segments = self.doc_items['sentences']
        self.segment_lengths = [len(sent.split()) for sent in self.segments]
        self.proposition_ids = []

        prop_id = 0
        id2prop_id = dict()
        id2seg_id = dict()
        for i, item in enumerate(self.doc_items['is_argument']):
            cur_id = self.doc_items['clause_id'][i]
            if item:
                self.proposition_ids.append(prop_id)
                id2prop_id[cur_id] = prop_id
                id2seg_id[cur_id] = i
                prop_id += 1
            else:
                self.proposition_ids.append(-1)

        self.relations = []

        for (src, tgt) in self.doc_items['relations']:
            self.relations.append([id2seg_id[src], id2seg_id[tgt], 'support'])
        self.types = None
        self.target_labels = None


    @classmethod
    def make_all_data(cls):
        docs = {'train': [], 'val': [], 'test': []}
        stats = Statistics(cls.dataset_type)

        assert os.path.exists(cls.dataset_path_prefix + "ECHR_sentences.jsonl"), "To convert ECHR, please run `echr_preproc.py` first."
        
        raw_data = [json.loads(ln) for ln in open(cls.dataset_path_prefix + "ECHR_sentences.jsonl")]
        for i, doc in enumerate(raw_data):

            cur_doc = cls(i, doc)
            cur_doc.load_document()

            if i in cls.train_ids:
                dsplit = 'train'
                docs['train'].append(cur_doc)
            elif i in cls.val_ids:
                dsplit = 'val'
                docs['val'].append(cur_doc)
            else:
                dsplit = 'test'
                docs['test'].append(cur_doc)
            stats.update(cur_doc, dsplit)

        stats.print()
        #cls.make_target_prediction_data(docs)
        cls.make_relation_doc_level_data(docs)


class AbstCRTDocument(ArgumentDocument):

    dataset_path_prefix = 'raw/abstrct-master/AbstRCT_corpus/data/'
    dataset_type = 'abst_rct'

    def __init__(self, file_id, path):
        super().__init__(file_id)
        self.text_path = path[:-3] + 'txt'
        self.ann_path = path
        assert os.path.exists(self.text_path), f'Path not found! {self.text_path}'

    def load_document(self):
        raw_text = open(self.text_path).read()
        props_info = dict() # T1 -> (begin, end)
        raw_rels = {'Support': [], 'Attack': []}

        for line in open(self.ann_path):
            if line[0] == 'T':
                fields = line.split("\t")
                position = fields[1].split()
                char_start = int(position[1])
                char_end = int(position[2])
                props_info[fields[0]] = (char_start, char_end)

            elif line[0] == 'R':
                fields = line.split("\t")
                rel_info = fields[1].split()
                rel_type = rel_info[0]
                if rel_type not in raw_rels:
                    continue
                src_id = rel_info[1].split(':')[1]
                tgt_id = rel_info[2].split(':')[1]
                raw_rels[rel_type].append((src_id, tgt_id))

        # old_ix: (T3, T1, T2)
        # sorted_prop_ids ((639, 1018), (1611, 1673), (1674, 1776))
        old_ix, sorted_prop_ids = zip(*sorted(props_info.items(), key=lambda x: x[1]))
        inv_idx = {k: v for v, k in enumerate(old_ix)}
        self.supports = [(inv_idx[src], inv_idx[tgt]) for (src, tgt) in raw_rels['Support']]
        self.attacks = [(inv_idx[src], inv_idx[tgt]) for (src, tgt) in raw_rels['Attack']]
        self.target_prop_ids = {"support" : set(), "attack" : set(), "union" : set()}
        
        for (s, t) in self.supports:
            self.target_prop_ids["support"].add(t)
            self.target_prop_ids["union"].add(t)

        for (s, t) in self.attacks:
            self.target_prop_ids["attack"].add(t)
            self.target_prop_ids["union"].add(t)

        prop_id_to_seg_id = dict() # map proposition id to natural segment id
        cur_char_ptr = 0
        self.segments = []
        self.target_labels = []
        self.is_argument = []
        self.num_args = len(sorted_prop_ids)
        self.prop_id_to_text = []
        self.proposition_ids = [] # -1 for non-arg, otherwise count from 0

        for ix, item in enumerate(sorted_prop_ids):
            # if there's anything, add as non-arg
            if cur_char_ptr < item[0] and item[0] - cur_char_ptr > 2:
                prev_seg = raw_text[cur_char_ptr: item[0]]
                self.is_argument.append(False)
                self.segments.append(prev_seg)
                self.proposition_ids.append(-1)
                self.target_labels.append(None)

            # add current argument (proposition)
            cur_seg = raw_text[item[0]: item[1]]
            prop_id_to_seg_id[ix] = len(self.segments)
            self.segments.append(cur_seg)
            self.is_argument.append(True)
            self.proposition_ids.append(ix)
            self.target_labels.append(ix in self.target_prop_ids)
            self.prop_id_to_text.append(cur_seg)
            cur_char_ptr = item[1]

        if cur_char_ptr < len(raw_text) - 1:
            cur_seg = raw_text[cur_char_ptr:]
            if cur_seg.strip() != '':
                self.segments.append(cur_seg)
                self.is_argument.append(False)
                self.target_labels.append(None)
                self.proposition_ids.append(-1)

        self.relations = []
        self.relations_in_prop_ids = []
        rel_tgt_to_ids = dict()
        tgt_to_ids = dict()

        for ix, relation_type in enumerate([self.supports, self.attacks]):
            for (src, tgt) in relation_type:
                src_real_id = prop_id_to_seg_id[src]
                tgt_real_id = prop_id_to_seg_id[tgt]
                if tgt_real_id not in rel_tgt_to_ids:
                    rel_tgt_to_ids[tgt_real_id] = [[], []]
                rel_tgt_to_ids[tgt_real_id][ix].append(src_real_id)

                if tgt not in tgt_to_ids:
                    tgt_to_ids[tgt] = [[], []]
                tgt_to_ids[tgt][ix].append(src)

        for t, (sup, att) in rel_tgt_to_ids.items():
            for s in sup:
                self.relations.append((s, t, "support"))
            for s in att:
                self.relations.append((s, t, "attack"))

        self.segment_lengths = [len(sent.split()) for sent in self.segments]


    @classmethod
    def make_all_data(cls):
        docs = {'train': [], 'val': [], 'test': []}
        stats = Statistics(cls.dataset_type)

        for dsplit in ['train', 'dev', 'test']:
            paths = glob.glob(cls.dataset_path_prefix + dsplit + '/*/*.ann')
            if dsplit == 'dev':
                dsplit = 'val'
            for path in paths:
                file_id = os.path.basename(path).split('.')[0]
                topic = path.split('/')[-2]
                file_id = (topic, file_id)
                cur_doc = cls(file_id, path)
                cur_doc.load_document()
                docs[dsplit].append(cur_doc)
                stats.update(cur_doc, dsplit)
        stats.print()
        cls.make_relation_doc_level_data(docs)