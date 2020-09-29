# -*- coding: utf-8 -*-
import os
import glob
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Input, Dense
from keras.models import Model

maxlen = 256


def strQ2B(ustring):
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def text_segmentate(text, maxlen, seps='\n', strips=None):
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


def load_data(path):
    datas = []
    for text_path in glob.glob(path):

        text_id = os.path.basename(text_path).replace('.txt', '')

        with open(text_path, encoding='utf-8') as text_f, open(text_path.replace('txt', 'ann'),
                                                               encoding='utf-8') as an_f:
            text = text_f.read()
            text = strQ2B(text)

            ans = an_f.readlines()

            label_entities = {}

            for an in ans:
                _, entity, start_index, end_index, sub_name = an.split()

                start_index = int(start_index)
                end_index = int(end_index) - 1

                sub_names = label_entities.get(entity, {})

                indexes = sub_names.get(sub_name, [])

                indexes.append((start_index, end_index))

                sub_names[sub_name] = indexes
                label_entities[entity] = sub_names

            words = list(text)
            labels = ['O'] * len(words)

            if label_entities is not None:
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        sub_name = sub_name.lower()
                        for start_index, end_index in sub_index:
                            assert ''.join(words[start_index:end_index + 1]) == sub_name
                            if start_index == end_index:
                                labels[start_index] = 'S-' + key
                            else:
                                labels[start_index] = 'B-' + key
                                labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)

            seps = '\n。！？!?；;，, '
            texts = text_segmentate(text, maxlen, seps)

            start, end = 0, 0
            for seg_text in texts:
                end += len(seg_text)
                seg_word = words[start:end]
                seg_label = labels[start:end]
                datas.append((seg_word, seg_label, text_id))

                # if len(seg_word) > maxlen:
                #     print(seg_word)

                start = end

    return datas


train_path = 'data/train/*.txt'
train_data = load_data(train_path)

dev_path = 'data/dev/*.txt'
dev_data = load_data(dev_path)

label_list = ["X", "B-DRUG", "B-DRUG_INGREDIENT", "B-DISEASE", "B-SYMPTOM", "B-SYNDROME", "B-DISEASE_GROUP", "B-FOOD",
              "B-FOOD_GROUP", "B-PERSON_GROUP", "B-DRUG_GROUP", "B-DRUG_DOSAGE", "B-DRUG_TASTE", "B-DRUG_EFFICACY",
              "I-DRUG", "I-DRUG_INGREDIENT", "I-DISEASE", "I-SYMPTOM", "I-SYNDROME", "I-DISEASE_GROUP", "I-FOOD",
              "I-FOOD_GROUP", "I-PERSON_GROUP", "I-DRUG_GROUP", "I-DRUG_DOSAGE", "I-DRUG_TASTE", "I-DRUG_EFFICACY",
              "S-DRUG", "S-DRUG_INGREDIENT", "S-DISEASE", "S-SYMPTOM", "S-SYNDROME", "S-DISEASE_GROUP", "S-FOOD",
              "S-FOOD_GROUP", "S-PERSON_GROUP", "S-DRUG_GROUP", "S-DRUG_DOSAGE", "S-DRUG_TASTE", "S-DRUG_EFFICACY",
              "O", "[CLS]", "[SEP]"]

num_class = len(label_list)
batch_size = 16
# bert配置
path = 'E:/project/bert_ner/chinese_L-12_H-768_A-12/'
config_path = path + 'bert_config.json'
checkpoint_path = path + 'bert_model.ckpt'
dict_path = path + 'vocab.txt'

# from sklearn.model_selection import train_test_split
#
# train_data, dev_data = train_test_split(datas, test_size=0.1)

label_map = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label_map.items()}
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (words, labels, _) in self.sample(random):

            '''数据集中的数据不能修改，否则每一轮都会迭代修改'''
            # words += ['[SEP]']
            # words = ['[CLS]'] + words

            token_ids, segment_ids = tokenizer.encode(words, maxlen=maxlen)

            token_ids = tokenizer.tokens_to_ids(['[CLS]']) + token_ids + tokenizer.tokens_to_ids(['[SEP]'])
            segment_ids = [0] + segment_ids + [0]

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            label_ids = [[label_map[label]] for label in labels]

            label_ids += [[label_map['[SEP]']]]
            label_ids = [[label_map['[CLS]']]] + label_ids
            batch_labels.append(label_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids, batch_segment_ids, batch_labels], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


train_generator = data_generator(train_data, batch_size)
# valid_generator = data_generator(dev_data, batch_size)

# 加载预训练模型（12层）
bert_model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=True,
    with_pool=False
)

output = Dense(units=num_class, activation='softmax')(bert_model.output)

model = Model(bert_model.inputs, output)

true_label_ids = Input(shape=(None, 1))

train_model = Model(bert_model.inputs + [true_label_ids], output)

loss = K.sparse_categorical_crossentropy(true_label_ids, output)

padding_mask = bert_model.get_layer('Embedding-Token').output_mask
padding_mask = K.cast(padding_mask, K.floatx())

loss = K.sum(loss * padding_mask) / K.sum(padding_mask)
train_model.add_loss(loss)
train_model.compile(optimizer=Adam(2e-5))

from collections import Counter


class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = self.get_entity_bios(label_path, self.id2label)
            pre_entities = self.get_entity_bios(pre_path, self.id2label)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])

    def get_entity_bios(self, seq, id2label):
        """Gets entities from sequence.
        note: BIOS
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
            # >>> get_entity_bios(seq)
            [['PER', 0,1], ['LOC', 3, 3]]
        """
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("S-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[2] = indx
                chunk[0] = tag.split('-')[1]
                chunks.append(chunk)
                chunk = (-1, -1, -1)
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return chunks


def evaluate(data):
    mectrics = SeqEntityScore(id2label)
    for (batch_token_ids, batch_segment_ids, batch_labels), _ in data:

        preds = model.predict([batch_token_ids, batch_segment_ids])

        preds = preds.argmax(axis=2)

        batch_labels = batch_labels.squeeze(axis=2)

        for i, label in enumerate(batch_labels):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif batch_labels[i][j] == label_map.get('[SEP]'):
                    mectrics.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(id2label.get(batch_labels[i][j]))
                    temp_2.append(preds[i][j])

    eval_info, entity_info = mectrics.result()

    results = {str(key): str(value) for key, value in eval_info.items()}
    info = "-".join([str(key) + ":" + str(value) for key, value in results.items()])
    print(info)
    for key in sorted(entity_info.keys()):
        print("******* %s results ********" % key)
        info = "-".join([str(key) + ":" + str(value) for key, value in entity_info[key].items()])
        print(info)
    return results


class Evaluator(keras.callbacks.Callback):
    def __init__(self, savename):
        self.best_val_acc = 0.
        self.savename = savename

    # def on_epoch_begin(self,epoch, logs=None):
    #     val_acc = evaluate(valid_generator)

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(self.savename)


def f1_evaluate(data):
    tmp = 0

    batch_token_ids, batch_segment_ids,text_ids = [], [],[]
    for words, labels, text_id in data:

        token_ids, segment_ids = tokenizer.encode(words, maxlen=maxlen)

        token_ids = tokenizer.tokens_to_ids(['[CLS]']) + token_ids + tokenizer.tokens_to_ids(['[SEP]'])
        segment_ids = [0] + segment_ids + [0]

        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

        text_ids.append(text_id)

        tmp += 1
        if len(batch_token_ids) == batch_size or tmp == len(data):
            batch_token_ids = sequence_padding(batch_token_ids)
            batch_segment_ids = sequence_padding(batch_segment_ids)

            preds = model.predict([batch_token_ids, batch_segment_ids])


            batch_token_ids, batch_segment_ids,text_ids = [], [],[]


# model.load_weights('checkpoint/clue_ner_softmax_best_model.weights')
#
# evaluate(valid_generator)


if __name__ == '__main__':
    # 训练predecessor
    model_evaluator = Evaluator('checkpoint/ner_softmax_best_model.weights')
    # print(evaluate(valid_generator,predecessor_model))
    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        callbacks=[model_evaluator]
    )
