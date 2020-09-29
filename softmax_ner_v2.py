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
import numpy as np

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


def load_train_data(path):
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
                            labels[start_index:end_index + 1] = [key] * len(sub_name)

            seps = '\n。！？!?；;，, '
            texts = text_segmentate(text, maxlen, seps)

            start, end = 0, 0
            for seg_text in texts:
                end += len(seg_text)
                seg_word = words[start:end]
                seg_label = labels[start:end]
                datas.append((seg_word, seg_label, text_id))

                start = end

    return datas


def load_test_data(path):
    datas = []
    for text_path in glob.glob(path):

        text_id = os.path.basename(text_path).replace('.txt', '')

        with open(text_path, encoding='utf-8') as text_f:
            text = text_f.read()
            text = strQ2B(text)
            words = list(text)
            seps = '\n。！？!?；;，, '
            texts = text_segmentate(text, maxlen, seps)
            start, end = 0, 0
            for seg_text in texts:
                end += len(seg_text)
                seg_word = words[start:end]
                datas.append((seg_word, [], text_id))
                start = end
    return datas


train_path = 'data/train/*.txt'
train_data = load_train_data(train_path)

dev_path = 'data/dev/*.txt'
dev_data = load_test_data(dev_path)

label_list = ["X", "DRUG", "DRUG_INGREDIENT", "DISEASE", "SYMPTOM", "SYNDROME", "DISEASE_GROUP", "FOOD",
              "FOOD_GROUP", "PERSON_GROUP", "DRUG_GROUP", "DRUG_DOSAGE", "DRUG_TASTE", "DRUG_EFFICACY",
              "O", "[CLS]", "[SEP]"]

num_class = len(label_list)
batch_size = 16
# bert配置
path = 'E:/project/bert_ner/chinese_L-12_H-768_A-12/'
config_path = path + 'bert_config.json'
checkpoint_path = path + 'bert_model.ckpt'
dict_path = path + 'vocab.txt'

label_map = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label_map.items()}
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (words, labels, _) in self.sample(random):

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


class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup

    def get_entity_bios(self, seq):
        id2label = self.id2label
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]

            if tag not in ["O", "[CLS]", "[SEP]", "X"]:
                if chunk[0] == -1:
                    chunk[0] = tag
                    chunk[1] = indx
                    chunk[2] = indx
                elif chunk[0] == tag:
                    chunk[2] = indx
                elif chunk[0] != tag:
                    chunk[2] = chunk[2] + 1
                    chunks.append(chunk)
                    chunk = [-1, -1, -1]
                    chunk[0] = tag
                    chunk[1] = indx
                    chunk[2] = indx
            else:
                if chunk[0] != -1:
                    chunk[2] = chunk[2] + 1
                    chunks.append(chunk)
                chunk = [-1, -1, -1]

        return chunks


class Evaluator(keras.callbacks.Callback):
    def __init__(self, savename):
        self.best_f1 = 0.
        self.savename = savename

    def on_epoch_end(self, epoch, logs=None):
        recall, precision, f1 = dev_evaluate(dev_data)

        if f1 > self.best_f1:
            self.model.save_weights(self.savename)
            self.best_f1 = f1


dev_root_path = 'data/dev/'


def dev_evaluate(dev_data):
    found_entities = predict(dev_data)

    text_ids = found_entities.keys()

    origin_entities = {}
    for text_id in text_ids:
        with open(dev_root_path + str(text_id) + '.ann', encoding='utf-8') as an_f:
            ans = an_f.readlines()
            entities = []
            for an in ans:
                _, entity, start_index, end_index, sub_name = an.split()
                start_index = int(start_index)
                end_index = int(end_index)
                entities.append([entity, start_index, end_index])
            origin_entities[text_id] = entities

    origin, found, right = 0.0, 0.0, 0.0
    for text_id in text_ids:
        founds = found_entities.get(text_id)
        origins = origin_entities.get(text_id)
        rights = []
        rights.extend([found for found in founds if found in origins])

        origin += len(origins)
        found += len(founds)
        right += len(rights)

    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)

    print(recall, precision, f1)

    return recall, precision, f1


def predict(data):
    mectrics = SeqEntityScore(id2label)
    tmp = 0
    pred_res = []
    batch_token_ids, batch_segment_ids, text_ids = [], [], []
    for words, _, text_id in data:

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

            preds = preds.argmax(axis=2)

            for i, (pred, token_ids, text_id) in enumerate(zip(preds, batch_token_ids, text_ids)):
                remove_padding_index = np.nonzero(token_ids)[0]
                pred_res.append((pred[remove_padding_index[1:-1]], text_id))
            batch_token_ids, batch_segment_ids, text_ids = [], [], []

    entities_map = {}

    for seq, text_id in pred_res:
        entities = entities_map.get(text_id, [])
        entities.extend(mectrics.get_entity_bios(seq))
        entities_map[text_id] = entities
    # print(entities_map)

    return entities_map


#model.load_weights('checkpoint/v2_ner_softmax_best_model.weights')

test_path = 'data/test/*.txt'
test_data = load_test_data(test_path)


def test_predict():
    entities_map = predict(test_data)

    for text_id, entities in entities_map.items():

        # if text_id == '1010':
        #     print()

        with open('data/test/' + text_id + '.txt', encoding='utf-8') as text_f, \
                open('data/test_an/' + text_id + '.ann', mode='a', encoding='utf-8') as an_f:
            text = text_f.read()
            text = strQ2B(text)

            for i, entity in enumerate(entities):
                start = entity[1]
                end = entity[2]
                name = text[start:end].strip()
                end = start + len(name)

                if any(filter in name for filter in ';。、,.,/()'):
                    continue

                if end - start >= 2 and name and text[start:end].strip():
                    an_f.write(
                        'T' + str(i) + '\t' + entity[0] + ' ' + str(start) + ' ' + str(end) + '\t' + text[start:end])

                if i < len(entities) - 1:
                    an_f.write('\n')


dev_evaluate(dev_data)

if __name__ == '__main__':

    model_evaluator = Evaluator('checkpoint/v2_ner_softmax_best_model.weights')
    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        callbacks=[model_evaluator]
    )
