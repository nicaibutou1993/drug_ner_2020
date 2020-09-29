from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import ViterbiDecoder
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
import numpy as np
import glob
import os
import re

maxlen = 256
epochs = 10
batch_size = 16
bert_layers = 12
learing_rate = 1e-5
crf_lr_multiplier = 1000

# bert配置
path = 'E:/project/bert_ner/chinese_L-12_H-768_A-12/'
config_path = path + 'bert_config.json'
checkpoint_path = path + 'bert_model.ckpt'
dict_path = path + 'vocab.txt'

'''全角转半角'''
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

'''
针对文本比较长的，进行切分
'''
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


'''
加载训练数据
'''
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
                            if start_index == end_index:
                                labels[start_index] = 'S-' + key
                            else:
                                labels[start_index] = 'B-' + key
                                labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)

            '''文本过长，进行拆分，一篇文章 可能生成多个 数据 '''
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

'''
加载验证或者测试数据
'''
def load_test_data(path):
    datas = []
    for text_path in glob.glob(path):

        text_id = os.path.basename(text_path).replace('.txt', '')

        with open(text_path, encoding='utf-8') as text_f:
            text = text_f.read()
            text = strQ2B(text)
            words = list(text)

            '''文本过长，进行拆分，一篇文章 可能生成多个 数据 '''
            seps = '\n。！？!?；;，, '
            texts = text_segmentate(text, maxlen, seps)
            start, end = 0, 0
            for seg_text in texts:
                end += len(seg_text)
                seg_word = words[start:end]
                datas.append((seg_word, [], text_id))
                start = end
    return datas


train_path = 'data/all_train/*.txt'
train_data = load_train_data(train_path)


dev_path = 'data/dev/*.txt'
dev_data = load_test_data(dev_path)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射
labels = ["O", "B-DRUG", "B-DRUG_INGREDIENT", "B-DISEASE", "B-SYMPTOM", "B-SYNDROME", "B-DISEASE_GROUP", "B-FOOD",
          "B-FOOD_GROUP", "B-PERSON_GROUP", "B-DRUG_GROUP", "B-DRUG_DOSAGE", "B-DRUG_TASTE", "B-DRUG_EFFICACY",
          "I-DRUG", "I-DRUG_INGREDIENT", "I-DISEASE", "I-SYMPTOM", "I-SYNDROME", "I-DISEASE_GROUP", "I-FOOD",
          "I-FOOD_GROUP", "I-PERSON_GROUP", "I-DRUG_GROUP", "I-DRUG_DOSAGE", "I-DRUG_TASTE", "I-DRUG_EFFICACY",
          "S-DRUG", "S-DRUG_INGREDIENT", "S-DISEASE", "S-SYMPTOM", "S-SYNDROME", "S-DISEASE_GROUP", "S-FOOD",
          "S-FOOD_GROUP", "S-PERSON_GROUP", "S-DRUG_GROUP", "S-DRUG_DOSAGE", "S-DRUG_TASTE", "S-DRUG_EFFICACY"
          ]

id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []

        for is_end, (words, labels, _) in self.sample(random):

            token_ids, segment_ids = tokenizer.encode(words, maxlen=maxlen)

            token_ids = tokenizer.tokens_to_ids(['[CLS]']) + token_ids + tokenizer.tokens_to_ids(['[SEP]'])
            segment_ids = [0] * len(token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            label_ids = [[0]] + [[label2id[label]] for label in labels] + [[0]]

            batch_labels.append(label_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learing_rate),
    metrics=[CRF.sparse_accuracy]
)


class NamedEntityRecognizer(ViterbiDecoder):


    '''
    获取实体，去掉 bios 前缀，生成 【（实体，start_index,end_index）】
    '''
    def get_entity_bios(self, seq):
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("S-"):
                if chunk[2] != -1:
                    chunk[2] = chunk[2] + 1
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[2] = indx + 1
                chunk[0] = tag.split('-')[1]
                chunks.append(chunk)
                chunk = (-1, -1, -1)
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunk[2] = chunk[2] + 1
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = tag.split('-')[1]
                if _type == chunk[0]:
                    chunk[2] = indx
                if indx == len(seq) - 1:
                    chunk[2] = chunk[2] + 1
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunk[2] = chunk[2] + 1
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return chunks


    '''
    针对数据预测
    
    返回 {文章id：对应的seq entities}
    '''
    def recognize(self, datas):
        batch_token_ids, batch_segment_ids, text_ids = [], [], []
        tmp = 0
        pred_res = []
        for words, _, text_id in datas:

            token_ids, segment_ids = tokenizer.encode(words, maxlen=maxlen)

            token_ids = tokenizer.tokens_to_ids(['[CLS]']) + token_ids + tokenizer.tokens_to_ids(['[SEP]'])
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            text_ids.append(text_id)
            tmp += 1
            if len(batch_token_ids) == batch_size or tmp == len(datas):
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)

                preds = model.predict([batch_token_ids, batch_segment_ids])
                # preds = preds.argmax(axis=2)
                for i, (pred, token_ids, text_id) in enumerate(zip(preds, batch_token_ids, text_ids)):

                    '''为每一条数据，去除padding，并解码'''
                    remove_padding_index = np.nonzero(token_ids)[0]
                    labels = self.decode(pred[remove_padding_index])
                    pred_res.append((labels[1:-1], text_id))
                batch_token_ids, batch_segment_ids, text_ids = [], [], []

        entities_map = {}


        '''之前针对长文本进行 拆分，最终预测时，需要对所有长文本数据集预测的结果 进行合并'''
        for seq, text_id in pred_res:
            entities = entities_map.get(text_id, [])
            entities.extend(self.get_entity_bios(seq))
            entities_map[text_id] = entities

        return entities_map


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

dev_root_path = 'data/dev/'


'''
验证集评估
'''
def dev_evaluate(dev_data):
    found_entities = NER.recognize(dev_data)
    text_ids = found_entities.keys()

    '''加载验证集 相关的标注数据 '''
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


    '''计算f1,recall,precision'''
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

    return recall, precision, f1


class Evaluator(keras.callbacks.Callback):
    def __init__(self,data):
        self.best_val_f1 = 0
        self.data = data

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        recall, precision, f1 = dev_evaluate(self.data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('checkpoint/crf_ner_best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

model.load_weights('checkpoint/crf_ner_best_model.weights')

test_root_path = 'data/test/'
test_root_an_path = 'data/test_crf_an/'

test_path = test_root_path + '*.txt'
test_data = load_test_data(test_path)

'''针对测试数据进行预测'''
def test_predict():
    entities_map = NER.recognize(test_data)

    for text_id, entities in entities_map.items():

        with open(test_root_path + text_id + '.txt', encoding='utf-8') as text_f, \
                open(test_root_an_path + text_id + '.ann', mode='a', encoding='utf-8') as an_f:
            text = text_f.read()
            text = strQ2B(text)

            for i, entity in enumerate(entities):
                start = entity[1]
                end = entity[2]
                name = text[start:end].strip()
                end = start + len(name)

                '''针对生成结果 做简单的预清洗'''
                if any(filter in name for filter in ';。、,.,/()~'):
                    continue

                if not text[start:end].strip():
                    continue

                if bool(re.search('\d|[A-Za-z]', name)):
                    continue

                #if end - start >= 2 and name and text[start:end].strip():
                an_f.write(
                    'T' + str(i) + '\t' + entity[0] + ' ' + str(start) + ' ' + str(end) + '\t' + text[start:end])

                if i < len(entities) - 1:
                    an_f.write('\n')

test_predict()


if __name__ == '__main__':

    evaluator = Evaluator(dev_data)
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('checkpoint/crf_ner_best_model.weights')
