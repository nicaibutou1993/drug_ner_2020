import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import glob
import os

maxlen = 256
epochs = 10
batch_size = 8
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

# bert配置
path = 'E:/project/bert_ner/chinese_L-12_H-768_A-12/'
config_path = path + 'bert_config.json'
checkpoint_path = path + 'bert_model.ckpt'
dict_path = path + 'vocab.txt'

is_split_long_text = True


def load_train_data(path):
    datas = []
    for text_path in glob.glob(path):

        text_id = os.path.basename(text_path).replace('.txt', '')

        with open(text_path, encoding='utf-8') as text_f, open(text_path.replace('txt', 'ann'),
                                                               encoding='utf-8') as an_f:
            text = text_f.read()

            ans = an_f.readlines()

            label_entities = []

            for an in ans:
                _, entity, start_index, end_index, sub_name = an.split()

                start_index = int(start_index)
                end_index = int(end_index)

                label_entities.append((start_index, end_index, entity, sub_name))

            label_entities = sorted(label_entities, key=lambda x: x[0])

            start = 0
            d = []
            for start_index, end_index, entity, sub_name in label_entities:

                # print(text[start:start_index])
                if start < start_index:
                    d.append([text[start:start_index], 'O', start, start_index])
                d.append([text[start_index:end_index], entity, start_index, end_index])
                # print(text[start_index:end_index])
                start = end_index
            if start < len(text):
                d.append([text[start:], 'O', start, len(text)])

            # for start_index, end_index, entity, sub_name in label_entities:
            #
            #     print(text[start:start_index])
            #     if start < start_index:
            #         d.append([text[start:start_index], 'O'])
            #     d.append([text[start_index:end_index], entity])
            #     print(text[start_index:end_index])
            #     start = end_index
            #     print()
            # if start < len(text):
            #     d.append([text[start:], 'O'])

            datas.append(d)

    return datas


train_path = 'data/all_train/*.txt'
train_data = load_train_data(train_path)

dev_path = 'data/dev/*.txt'
valid_data = load_train_data(dev_path)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射

labels = ["O", "DRUG", "DRUG_INGREDIENT", "DISEASE", "SYMPTOM", "SYNDROME", "DISEASE_GROUP", "FOOD",
          "FOOD_GROUP", "PERSON_GROUP", "DRUG_GROUP", "DRUG_DOSAGE", "DRUG_TASTE", "DRUG_EFFICACY"]

id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


class data_generator(DataGenerator):
    """数据生成器
    """

    def split_long_text(self, item, batch_token_ids, batch_segment_ids, batch_labels):
        try:
            left_item = []
            if len(item) > 0:
                token_ids, labels = [tokenizer._token_start_id], [0]

                is_continue = True
                for w, l, _, _ in item:
                    w_token_ids = tokenizer.encode(w)[0][1:-1]

                    if len(w_token_ids) > maxlen:
                        continue

                    if len(token_ids) + len(w_token_ids) < maxlen and is_continue:
                        token_ids += w_token_ids
                        if l == 'O':
                            labels += [0] * len(w_token_ids)
                        else:
                            B = label2id[l] * 2 + 1
                            I = label2id[l] * 2 + 2
                            labels += ([B] + [I] * (len(w_token_ids) - 1))
                    else:
                        is_continue = False

                    if not is_continue:
                        left_item.append((w, l, _, _))
                token_ids += [tokenizer._token_end_id]
                labels += [0]
                segment_ids = [0] * len(token_ids)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)

                if len(left_item) > 0 and len(left_item) < len(item):
                    self.split_long_text(left_item, batch_token_ids, batch_segment_ids, batch_labels)

            return
        except Exception as e:

            print(len(batch_token_ids))
            print(left_item)
            print(e)

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []

        _num = 0
        for is_end, item in self.sample(random):
            # token_ids, labels = [tokenizer._token_start_id], [0]
            # for w, l, _, _ in item:
            #     w_token_ids = tokenizer.encode(w)[0][1:-1]
            #     if len(token_ids) + len(w_token_ids) < maxlen:
            #         token_ids += w_token_ids
            #         if l == 'O':
            #             labels += [0] * len(w_token_ids)
            #         else:
            #             B = label2id[l] * 2 + 1
            #             I = label2id[l] * 2 + 2
            #             labels += ([B] + [I] * (len(w_token_ids) - 1))
            #     else:
            #         break
            # token_ids += [tokenizer._token_end_id]
            # labels += [0]
            # segment_ids = [0] * len(token_ids)
            # batch_token_ids.append(token_ids)
            # batch_segment_ids.append(segment_ids)
            # batch_labels.append(labels)

            self.split_long_text(item, batch_token_ids, batch_segment_ids, batch_labels)

            _num += len(batch_token_ids)

            if len(batch_token_ids) >= self.batch_size or is_end:
                print()
                print(_num)
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
                _num = 0


"""
后面的代码使用的是bert类型的模型，如果你用的是albert，那么前几行请改为：

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='albert',
)

output_layer = 'Transformer-FeedForward-Norm'
output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
"""

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

# class NamedEntityRecognizer(ViterbiDecoder):
#     """命名实体识别器
#     """
#
#     def recognize(self, text):
#         tokens = tokenizer.tokenize(text)
#         while len(tokens) > 512:
#             tokens.pop(-2)
#         mapping = tokenizer.rematch(text, tokens)
#         token_ids = tokenizer.tokens_to_ids(tokens)
#         segment_ids = [0] * len(token_ids)
#         token_ids, segment_ids = to_array([token_ids], [segment_ids])
#         nodes = model.predict([token_ids, segment_ids])[0]
#         labels = self.decode(nodes)
#         entities, starting = [], False
#         for i, label in enumerate(labels):
#             if label > 0:
#                 if label % 2 == 1:
#                     starting = True
#                     entities.append([[i], id2label[(label - 1) // 2]])
#                 elif starting:
#                     entities[-1][0].append(i)
#                 else:
#                     starting = False
#             else:
#                 starting = False
#
#         return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
#                 for w, l in entities]

trans = [[9.07323062e-01, -1.06969833e+00, -7.21103966e-01, 1.78381354e-01
             , -2.42564106e+00, 2.99284786e-01, -2.90545321e+00, 4.08968389e-01
             , -2.53147531e+00, 4.57888961e-01, -2.70983028e+00, 1.70171872e-01
             , -2.91970396e+00, 3.39036472e-02, -2.96700096e+00, -3.47899884e-01
             , -1.67979991e+00, 3.99931192e-01, -3.39036942e+00, 2.94282556e-01
             , -3.29567623e+00, -2.95859456e-01, -1.47292113e+00, 3.09826106e-01
             , -2.71119261e+00, 4.58037138e-01, -1.94187212e+00, 9.10078511e-02
             , -3.88973355e+00]
    , [-1.54191136e+00, -2.54735304e-03, -4.56077427e-01, -5.19874156e-01
             , -8.42664540e-02, -7.49395669e-01, -4.74997520e-01, -1.19165450e-01
             , -1.46249563e-01, -5.86455405e-01, -7.57628858e-01, 1.00353463e-02
             , -4.01426822e-01, -4.23456877e-01, -6.76197231e-01, -1.98812187e-01
             , -1.71763510e-01, 6.44510314e-02, -3.19108576e-01, -4.87989575e-01
             , -8.51185203e-01, -3.82121116e-01, 2.31751464e-02, -5.10118008e-01
             , -7.27967501e-01, -4.57156062e-01, -4.17250127e-01, -5.07189810e-01
             , -9.88694251e-01]
    , [-1.04382753e+00, -3.83047968e-01, -2.81555176e-01, -9.64363813e-02
             , -3.09622914e-01, -5.43374002e-01, -5.41535020e-01, -3.42405260e-01
             , -3.67299914e-01, -2.73054272e-01, -8.86999965e-01, -1.80295065e-01
             , -2.08373398e-01, -1.91956773e-01, -3.93448398e-02, -1.86531633e-01
             , -2.84390807e-01, 6.79655299e-02, -1.89603880e-01, -4.54156280e-01
             , -3.74325633e-01, 1.37955517e-01, -3.99772882e-01, -4.49539274e-01
             , -2.81923681e-01, 2.08407510e-02, -4.13733304e-01, -1.74705595e-01
             , -2.54215032e-01]
    , [-2.24193788e+00, -6.26695752e-01, -2.84929071e-02, -1.50040174e+00
             , 2.10624599e+00, -2.71614480e+00, -3.08150554e+00, -8.30330372e-01
             , -8.60334635e-01, -1.07925391e+00, -1.59313500e+00, -4.92765278e-01
             , -1.78467488e+00, -5.24252772e-01, -1.29359996e+00, -7.96587050e-01
             , -5.96080840e-01, -4.17918354e-01, -1.32456827e+00, -5.95808923e-01
             , -1.55535698e+00, -6.10820115e-01, -4.38368171e-01, -1.62274349e+00
             , -2.33072567e+00, -1.74079746e-01, -1.09276772e+00, -3.27926517e-01
             , -1.63630211e+00]
    , [-1.66377175e+00, 2.43963543e-02, -4.26269293e-01, -8.12915087e-01
             , 1.48048806e+00, -1.37993383e+00, -2.09931874e+00, -4.77502257e-01
             , -5.78539193e-01, -1.25016123e-01, -8.60588968e-01, -2.70847708e-01
             , -7.93434143e-01, 8.56252089e-02, -6.67222977e-01, -5.35393238e-01
             , -5.17672539e-01, -4.98884112e-01, -3.53254944e-01, -5.38865209e-01
             , -9.67133522e-01, -6.76427305e-01, -1.51179016e-01, -8.70792091e-01
             , -2.19081235e+00, -2.88837373e-01, -5.79945683e-01, -4.03775394e-01
             , -9.55067694e-01]
    , [-3.60574746e+00, -9.31654096e-01, -9.55969095e-01, -2.07759023e+00
             , -2.66501451e+00, -2.57579136e+00, 1.37480617e+00, -1.30023575e+00
             , -1.64800107e+00, -1.11614656e+00, -1.98989916e+00, -4.43077087e-01
             , -1.77238846e+00, -5.82029998e-01, -1.85957634e+00, -1.76337218e+00
             , -2.62174344e+00, -1.12616420e+00, -2.36557531e+00, -7.35932171e-01
             , -2.27982807e+00, -1.50492513e+00, -1.80523825e+00, -2.11454940e+00
             , -2.68951392e+00, -1.06126201e+00, -9.00140703e-01, -1.14396703e+00
             , -2.26027131e+00]
    , [-7.10632265e-01, -5.46527445e-01, -4.92442876e-01, -1.11882949e+00
             , -1.67209423e+00, 3.88586283e-01, 1.38507855e+00, -6.04676530e-02
             , -5.21397531e-01, -1.95766628e-01, -9.90779459e-01, -9.69043300e-02
             , -1.41442156e+00, -4.30205643e-01, -5.17461836e-01, -4.59979832e-01
             , -7.68094480e-01, -3.34260255e-01, -7.46344507e-01, -3.79686765e-02
             , -1.27657080e+00, -9.36597288e-01, -6.48901403e-01, -1.06451690e+00
             , -1.77002597e+00, 2.15938285e-01, -4.63402957e-01, 6.51472732e-02
             , -1.15686440e+00]
    , [-3.03247976e+00, -7.52908587e-01, -5.54605722e-01, -8.53923798e-01
             , -1.11220384e+00, -1.68657768e+00, -1.48064125e+00, -1.22791970e+00
             , 2.40575480e+00, -1.40008521e+00, -3.44530535e+00, -6.57467365e-01
             , -2.95014405e+00, -1.02234089e+00, -3.21145225e+00, -7.28750646e-01
             , -9.38493967e-01, -9.47046280e-01, -2.22063613e+00, -9.81655419e-01
             , -3.31954312e+00, -5.65401554e-01, -8.74462903e-01, -1.38175559e+00
             , -1.67807710e+00, -5.84321499e-01, -4.81146842e-01, -9.25144017e-01
             , -2.32708097e+00]
    , [-1.60221362e+00, -6.40796304e-01, -2.03881085e-01, 2.65592128e-01
             , -6.03643000e-01, -1.71044633e-01, -8.32331181e-01, 3.53398733e-02
             , 2.11731791e+00, -1.17557406e-01, -2.86737156e+00, -4.13220018e-01
             , -2.31505966e+00, -6.20220959e-01, -2.97384834e+00, -4.22968924e-01
             , -3.62695247e-01, -1.39474779e-01, -1.10409403e+00, -3.18121970e-01
             , -2.28174472e+00, -4.42424983e-01, -3.29722524e-01, 8.13946575e-02
             , -1.16765285e+00, -1.99759692e-01, -6.24948561e-01, -5.70155203e-01
             , -1.64159548e+00]
    , [-3.41314554e+00, -6.72244191e-01, -9.05084610e-02, -1.06480777e+00
             , -9.48606670e-01, -1.20664692e+00, -1.18060589e+00, -1.10433078e+00
             , -2.62519288e+00, -1.20594788e+00, 2.16455340e+00, -1.35976422e+00
             , -3.26961470e+00, -8.90719533e-01, -3.09221840e+00, -7.76306152e-01
             , -1.07048202e+00, -7.62787521e-01, -2.57662272e+00, -6.67411804e-01
             , -3.23724413e+00, -1.13850820e+00, -1.12558854e+00, -1.48472357e+00
             , -1.07520378e+00, -1.02868390e+00, -1.12167001e+00, -7.37785101e-01
             , -3.03713346e+00]
    , [-2.08212614e+00, -1.12239218e+00, -5.55721581e-01, -4.33458477e-01
             , -1.37191558e+00, -2.72176594e-01, -1.76885140e+00, -7.00242281e-01
             , -3.10501575e+00, -5.92510939e-01, 1.99216676e+00, -6.20586216e-01
             , -3.45363235e+00, -9.45855975e-01, -3.17631006e+00, -8.84207606e-01
             , -1.55988371e+00, -6.46277308e-01, -2.41470814e+00, -6.71987832e-01
             , -3.65529633e+00, -8.39962721e-01, -1.55126619e+00, -5.15694559e-01
             , -1.63492703e+00, -5.62714279e-01, -1.17213225e+00, -7.31951118e-01
             , -2.25037289e+00]
    , [-2.52977371e+00, -5.68252027e-01, -8.83541107e-02, -1.58572763e-01
             , -6.62318766e-01, -3.48774523e-01, -8.60625148e-01, -7.94595599e-01
             , -1.23439109e+00, -6.28405035e-01, -2.64477062e+00, -9.30465937e-01
             , 2.65547943e+00, -3.05981599e-02, -2.10955262e+00, -2.81766057e-01
             , -5.43511927e-01, -3.43261868e-01, -1.79764974e+00, 1.10941119e-02
             , -2.03167462e+00, -2.74132013e-01, -3.09789211e-01, -7.43825376e-01
             , -7.00616539e-01, -4.58061129e-01, -8.54699552e-01, -2.48708785e-01
             , -2.46527910e+00]
    , [-1.42779005e+00, -7.40054309e-01, -3.21376622e-02, -3.41148376e-01
             , -1.07333183e+00, 1.59594461e-01, -1.44682431e+00, -7.17244923e-01
             , -2.06233549e+00, -6.19745106e-02, -3.20224357e+00, 3.08396399e-01
             , 2.11556816e+00, -9.54335332e-02, -2.34170699e+00, -6.31780863e-01
             , -6.06613517e-01, -4.16104764e-01, -1.90457571e+00, 3.76867414e-01
             , -3.25337720e+00, -9.80913401e-01, -1.29767120e+00, -5.14380753e-01
             , -1.54456210e+00, -7.14457361e-03, -6.76002145e-01, -6.28718315e-03
             , -2.34904504e+00]
    , [-1.69549513e+00, 3.83374351e-03, 1.37227938e-01, -4.27170932e-01
             , -3.28962594e-01, -2.24737555e-01, -5.91187589e-02, -7.60263264e-01
             , -1.64574814e+00, -6.68546975e-01, -2.03771234e+00, -6.62245527e-02
             , -1.48558116e+00, 3.20480317e-02, 2.58210945e+00, -2.59070396e-01
             , -2.94649959e-01, -3.12690496e-01, -6.67737484e-01, -4.70194340e-01
             , -1.01325381e+00, -5.63166678e-01, -3.25003624e-01, -6.54709414e-02
             , -2.34950885e-01, 1.26893595e-01, -4.13702279e-01, 3.35487053e-02
             , -8.86344612e-01]
    , [-1.31625187e+00, -4.40551311e-01, -2.05482021e-01, -6.83194399e-01
             , -9.33299959e-01, -2.81129956e-01, -9.42178607e-01, -4.97696102e-01
             , -2.50282812e+00, -9.32831526e-01, -2.26383448e+00, -4.21331614e-01
             , -2.08131480e+00, -8.44384953e-02, 2.06927848e+00, -2.28125870e-01
             , -5.73740721e-01, 3.01831782e-01, -1.49488270e+00, -5.13428628e-01
             , -2.88819766e+00, -7.21760452e-01, -8.64362717e-01, 8.74721855e-02
             , -8.51613641e-01, -6.11493111e-01, -4.93734479e-01, -4.01496410e-01
             , -1.60317373e+00]
    , [-1.62751067e+00, -1.53549090e-01, -4.13025975e-01, -7.26973891e-01
             , -7.35050201e-01, -1.79834032e+00, -1.98643279e+00, -3.43438238e-01
             , -7.44663417e-01, -1.00185096e+00, -1.20222640e+00, -2.98986018e-01
             , -1.20593011e+00, -4.50937837e-01, -7.36330628e-01, -6.73491538e-01
             , 2.98653626e+00, -4.28115070e-01, -2.18405461e+00, -3.57524902e-01
             , -1.27111125e+00, -1.74351096e-01, -3.10649395e-01, -1.02486861e+00
             , -1.14278758e+00, -4.35133949e-02, -2.18027011e-01, -4.38944459e-01
             , -1.31833172e+00]
    , [-6.95026815e-01, 1.14314981e-01, 1.16181433e-01, 1.47424312e-02
             , -1.67392999e-01, -4.00272369e-01, -9.90484595e-01, -7.27806017e-02
             , -5.66177443e-02, -5.75972974e-01, -1.08163416e+00, -4.92771208e-01
             , -6.41363561e-01, -1.13261014e-01, -3.94104511e-01, 9.92348194e-01
             , 8.46784413e-01, -1.83081150e-01, -1.02392006e+00, 1.96558744e-01
             , -2.81007409e-01, -3.57844740e-01, -4.47988153e-01, 2.97967792e-02
             , -7.89488912e-01, -3.19777697e-01, -4.91339892e-01, -1.72316849e-01
             , -1.02341342e+00]
    , [-2.20141196e+00, -4.14764851e-01, 4.21180092e-02, -4.43876624e-01
             , -7.72289038e-01, -6.09469891e-01, -1.44214857e+00, -3.35356116e-01
             , -6.78611457e-01, -3.90409589e-01, -2.25481892e+00, -1.08734824e-01
             , -1.13853502e+00, -3.04189056e-01, -1.21353006e+00, -7.80296385e-01
             , -1.13446200e+00, -1.45930564e+00, 2.27713537e+00, -5.44776320e-01
             , -1.57677567e+00, -6.27027094e-01, -4.30160910e-01, -8.87626588e-01
             , -9.35229897e-01, -1.25565618e-01, -8.75476658e-01, -7.16633126e-02
             , -1.25688195e+00]
    , [-4.07352000e-01, -3.44211221e-01, -1.06412746e-01, -6.40815318e-01
             , -5.70692122e-01, 1.46020815e-01, -1.02831399e+00, 7.09096789e-02
             , -8.47934186e-01, 3.57806198e-02, -1.26897025e+00, -4.27907526e-01
             , -7.68448532e-01, -1.05773218e-01, -7.28694856e-01, -1.08312154e+00
             , -3.85018915e-01, 2.91626453e-01, 1.55988300e+00, -2.10161269e-01
             , -1.11678088e+00, -9.34473932e-01, -5.24991512e-01, 7.04172790e-01
             , -8.32654536e-01, -6.90837979e-01, -7.01642454e-01, 4.01755601e-01
             , -1.16468978e+00]
    , [-2.38663578e+00, -2.95489490e-01, -1.08033746e-01, -4.14574653e-01
             , -5.99805951e-01, -4.39456195e-01, -1.11713564e+00, -7.47196496e-01
             , -1.18360853e+00, -1.00618231e+00, -2.38496470e+00, -7.11681902e-01
             , -2.06769085e+00, -4.61237282e-02, -1.43790984e+00, -7.26535618e-01
             , -6.89558387e-01, -7.40791380e-01, -1.56809032e+00, -5.25382638e-01
             , 2.57610202e+00, -6.27341568e-01, -8.57393503e-01, -1.01520252e+00
             , -1.27574456e+00, -3.24941754e-01, -5.89956641e-01, -5.63836694e-01
             , -1.27444530e+00]
    , [-1.59760761e+00, -8.06379437e-01, -9.25821364e-02, -9.61456537e-01
             , -9.11985397e-01, 2.44940549e-01, -1.43285871e+00, -3.79903585e-01
             , -1.29092872e+00, -7.64786720e-01, -2.07167029e+00, -1.99967146e-01
             , -1.64264488e+00, -1.29361644e-01, -1.62457836e+00, -9.67924535e-01
             , -5.23706675e-01, 8.78473222e-01, -1.52705407e+00, -2.91235119e-01
             , 2.27853298e+00, -6.09652400e-01, -6.60223126e-01, 2.00709477e-01
             , -1.06473339e+00, -8.07695925e-01, -7.23637760e-01, 4.78644408e-02
             , -1.78307641e+00]
    , [-1.66998911e+00, 2.66521294e-02, -3.21164340e-01, -6.64081872e-01
             , -8.12371373e-01, -1.00253832e+00, -1.21456528e+00, -7.65986919e-01
             , -9.91934419e-01, -7.06511080e-01, -1.41257572e+00, -3.81081671e-01
             , -1.54348850e+00, -2.68161416e-01, -1.10905886e+00, -1.41783580e-01
             , -4.47706312e-01, -6.97209299e-01, -1.34931409e+00, -5.20147800e-01
             , -1.15889668e+00, -4.84370053e-01, 1.94328201e+00, -8.71804416e-01
             , -7.28614926e-01, 4.18523774e-02, -6.27854228e-01, -7.13224888e-01
             , -1.75843871e+00]
    , [-1.25511336e+00, 9.45350304e-02, 1.52301699e-01, -3.41706008e-01
             , -3.50160539e-01, -7.13543415e-01, -5.37232280e-01, -2.03374341e-01
             , -3.91315579e-01, -8.25592816e-01, -7.49326110e-01, -2.42754310e-01
             , -1.22269762e+00, -3.88615340e-01, -5.57305872e-01, -2.27559149e-01
             , -5.35009384e-01, -5.07654488e-01, -6.90686762e-01, -4.20702666e-01
             , -9.39432919e-01, -5.70464969e-01, 1.44046807e+00, -5.12688637e-01
             , -4.17686462e-01, -2.59994209e-01, -5.94286859e-01, -6.67062178e-02
             , -9.95318770e-01]
    , [-2.88751292e+00, -1.07172799e+00, -8.25133681e-01, -2.09260178e+00
             , -2.96753669e+00, -2.01592350e+00, -2.44905782e+00, -9.78261828e-01
             , -1.66162598e+00, -1.33871543e+00, -1.34376132e+00, -6.02677166e-01
             , -2.11320734e+00, -8.80822957e-01, -2.09217072e+00, -1.23720014e+00
             , -2.19687557e+00, -1.39837945e+00, -2.60092473e+00, -1.07008767e+00
             , -1.88070428e+00, -1.24705541e+00, -9.89054084e-01, -3.34119296e+00
             , 2.02612424e+00, -1.89350545e+00, -2.29655075e+00, -1.14164495e+00
             , -2.31533074e+00]
    , [-1.10990024e+00, -2.20633179e-01, -5.25087893e-01, -1.26449049e-01
             , -1.77162826e+00, 8.66238654e-01, -1.63240278e+00, -2.97480017e-01
             , -8.00645292e-01, -3.87403607e-01, -1.10623050e+00, -8.37421894e-01
             , -1.15315473e+00, -1.36153921e-02, -7.48399198e-01, -8.24602902e-01
             , -4.67003107e-01, 1.51547551e-01, -1.22318196e+00, -4.08805646e-02
             , -1.32934558e+00, -9.27149832e-01, -7.34277844e-01, -1.04545557e+00
             , 1.87528133e+00, -2.26321772e-01, -6.44532084e-01, -2.13035733e-01
             , -9.62781250e-01]
    , [-2.12671232e+00, -4.11880948e-02, 9.15724412e-02, -8.87573004e-01
             , -5.70861340e-01, -9.98528004e-01, -1.10011971e+00, -5.59601545e-01
             , -7.52505660e-01, -6.65199876e-01, -1.60100806e+00, -8.10408950e-01
             , -1.36322522e+00, -3.49365562e-01, -7.29620636e-01, -1.69499725e-01
             , -9.51065421e-01, -1.14389944e+00, -2.09483027e+00, -5.73813841e-02
             , -9.69869554e-01, -7.78796732e-01, -7.50597596e-01, -3.54664057e-01
             , -1.91449881e+00, -1.66132820e+00, 2.13080573e+00, -6.96168184e-01
             , -1.49024260e+00]
    , [-1.31482041e+00, -2.98657238e-01, -1.92223832e-01, 6.41231462e-02
             , -8.30214322e-01, 2.58928388e-01, -8.09421659e-01, -1.57921419e-01
             , -6.02761954e-02, -5.27485907e-01, -9.74020481e-01, -5.34259081e-01
             , -1.05721855e+00, -2.44464174e-01, -7.22323716e-01, -4.44794446e-01
             , -5.83046675e-01, -2.57672966e-01, -1.35249043e+00, 1.45731464e-01
             , -8.42067182e-01, -6.55344784e-01, -1.67893037e-01, 7.99149513e-01
             , -1.25696576e+00, 7.28018999e-01, 1.91346180e+00, -2.75267810e-01
             , -9.99732852e-01]
    , [-1.95720065e+00, -3.82291913e-01, -1.37908429e-01, -6.55523837e-01
             , -4.16437298e-01, -5.96875966e-01, -6.76749945e-01, -4.88201410e-01
             , -7.63665438e-01, -9.70722079e-01, -1.69462597e+00, -1.40442997e-01
             , -1.39039326e+00, -1.26156315e-01, -7.68437266e-01, -2.12138787e-01
             , -5.08628599e-02, -3.02820146e-01, -1.03845549e+00, -2.76055813e-01
             , -9.97112691e-01, -3.80736798e-01, -3.75435680e-01, -9.34021950e-01
             , -6.17276132e-01, -5.76009274e-01, -4.49434489e-01, -6.53828323e-01
             , 2.11845660e+00]
    , [-1.85664535e+00, -3.95494998e-01, -1.62059620e-01, -2.85538554e-01
             , -1.44769001e+00, -3.74322772e-01, -1.47831738e+00, -1.00701964e+00
             , -1.16043794e+00, -1.05183017e+00, -1.78927588e+00, -1.18230593e+00
             , -2.49106097e+00, -5.04089832e-01, -2.09779334e+00, -9.47192252e-01
             , -6.00206077e-01, 6.46055877e-01, -1.31728578e+00, 8.62120688e-02
             , -1.72976255e+00, -1.15643215e+00, -8.80988061e-01, 5.82811952e-01
             , -1.33672881e+00, 8.19171071e-02, -9.82628226e-01, -3.62462670e-01
             , 1.52860045e+00]]

trans = np.array(trans)


#
# def evaluate(data):
#     """评测函数
#     """
#     X, Y, Z = 1e-10, 1e-10, 1e-10
#     for d in tqdm(data):
#         text = ''.join([i[0] for i in d])
#         R = set(NER.recognize(text))
#         T = set([tuple(i) for i in d if i[1] != 'O'])
#         X += len(R & T)
#         Y += len(R)
#         Z += len(T)
#     f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
#     return f1, precision, recall
#
#
# class Evaluator(keras.callbacks.Callback):
#     def __init__(self):
#         self.best_val_f1 = 0
#
#     def on_epoch_end(self, epoch, logs=None):
#         trans = K.eval(CRF.trans)
#         NER.trans = trans
#         print(NER.trans)
#         f1, precision, recall = evaluate(valid_data)
#         # 保存最优
#         if f1 >= self.best_val_f1:
#             self.best_val_f1 = f1
#             model.save_weights('checkpoint/ner_best_model.weights')
#         print(
#             'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
#             (f1, precision, recall, self.best_val_f1)
#         )
#
#
# model.load_weights('checkpoint/ner_best_model.weights')
#
# f1, precision, recall = evaluate(valid_data)
# print(f1, precision, recall)
#
# if __name__ == '__main__':
#
#     evaluator = Evaluator()
#     train_generator = data_generator(train_data, batch_size)
#
#     model.fit_generator(
#         train_generator.forfit(),
#         steps_per_epoch=len(train_generator),
#         epochs=epochs,
#         callbacks=[evaluator]
#     )
#
# else:
#
#     model.load_weights('checkpoint/ner_best_model.weights')


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def cut_txt(self, text):

        num = 510
        cut = len(text) // num + 1
        texts = []
        for i in range(cut):
            texts.append(text[i * num:(i + 1) * num])

        return texts

    def recognize(self, text):
        texts = self.cut_txt(text)

        datas = []
        cut_start_index = 0
        for text in texts:

            tokens = tokenizer.tokenize(text)
            while len(tokens) > 512:
                tokens.pop(-2)
            mapping = tokenizer.rematch(text, tokens)
            token_ids = tokenizer.tokens_to_ids(tokens)

            # text = text.lower()
            # text_arr = list(text)
            #
            # token_ids = ['[CLS]'] + text_arr + ['[SEP]']
            #
            #
            #
            # #token_ids = tokenizer.tokens_to_ids(['[CLS]'])+token_ids +tokenizer.tokens_to_ids(['[SEP]'])
            #
            #
            # while len(token_ids) > 512:
            #     token_ids.pop(-2)
            # mapping = tokenizer.rematch(text, token_ids)
            #
            # token_ids, _ = tokenizer.encode(token_ids, maxlen=512)

            # token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            token_ids, segment_ids = to_array([token_ids], [segment_ids])
            nodes = model.predict([token_ids, segment_ids])[0]
            labels = self.decode(nodes)
            entities, starting = [], False
            for i, label in enumerate(labels):
                if label > 0:
                    if label % 2 == 1:
                        starting = True
                        entities.append([[i], id2label[(label - 1) // 2]])
                    elif starting:
                        entities[-1][0].append(i)
                    else:
                        starting = False
                else:
                    starting = False

            datas.extend([(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l, mapping[w[0]][0] + cut_start_index,
                           mapping[w[-1]][-1] + 1 + cut_start_index)
                          for w, l in entities])

            cut_start_index += len(text)

            # return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l,mapping[w[0]][0],mapping[w[-1]][-1] + 1)
            #         for w, l in entities]

            # print([(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l,mapping[w[0]][0],mapping[w[-1]][-1] + 1)
            #         for w, l in entities])
        # return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
        #         for w, l in entities]
        return datas


NER = NamedEntityRecognizer(trans=trans, starts=[0], ends=[0])


#NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in data:
        text = ''.join([i[0] for i in d])

        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        # print(R)
        # R = sorted(R,key=lambda x : x[2])
        # T = sorted(T, key=lambda x: x[2])
        # print(R)
        # print(T)

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        print(trans)
        NER.trans = trans
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('checkpoint/%s_ner_best_model.weights' % str(epoch))
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


# model.load_weights('checkpoint/8_ner_best_model.weights')
# f1, precision, recall = evaluate(valid_data)
# print(f1, precision, recall)


def test():
    """评测函数
    """
    files = glob.glob('data/test/*')
    for file_name in files:

        text_id = os.path.basename(file_name).replace('.txt', '')
        with open(file_name, encoding='utf-8') as test_f, open('data/test_an/' + text_id + '.ann', mode='a',
                                                               encoding='utf-8') as an_f:

            text = test_f.read()

            datas = NER.recognize(text)
            for i, data in enumerate(datas):

                an_f.write(
                    'T' + str(i) + '\t' + data[1] + ' ' + str(data[2]) + ' ' + str(data[3]) + '\t' + data[0])

                if i < len(datas) - 1:
                    an_f.write('\n')


model.load_weights('checkpoint/8_ner_best_model.weights')
test()

if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('checkpoint/ner_best_model.weights')
