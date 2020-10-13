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
batch_size = 16
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

# bert配置
path = 'E:/project/bert_ner/chinese_L-12_H-768_A-12/'
config_path = path + 'bert_config.json'
checkpoint_path = path + 'bert_model.ckpt'
dict_path = path + 'vocab.txt'


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

                print(text[start:start_index])
                if start < start_index:
                    d.append([text[start:start_index], 'O',start,start_index])
                d.append([text[start_index:end_index], entity,start_index,end_index])
                print(text[start_index:end_index])
                start = end_index
                print()
            if start < len(text):
                d.append([text[start:], 'O',start,len(text)])

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

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l,_,_ in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


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

trans = [[6.32001042e-01, -4.52449560e-01, -2.83744663e-01, 2.67034739e-01
             , -8.43532264e-01, -1.42335415e-01, -5.69641531e-01, 3.47325057e-01
             , -1.52193797e+00, 4.42719221e-01, -1.38135374e+00, 2.95293510e-01
             , -1.11414683e+00, 5.75204492e-01, -2.03461170e+00, -9.59157720e-02
             , -7.90248752e-01, 2.60497957e-01, -1.15866673e+00, 2.86298156e-01
             , -1.49805522e+00, -7.09916055e-01, -4.09749508e-01, 5.66356242e-01
             , -1.55662537e+00, 3.59291226e-01, -8.15140784e-01, 5.11369050e-01
             , -9.90797639e-01]
    , [-2.96236962e-01, -1.15704365e-01, 6.02378026e-02, 4.94668633e-02
             , -1.71955451e-01, -2.37263385e-02, -3.89670342e-01, -4.28241156e-02
             , -3.48846734e-01, -3.46537858e-01, -7.16764450e-01, -2.52106488e-01
             , -4.64352727e-01, 2.96393298e-02, -2.10213065e-01, -3.60750675e-01
             , -4.00546432e-01, 4.95702513e-02, -1.08984202e-01, 1.16302535e-01
             , -4.30591345e-01, -3.16737950e-01, -2.01783061e-01, -2.60518938e-02
             , -3.67363185e-01, -4.22712207e-01, -1.25987023e-01, -5.06982505e-01
             , -7.48496175e-01]
    , [-4.20617342e-01, -5.72697110e-02, -2.18420193e-01, -2.85863638e-01
             , 3.17124538e-02, -5.32887764e-02, -3.50542635e-01, -2.36390620e-01
             , -2.12152869e-01, -1.56723633e-01, -5.00704348e-01, 4.48101796e-02
             , -8.20697621e-02, 5.91920391e-02, -4.03337479e-01, -1.55436277e-01
             , -7.89841563e-02, -4.03700829e-01, -8.08076337e-02, -1.89334914e-01
             , -1.97886959e-01, -2.39617497e-01, -3.33987594e-01, 7.91702271e-02
             , -2.28977919e-01, -4.54837680e-01, 1.42108753e-01, 8.43201727e-02
             , -2.03254059e-01]
    , [-8.29863250e-01, -3.87411624e-01, -4.00497556e-01, -1.91249043e-01
             , 1.28092122e+00, 3.44262943e-02, -6.16755664e-01, -2.14569762e-01
             , -9.48510349e-01, -4.31169957e-01, -8.42013896e-01, -3.56448144e-01
             , -6.11470699e-01, -2.91267931e-01, -8.03202868e-01, 1.62947357e-01
             , -2.19377145e-01, -1.43094594e-02, 1.72218587e-03, -5.61661005e-01
             , -1.28965116e+00, 1.27103597e-01, -2.20213845e-01, -2.95815557e-01
             , -1.38116017e-01, -3.47310096e-01, -3.38834971e-01, -1.24357991e-01
             , -4.24250335e-01]
    , [-3.95530194e-01, 1.56708434e-01, -2.81342506e-01, -1.45127848e-01
             , 8.92386198e-01, -1.00735143e-01, -4.55442697e-01, -5.63998930e-02
             , -6.54183209e-01, -6.80539384e-02, -3.61354828e-01, -2.38503665e-02
             , 8.47362578e-02, -4.02165413e-01, -1.48075387e-01, -1.80499911e-01
             , -4.06567454e-01, -4.45648104e-01, 1.24906018e-01, 7.64260814e-02
             , -4.66267347e-01, 6.42354935e-02, -3.22197080e-01, -3.16993266e-01
             , -1.56385407e-01, -2.49009341e-01, -4.04272452e-02, -2.02182874e-01
             , -1.85999751e-01]
    , [-8.68660212e-01, -3.76461238e-01, -3.56011599e-01, -3.32402527e-01
             , -4.53358628e-02, -1.57696873e-01, 8.69476914e-01, -4.21253264e-01
             , -6.84393346e-01, -9.04222280e-02, -5.57395577e-01, -3.58526289e-01
             , -4.41168621e-02, 6.53315708e-02, -6.09623075e-01, -3.16435635e-01
             , -1.52825907e-01, -6.69635758e-02, -5.13232172e-01, -2.83482283e-01
             , -3.44038785e-01, -1.84809253e-01, -8.85739401e-02, 3.20064910e-02
             , -1.40556127e-01, -1.30140394e-01, -1.44191876e-01, -6.01774501e-03
             , -5.27516484e-01]
    , [-1.98802680e-01, -1.31137401e-01, -2.92337358e-01, -9.31150243e-02
             , -5.84091783e-01, 3.24046686e-02, 5.85575223e-01, 5.10283373e-02
             , -5.28570533e-01, -4.22460496e-01, -4.80991483e-01, -5.29696822e-01
             , -2.10740836e-03, -5.01997471e-01, -3.73137534e-01, 7.34301135e-02
             , -9.70244557e-02, -1.98004186e-01, -3.16022336e-01, -2.57501453e-01
             , -6.15241289e-01, -3.35309684e-01, 8.92420020e-03, -1.50499037e-02
             , -7.19378531e-01, -2.89270341e-01, -4.23573673e-01, 3.62533033e-02
             , -3.58265609e-01]
    , [-7.03330040e-01, -2.73524016e-01, 1.16730638e-01, 3.55217233e-02
             , -4.17417258e-01, -3.90215740e-02, -2.10441321e-01, -4.67814356e-01
             , 1.42786956e+00, -1.76239014e-01, -1.42299831e+00, -5.70279002e-01
             , -2.69167572e-01, -4.79184717e-01, -1.28841400e+00, -1.95112437e-01
             , 8.69044065e-02, -2.88482487e-01, -4.91102755e-01, -2.49061242e-01
             , -1.30204058e+00, 1.13091238e-01, -7.20433220e-02, -2.57125884e-01
             , -6.28338307e-02, -8.02462697e-02, -1.51734784e-01, 4.06636577e-03
             , -4.20660108e-01]
    , [-6.28104627e-01, -2.94657826e-01, -3.76186054e-03, 4.57046092e-01
             , -4.78683531e-01, -1.12235568e-01, -3.49085689e-01, 3.82308513e-02
             , 1.25376368e+00, 2.80730575e-01, -1.14262009e+00, -1.13324579e-02
             , -8.23022246e-01, -4.21079367e-01, -2.17252898e+00, -5.28380096e-01
             , -7.31472790e-01, 9.16096792e-02, -2.97060072e-01, -2.18215883e-01
             , -1.37115395e+00, -1.14733912e-01, -2.09666252e-01, -1.68792725e-01
             , -8.53339493e-01, -4.09011751e-01, -2.81534195e-01, -2.05931023e-01
             , -1.03554547e+00]
    , [-1.20296037e+00, -3.24937940e-01, -4.50650454e-01, -4.87251222e-01
             , -4.79975224e-01, -4.41275895e-01, -3.97142828e-01, -5.48047602e-01
             , -1.74150872e+00, -5.30134678e-01, 8.88496697e-01, -9.64034677e-01
             , -1.38486910e+00, -1.16581404e+00, -1.76828134e+00, -4.59488094e-01
             , -4.88468319e-01, -6.12838864e-01, -9.31935310e-01, -7.15306342e-01
             , -1.21119237e+00, -7.61211395e-01, -5.71115196e-01, -6.06985152e-01
             , -9.69403982e-01, -6.53217912e-01, -7.74758816e-01, -5.68631291e-01
             , -5.29188275e-01]
    , [-7.28521526e-01, -3.98472220e-01, 2.59824307e-03, 1.97321653e-01
             , -2.31795520e-01, -3.28924209e-01, -2.01299220e-01, -4.08115834e-01
             , -1.57901156e+00, -2.62804270e-01, 1.26986921e+00, 1.81823261e-02
             , -7.15805471e-01, -2.85054684e-01, -1.67780483e+00, -3.08706731e-01
             , -5.33322275e-01, -6.32611886e-02, -4.40775931e-01, 1.84277803e-01
             , -1.66238606e+00, -6.46931410e-01, -8.49289834e-01, -1.11119160e-02
             , -5.37913740e-01, -2.14514256e-01, -6.64369404e-01, 2.49531180e-01
             , -8.68637085e-01]
    , [-1.00521517e+00, 5.18719889e-02, -4.91528720e-01, -6.03931785e-01
             , -3.31064284e-01, -1.11593775e-01, -2.06602976e-01, 5.64787202e-02
             , -1.00138199e+00, -8.98983359e-01, -1.26940632e+00, -5.98038614e-01
             , 1.32216907e+00, -6.53402209e-01, -9.10285234e-01, -3.00145984e-01
             , -1.44536898e-01, -4.59434718e-01, -2.95674264e-01, -3.13832343e-01
             , -1.04261327e+00, -4.78067696e-01, -5.47417819e-01, -6.25062048e-01
             , -5.38072348e-01, -2.62383223e-01, -5.43480992e-01, -7.43985057e-01
             , -6.95374072e-01]
    , [-7.46485353e-01, -1.45443588e-01, -3.02405894e-01, -3.68329823e-01
             , -3.77303362e-01, 9.81113836e-02, -3.89090106e-02, -2.72341877e-01
             , -8.77616465e-01, -9.57168117e-02, -1.13016689e+00, 4.42546487e-01
             , 1.35560262e+00, -3.62275019e-02, -1.15263677e+00, -1.64137408e-01
             , -4.88466412e-01, -2.20191628e-01, -4.76152822e-02, 2.21161336e-01
             , -1.18920243e+00, 6.24276586e-02, -1.97967589e-01, -4.79811430e-01
             , -2.65611619e-01, -3.80149990e-01, -4.89032477e-01, -6.25528991e-02
             , -3.88201296e-01]
    , [-1.46651804e+00, -1.34204522e-01, -3.94631892e-01, -3.22190791e-01
             , -5.33895433e-01, -5.38494468e-01, -3.13631803e-01, -8.87850642e-01
             , -1.86540103e+00, -6.99140489e-01, -1.27402413e+00, -7.05054522e-01
             , -8.19222331e-01, -1.33281887e+00, 1.26613116e+00, 7.98723996e-02
             , -2.95828462e-01, -1.81612164e-01, -5.59263825e-01, -7.34838128e-01
             , -1.58449686e+00, -8.16657618e-02, -2.12286189e-01, -3.44501168e-01
             , -5.89070022e-01, -5.46159387e-01, -6.64904892e-01, -4.72602218e-01
             , -9.75567222e-01]
    , [-4.29205000e-01, -5.23955422e-03, -7.05573335e-02, -7.62227178e-01
             , -5.59287131e-01, -3.24654937e-01, -5.11810243e-01, -5.46132445e-01
             , -2.22429895e+00, -5.24188697e-01, -1.38922262e+00, -1.02303326e-02
             , -9.00452852e-01, -2.59854525e-01, 1.18367410e+00, 9.64102671e-02
             , -8.01157117e-01, 1.98020432e-02, -6.08073473e-01, -1.97039247e-01
             , -1.73427689e+00, -8.63154382e-02, -5.40236533e-01, 1.08906731e-01
             , -2.08495304e-01, -1.55165479e-01, -2.00104848e-01, -1.01742230e-01
             , -5.77570021e-01]
    , [-6.89111173e-01, -1.76907569e-01, 1.68290153e-01, 6.34450987e-02
             , 4.30903323e-02, -5.59732169e-02, -3.04427981e-01, -7.50869811e-02
             , -3.88601005e-01, -8.41573253e-02, -1.76966190e-03, -1.15477003e-01
             , -4.49818522e-01, -2.18581498e-01, -6.89176202e-01, 1.01308897e-01
             , 1.00788558e+00, -3.58423799e-01, 6.01736568e-02, -4.43803877e-01
             , -2.26326600e-01, 1.82588603e-02, -2.01356169e-02, -3.49382848e-01
             , -3.61071378e-01, 4.95794155e-02, -4.18306023e-01, 2.97734905e-02
             , -3.71110469e-01]
    , [-2.22269874e-02, -2.40580603e-01, -3.86165172e-01, -4.56234694e-01
             , -2.48360559e-01, -2.66124696e-01, -1.00291416e-01, -2.69235551e-01
             , -4.46177065e-01, -1.40787318e-01, -8.05834293e-01, -3.09343100e-01
             , -4.15868871e-02, 3.11870966e-02, -5.47299862e-01, 2.93219507e-01
             , 2.16812730e-01, 5.57985231e-02, 2.33901124e-02, -3.77876788e-01
             , -1.05732262e+00, -3.12323689e-01, 9.87700932e-03, 3.36044699e-01
             , -6.40398264e-02, -1.92858204e-01, -3.78967732e-01, -2.52189249e-01
             , -3.37000072e-01]
    , [-7.76930809e-01, -2.53448635e-01, -1.80847332e-01, -2.67931502e-02
             , 1.48927391e-01, -8.67856592e-02, -1.08909450e-01, -4.05258596e-01
             , -7.28147388e-01, -3.33686888e-01, -5.54234326e-01, -5.21447361e-01
             , -2.75965273e-01, 1.87569503e-02, -8.91463697e-01, 1.06926486e-01
             , 5.70008531e-02, -2.79297411e-01, 1.49476051e+00, -3.89439762e-01
             , -1.12494016e+00, 5.02959937e-02, -1.32558107e-01, -4.62364346e-01
             , -9.08429176e-02, -6.05097353e-01, -2.06619799e-01, -3.58695060e-01
             , -3.18833023e-01]
    , [1.13773547e-01, -4.38476861e-01, -2.13355109e-01, -5.13416469e-01
             , -3.97770733e-01, 8.03785101e-02, -3.92356843e-01, -1.24492913e-01
             , -3.26863885e-01, -4.60331351e-01, -4.57687527e-01, -3.27602327e-01
             , 3.55116278e-02, -4.87614989e-01, -2.98962802e-01, -2.42479339e-01
             , -2.22967640e-01, 6.36788726e-01, 8.89933407e-01, 3.42081785e-02
             , -8.23722720e-01, 3.36515121e-02, -4.24283445e-01, 1.93519264e-01
             , -2.50600785e-01, -5.89083314e-01, -5.74869104e-02, -2.74968952e-01
             , -5.77984512e-01]
    , [-1.75448763e+00, -2.03295071e-02, -3.65181953e-01, -3.65094632e-01
             , 2.27145031e-02, -5.28067708e-01, -6.05359912e-01, -3.25679779e-01
             , -1.32674277e+00, -7.73535192e-01, -1.53816569e+00, -7.25746989e-01
             , -5.87506652e-01, -7.72482216e-01, -1.70305383e+00, -4.75155264e-01
             , -6.54745281e-01, -5.67288220e-01, -9.29695010e-01, -1.23232937e+00
             , 1.37502563e+00, -4.09613639e-01, -4.34575468e-01, -7.86593556e-01
             , -5.89816928e-01, -3.36077571e-01, -2.84484476e-01, -4.81034517e-01
             , -7.57222116e-01]
    , [-5.57324827e-01, -1.22515716e-01, -5.51174343e-01, -6.11912549e-01
             , -6.16714060e-01, -2.70383107e-03, -4.76503015e-01, 1.03003785e-01
             , -1.34434080e+00, -2.79533584e-02, -1.38394654e+00, 1.70910671e-01
             , -4.75609690e-01, -1.61830544e-01, -1.67978656e+00, -5.46433508e-01
             , -2.29841366e-01, 5.10910869e-01, -7.77774811e-01, -3.59769374e-01
             , 1.37791991e+00, -1.16647445e-01, -6.68623447e-01, -3.51334900e-01
             , -6.92811310e-01, -6.98375165e-01, -6.20737672e-01, 2.92856842e-01
             , -7.34719455e-01]
    , [-7.66887069e-01, -4.20953780e-01, -3.44670475e-01, 1.02240220e-01
             , 1.07966341e-01, -2.18715459e-01, -4.74727958e-01, -4.58408207e-01
             , -3.26123953e-01, -3.22828293e-01, -6.06313229e-01, -1.29405841e-01
             , -2.83968896e-01, 6.70182481e-02, -7.15113819e-01, -3.00501108e-01
             , -2.23554552e-01, -1.14122450e-01, -3.60449925e-02, -1.41952395e-01
             , -6.04334295e-01, -4.79309410e-02, 2.71494299e-01, 1.00205235e-01
             , -4.58747655e-01, -1.40194129e-02, -1.45463571e-01, -7.91530460e-02
             , -2.51312077e-01]
    , [-5.63537419e-01, -2.44673371e-01, -3.12439561e-01, -3.75155330e-01
             , -2.79784128e-02, -2.56609730e-03, -3.99986535e-01, -6.74679205e-02
             , -6.71884894e-01, -3.77605200e-01, -7.60121822e-01, -3.66071165e-01
             , -1.96567565e-01, -4.07761395e-01, -8.27344477e-01, -7.18855634e-02
             , 1.57230515e-02, 1.50625646e-01, -1.58088282e-01, -1.16952591e-01
             , -4.77003425e-01, -4.42438483e-01, 4.16669756e-01, 7.30176046e-02
             , -5.95973909e-01, -1.63331106e-01, 7.33748730e-03, -2.51585059e-02
             , -5.13621151e-01]
    , [-1.25077915e+00, -3.99945050e-01, -3.61354589e-01, -2.73752987e-01
             , -2.29901940e-01, 8.04984197e-02, -4.10002381e-01, -1.26524605e-02
             , -6.25231981e-01, -6.96334183e-01, -6.47873521e-01, -3.19309175e-01
             , -3.24268460e-01, -5.52571476e-01, -1.02379310e+00, 6.64557889e-03
             , -4.42463517e-01, -1.78335026e-01, -8.54247034e-01, -2.90790766e-01
             , -1.15229261e+00, -2.49721743e-02, -2.04042271e-01, -1.34941185e+00
             , 1.41215777e+00, -4.79302973e-01, -8.70992661e-01, -4.74147201e-01
             , -7.13985324e-01]
    , [-3.19320112e-01, -2.04420626e-01, -3.94440114e-01, 2.45988101e-01
             , -4.05666560e-01, 1.36256471e-01, -6.30588174e-01, -1.31429374e-01
             , -5.11269748e-01, -5.80744088e-01, -6.99872673e-01, 9.36566889e-02
             , -8.36396068e-02, -6.99881837e-02, -4.66677576e-01, -1.61819518e-01
             , -2.62958050e-01, -3.30444157e-01, -2.36155391e-02, -1.27307385e-01
             , -5.57465017e-01, 5.21162003e-02, -4.84614640e-01, -4.78237689e-01
             , 8.18428576e-01, -4.21587229e-01, -2.42237791e-01, -2.93369085e-01
             , -2.66904324e-01]
    , [-9.14176464e-01, -1.60858408e-01, -3.58306676e-01, -1.79488972e-01
             , -4.74496573e-01, -3.90564919e-01, -6.06333852e-01, -3.24983656e-01
             , -1.31723392e+00, -5.42401075e-01, -7.98361361e-01, -4.95272815e-01
             , -4.81144100e-01, -7.17469931e-01, -1.04370153e+00, 4.20563854e-02
             , -2.16611847e-01, -6.89381421e-01, -9.62729275e-01, -7.53989577e-01
             , -8.62055898e-01, -3.89605343e-01, -4.03751433e-01, -2.10293442e-01
             , -9.13593054e-01, -1.15166545e+00, 1.20055699e+00, -5.79535723e-01
             , -1.13069689e+00]
    , [-9.08714160e-02, -2.22046137e-01, -2.62536556e-01, -5.65682026e-03
             , 1.18088506e-01, -9.25026610e-02, -2.65268952e-01, -2.39905134e-01
             , -3.12527984e-01, -5.48494101e-01, -6.14037275e-01, -5.29450119e-01
             , -1.54876346e-02, -2.56357908e-01, -2.55746931e-01, -4.02773440e-01
             , 5.02773039e-02, -3.21960479e-01, -3.33626270e-02, -3.11879069e-01
             , -1.99949592e-01, -4.14134443e-01, -2.01632664e-01, 3.87555122e-01
             , -4.50599879e-01, 3.41840744e-01, 7.24646568e-01, -1.00856967e-01
             , -4.66550052e-01]
    , [-1.55576086e+00, -3.45158041e-01, -3.55792433e-01, -5.59405804e-01
             , -5.16703665e-01, -5.85383058e-01, -5.88217854e-01, -5.36641598e-01
             , -8.78167748e-01, -7.95686543e-01, -9.97305512e-01, -5.85142016e-01
             , -6.38017416e-01, -5.52475214e-01, -1.02823281e+00, -3.08248192e-01
             , -1.20250747e-01, -2.77728975e-01, -6.56644344e-01, -6.93049312e-01
             , -1.35883391e+00, -5.34212828e-01, -4.96678412e-01, -2.28569582e-01
             , -5.83031595e-01, -1.00276279e+00, -7.59981155e-01, -1.02495694e+00
             , 9.01661515e-01]
    , [-1.00765479e+00, -6.09230220e-01, -1.63053572e-01, 1.56491231e-02
             , -4.53906149e-01, 1.59242633e-03, -6.65593266e-01, -3.96686256e-01
             , -9.24402475e-01, -4.14897710e-01, -6.65965497e-01, -4.08155918e-01
             , -4.17189330e-01, -3.96052420e-01, -8.68510127e-01, -4.62063700e-01
             , -5.72978139e-01, -3.69847000e-01, -4.44568962e-01, -2.37310857e-01
             , -1.11639643e+00, -1.47809371e-01, -7.30329752e-01, 1.39481470e-01
             , -4.67131793e-01, -5.48993468e-01, -6.57683969e-01, -3.16504478e-01
             , 1.29379666e+00]]

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

    def cut_txt(self,text):

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


            #token_ids = tokenizer.tokens_to_ids(tokens)
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

            datas.extend([(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l,mapping[w[0]][0] + cut_start_index,mapping[w[-1]][-1] + 1 + cut_start_index)
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
        print(R)
        #R = sorted(R,key=lambda x : x[2])
        # T = sorted(T, key=lambda x: x[2])
        #print(R)
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


# model.load_weights('checkpoint/ner_best_model.weights')
# f1, precision, recall = evaluate(valid_data)
# print(f1, precision, recall)

def test():
    """评测函数
    """
    files = glob.glob('data/test/*')
    for file_name in files:

        text_id = os.path.basename(file_name).replace('.txt', '')
        with open(file_name,encoding='utf-8') as test_f,open('data/test_an/' + text_id + '.ann', mode='a', encoding='utf-8') as an_f:

            text = test_f.read()

            datas = NER.recognize(text)
            for i,data in enumerate(datas):

                an_f.write(
                    'T' + str(i) + '\t' + data[1] + ' ' + str(data[2]) + ' ' + str(data[3]) + '\t' + data[0])

                if i < len(datas) - 1:
                    an_f.write('\n')


model.load_weights('checkpoint/9_ner_best_model.weights')
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
