import numpy as np
from collections import defaultdict
from scipy.misc import logsumexp
import datetime
import random


class DataSet:
    def __init__(self, filename):
        self.filename = filename
        self.sentences = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        file = open(filename, encoding='utf-8')
        while True:
            line = file.readline()
            if not line:
                break
            if line == '\n':
                self.sentences.append(sentence)  # [[word1,word2,...],[word1...],[...]]
                self.tags.append(tag)  # [[tag1,tag2,...],[tag1...],[...]]
                sentence = []
                tag = []
            else:
                sentence.append(line.split()[1])  # [word1,word2,...]
                tag.append(line.split()[3])  # [tag1,tag2,...]
                word_num += 1
        self.sentences_num = len(self.sentences)  # 统计句子个数
        self.word_num = word_num  # 统计词语个数

        print('{}:共{}个句子,共{}个词。'.format(filename, self.sentences_num, self.word_num))
        file.close()

    def shuffle(self):
        temp = [(s, t) for s, t in zip(self.sentences, self.tags)]
        random.shuffle(temp)
        self.sentences = []
        self.tags = []
        for s, t in temp:
            self.sentences.append(s)
            self.tags.append(t)


class CRF(object):
    def __init__(self, train_file, dev_file, test_file):
        self.train_data = DataSet(train_file)  # 处理训练集文件
        self.dev_data = DataSet(dev_file)  # 处理开发集文件
        self.test_data = DataSet(test_file) # 处理测试集文件
        self.features = {}  # 存放所有特征及其对应编号的字典
        self.index_to_tag = {}  # 存放所有词性及其对应编号的字典
        self.tag_to_index = {}
        self.tag_list = []  # 存放所有词性的列表
        self.weights = []  # 特征权重矩阵
        self.g = {}
        self.BOS = 'BOS'

    def create_bigram_feature(self, pre_tag):
        return ['01:' + pre_tag]

    def create_unigram_feature(self, sentence, position):
        template = []
        cur_word = sentence[position]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if position == 0:
            last_word = '##'
            last_word_last_char = '#'
        else:
            last_word = sentence[position - 1]
            last_word_last_char = sentence[position - 1][-1]

        if position == len(sentence) - 1:
            next_word = '$$'
            next_word_first_char = '$'
        else:
            next_word = sentence[position + 1]
            next_word_first_char = sentence[position + 1][0]

        template.append('02:' + cur_word)
        template.append('03:' + last_word)
        template.append('04:' + next_word)
        template.append('05:' + cur_word + '*' + last_word_last_char)
        template.append('06:' + cur_word + '*' + next_word_first_char)
        template.append('07:' + cur_word_first_char)
        template.append('08:' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + sentence[position][i])
            template.append('10:' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + sentence[position][-1] + '*' + sentence[position][i])
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + sentence[position][i] + '*' + 'consecutive')

        if len(sentence[position]) > 1 and sentence[position][0] == sentence[position][1]:
            template.append('13:' + sentence[position][0] + '*' + 'consecutive')

        if len(sentence[position]) == 1:
            template.append('12:' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + sentence[position][0:i + 1])
            template.append('15:' + sentence[position][-(i + 1)::])
        return template

    def create_feature_template(self, sentence, position, pre_tag):
        template = []
        template.extend(self.create_bigram_feature(pre_tag))
        template.extend(self.create_unigram_feature(sentence, position))
        return template

    def create_feature_space(self):
        for i in range(len(self.train_data.sentences)):
            sentence = self.train_data.sentences[i]
            tags = self.train_data.tags[i]
            for j in range(len(sentence)):
                if j == 0:
                    pre_tag = self.BOS
                else:
                    pre_tag = tags[j - 1]
                template = self.create_feature_template(sentence, j, pre_tag)
                for f in template:  # 对特征进行遍历
                    if f not in self.features:  # 如果特征不在特征字典中，则添加进去
                        self.features[f] = len(self.features)  # 给该特征一个独立的序号标记
                for tag in tags:
                    if tag not in self.tag_list:
                        self.tag_list.append(tag)
        self.tag_list = sorted(self.tag_list)
        self.tag_to_index = {t: i for i, t in enumerate(self.tag_list)}
        self.index_to_tag = {i: t for i, t in enumerate(self.tag_list)}
        self.weights = np.zeros((len(self.features), len(self.tag_list)))
        self.g = defaultdict(float)
        self.bigram_features = [self.create_bigram_feature(prev_tag) for prev_tag in self.tag_list]
        self.bigram_scores = np.array([self.get_score(f) for f in self.bigram_features])

        print("特征的总数是：{}".format(len(self.features)))

    def get_score(self, feature):
        scores = [self.weights[self.features[f]]  # weights是二维的
                  for f in feature if f in self.features]
        return np.sum(scores, axis=0)

    def predict(self, sentence):
        length = len(sentence)
        delta = np.zeros((length, len(self.tag_list)))
        path = np.zeros((length, len(self.tag_list)), dtype=int)
        delta[0] = self.get_score(self.create_feature_template(sentence, 0, self.BOS))
        path[0] = -1

        for i in range(1, length):
            unigram_features = self.create_unigram_feature(sentence, i)
            unigram_scores = self.get_score(unigram_features)
            scores = np.transpose(self.bigram_scores + unigram_scores) + delta[i - 1]
            path[i] = np.argmax(scores, axis=1)
            delta[i] = np.max(scores, axis=1)
        predict_tag_list = []
        tag_index = np.argmax(delta[length - 1])
        predict_tag_list.append(self.index_to_tag[tag_index])
        for i in range(length - 1):
            tag_index = path[length - 1 - i][tag_index]
            predict_tag_list.insert(0, self.index_to_tag[tag_index])
        return predict_tag_list

    def evaluate(self, data):
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total_num += len(tags)
            predict = self.predict(sentence)
            for j in range(len(tags)):
                if tags[j] == predict[j]:
                    correct_num += 1
        return correct_num, total_num, correct_num / total_num

    def forward(self, sentence):
        path_scores = np.zeros((len(sentence), len(self.tag_list)))
        path_scores[0] = self.get_score(self.create_feature_template(sentence, 0, self.BOS))
        for i in range(1, len(sentence)):
            unigram_scores = self.get_score(self.create_unigram_feature(sentence, i))
            scores = (self.bigram_scores + unigram_scores).T + path_scores[i - 1]
            path_scores[i] = logsumexp(scores, axis=1)
        return path_scores

    def backward(self, sentence):
        path_scores = np.zeros((len(sentence), len(self.tag_list)))
        for i in reversed(range(len(sentence) - 1)):
            unigram_scores = self.get_score(self.create_unigram_feature(sentence, i + 1))
            scores = self.bigram_scores + unigram_scores + path_scores[i + 1]
            path_scores[i] = logsumexp(scores, axis=1)
        return path_scores

    def update_gradient(self, sentence, tags):
        for i in range(len(sentence)):
            unigram_feature = self.create_unigram_feature(sentence, i)
            if i == 0:
                bigram_feature = self.create_bigram_feature(self.BOS)
            else:
                bigram_feature = self.bigram_features[self.tag_to_index[tags[i - 1]]]
            cur_tag = tags[i]
            for f in unigram_feature:
                if f in self.features:
                    self.g[(self.features[f], self.tag_to_index[cur_tag])] += 1
            for f in bigram_feature:
                if f in self.features:
                    self.g[(self.features[f], self.tag_to_index[cur_tag])] += 1

        forward_scores = self.forward(sentence)
        backward_scores = self.backward(sentence)
        log_dinominator = logsumexp(forward_scores[-1])  # 得到分母log(Z(S))
        feature = self.create_feature_template(sentence, 0, self.BOS)
        feature_id = (self.features[f] for f in feature if f in self.features)
        p = np.exp(self.get_score(feature) + backward_scores[0] - log_dinominator)

        for id in feature_id:
            self.g[id] -= p

        for i in range(1, len(sentence)):
            unigram_feature = self.create_unigram_feature(sentence, i)
            unigram_feature_id = [self.features[f] for f in unigram_feature if f in self.features]
            scores = self.bigram_scores + self.get_score(unigram_feature)
            probs = np.exp(scores + forward_scores[i - 1][:, None] + backward_scores[i] - log_dinominator)
            # 验证概率和是否为1
            # print(sum(sum(probs)))
            for bigram_feature, p in zip(self.bigram_features, probs):
                bigram_feature_id = [self.features[f] for f in bigram_feature if f in self.features]
                for id in bigram_feature_id + unigram_feature_id:
                    self.g[id] -= p

    def SGD_train(self, iteration, batch_size, shuffle, regularization, step_opt, eta, C, stop_iteration):
        b = 0
        counter = 0
        max_dev_precision = 0
        global_step = 1
        decay_steps = len(self.train_data.sentences) / batch_size
        decay_rate = 0.96
        learn_rate = eta
        max_iterator = 0

        print('eta={}'.format(eta))
        if regularization:
            print('使用正则化   C={}'.format(C))
        if step_opt:
            print('使用步长优化')
        for iter in range(iteration):
            start_time = datetime.datetime.now()
            print('当前迭代次数：{}'.format(iter))
            if shuffle:
                print('正在打乱训练数据...', end='')
                self.train_data.shuffle()
                print('数据打乱完成')
            for i in range(len(self.train_data.sentences)):
                b += 1
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                self.update_gradient(sentence, tags)
                if b == batch_size:
                    if regularization:
                        self.weights *= (1 - C * eta)
                    for id, value in self.g.items():
                        self.weights[id] += value * learn_rate
                    if step_opt:
                        learn_rate = eta * decay_rate ** (global_step / decay_steps)
                    global_step += 1
                    self.g = defaultdict(float)
                    self.bigram_scores = np.array(
                        [self.get_score(bigram_f) for bigram_f in self.bigram_features])
                    b = 0

            if b > 0:
                if regularization:
                    self.weights *= (1 - C * eta)
                for id, value in self.g.items():
                    self.weights[id] += value * learn_rate
                if step_opt:
                    learn_rate = eta * decay_rate ** (global_step / decay_steps)
                global_step += 1
                self.g = defaultdict(float)
                self.bigram_scores = np.array(
                    [self.get_score(bigram_f) for bigram_f in self.bigram_features])
                b = 0

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data)
            print('\t' + 'train准确率：{} / {} = {}'.format(train_correct_num, total_num, train_precision))
            test_correct_num, test_num, test_precision = self.evaluate(self.test_data)
            print('\t' + 'test准确率：{} / {} = {}'.format(test_correct_num, test_num, test_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data)
            print('\t' + 'dev准确率：{} / {} = {}'.format(dev_correct_num, dev_num, dev_precision))

            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iter
                counter = 0
            else:
                counter += 1
            end_time = datetime.datetime.now()
            print("\t迭代执行时间为：" + str((end_time - start_time).seconds) + " s")
            if counter >= stop_iteration:
                break
        print('最优迭代轮次 = {} , 开发集准确率 = {}'.format(max_iterator, max_dev_precision))


if __name__ == '__main__':
    train_data_file = 'data/train.conll'  # 训练集文件
    dev_data_file = 'data/dev.conll'  # 开发集文件
    test_data_file = 'data/test.conll'  # 测试集文件
    iteration = 100  # 最大迭代次数
    batch_size = 1  # 批次大小
    shuffle = False  # 每次迭代是否打乱数据
    regularization = False  # 是否正则化
    step_opt = False  # 是否步长优化,设为true步长会逐渐衰减，否则为初始步长不变
    eta = 0.5  # 初始步长
    C = 0.0001  # 正则化系数,regularization为False时无效
    stop_iteration = 10  # 连续多少次迭代没有提升效果就退出

    total_start_time = datetime.datetime.now()
    lm = CRF(train_data_file, dev_data_file, test_data_file)
    lm.create_feature_space()
    lm.SGD_train(iteration, batch_size, shuffle, regularization, step_opt, eta, C, stop_iteration)
    total_end_time = datetime.datetime.now()
    print("总执行时间为：" + str((total_end_time - total_start_time).seconds) + " s")
