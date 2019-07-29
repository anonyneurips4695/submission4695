# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import random
import math


class MaskRecommendationDataGenerator(object):
    def __init__(self, base_dir, batch_size, max_seq_length, seq_len_rate=0.7, item_size=7000, masked_lm_prob=0.15, mask_rate=1.0):
        self.train_file = os.path.join(base_dir, 'training_aggregation.txt')
        self.valid_file = os.path.join(base_dir, 'validation_aggregation.txt')
        self.train_valid_file = os.path.join(base_dir, 'train_validation_aggregation.txt')
        self.test_file = os.path.join(base_dir, 'test_aggregation.txt')
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.seq_len_rate = seq_len_rate
        self.item_size = item_size
        self.masked_lm_prob = masked_lm_prob
        self.masked_len = int(round(max_seq_length * masked_lm_prob))
        self.mask_rate = mask_rate

        print(self.train_file, self.valid_file, self.train_valid_file, self.test_file)

        self.train_set, self.train_lens = self._read_train_set(self.train_file)
        self.valid_users, self.valid_items, self.valid_masks, self.valid_labels, self.train_its = self._read_valid_set(self.train_file, self.valid_file)
        self.test_users, self.test_items, self.test_masks, self.test_labels, self.train_val_its = self._read_test_set(self.train_valid_file, self.test_file)

        self.valid_step = int(math.ceil(len(self.valid_users) / batch_size))
        self.valid_index = 0
        self.test_step = int(math.ceil(len(self.test_users) / batch_size))
        self.test_index = 0

    def _read_train_set(self, train_file):
        res = []
        lens = []
        with open(train_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                user = int(line[0])
                items = [int(item) + 3 for item in line[1:]]
                res.append([user, items])
                lens.append(len(items))
        sum_len = sum(lens)
        lens = [float(ll) / float(sum_len) for ll in lens]
        return res, lens

    def _read_valid_set(self, train_file, valid_file):
        valid_users = []
        valid_items = []
        valid_masks = []
        valid_labels = []
        train_its = []
        with open(train_file, 'r') as train_f:
            train_lines = train_f.readlines()
            with open(valid_file, 'r') as val_f:
                for val_line in val_f.readlines():
                    val_line = val_line.strip().split(' ')
                    user = int(val_line[0])
                    train_line = train_lines[user].strip().split(' ')[1:]
                    val_items = [int(item) + 3 for item in val_line[1:]]
                    train_items = [int(item) + 3 for item in train_line[-self.max_seq_length:]]
                    train_masks = [1] * len(train_items)
                    if len(train_items) < self.max_seq_length:
                        train_masks.extend([0] * (self.max_seq_length - len(train_items)))
                        train_items.extend([0] * (self.max_seq_length - len(train_items)))
                    assert len(train_items) == self.max_seq_length
                    assert len(train_masks) == self.max_seq_length
                    assert train_items.count(0) == train_masks.count(0)
                    valid_users.append(user)
                    valid_items.append(train_items)
                    valid_masks.append(train_masks)
                    valid_labels.append(val_items)
        with open(train_file, 'r') as train_f:
            for train_line in train_f.readlines():
                train_items = set([int(item) + 3 for item in train_line.strip().split(' ')[1:]])
                train_its.append(train_items)
        return np.asarray(valid_users, np.int), np.asarray(valid_items, np.int), np.asarray(valid_masks, np.int), valid_labels, train_its

    def _read_test_set(self, train_valid_file, test_file):
        test_users = []
        test_its = []
        test_masks = []
        test_labels = []
        train_val_its = []
        with open(train_valid_file, 'r') as train_val_f:
            train_val_lines = train_val_f.readlines()
            with open(test_file, 'r') as test_f:
                for test_line in test_f.readlines():
                    test_line = test_line.strip().split(' ')
                    user = int(test_line[0])
                    train_val_line = train_val_lines[user].strip().split(' ')[1:]
                    test_items = [int(item) + 3 for item in test_line[1:]]
                    train_val_items = [int(item) + 3 for item in train_val_line[-self.max_seq_length:]]
                    train_val_masks = [1] * len(train_val_items)
                    if len(train_val_items) < self.max_seq_length:
                        train_val_masks.extend([0] * (self.max_seq_length - len(train_val_items)))
                        train_val_items.extend([0] * (self.max_seq_length - len(train_val_items)))
                    assert len(train_val_items) == self.max_seq_length
                    assert len(train_val_masks) == self.max_seq_length
                    assert train_val_items.count(0) == train_val_items.count(0)
                    test_users.append(user)
                    test_its.append(train_val_items)
                    test_masks.append(train_val_masks)
                    test_labels.append(test_items)
        with open(train_valid_file, 'r') as train_val_f:
            for train_val_line in train_val_f.readlines():
                train_val_items = set([int(item) + 3 for item in train_val_line.strip().split(' ')[1:]])
                train_val_its.append(train_val_items)
        return np.asarray(test_users, np.int), np.asarray(test_its, np.int), np.asarray(test_masks, np.int), test_labels, train_val_its

    def get_train_batch(self):

        def neg_sample(neg_user, item_size):
            neg = np.random.randint(3, item_size)
            while neg in self.train_its[neg_user]:
                neg = np.random.randint(3, item_size)
            return neg

        batch_user = []
        batch_items = []
        batch_masks = []
        batch_labels = []
        batch_neg = []
        sample_lines = np.random.choice(len(self.train_set), self.batch_size, p=self.train_lens)

        for line in sample_lines:
            user, items = self.train_set[line]
            seq_len = np.random.randint(int(self.max_seq_length * self.seq_len_rate), self.max_seq_length + 1)
            start = np.random.randint(0, max(1, len(items) - seq_len + 1))
            batch_user.append(user)
            if (start + seq_len) < len(items):
                its = items[start: start + seq_len]
                masks = [1] * len(its)
                labels = its[1:] + [items[start + seq_len]]
                negs = [neg_sample(user, self.item_size) for _ in range(len(labels))]
            else:
                its = items[start: -1]
                masks = [1] * len(its)
                labels = its[1:] + [items[-1]]
                negs = [neg_sample(user, self.item_size) for _ in range(len(labels))]

            if np.random.random_sample() < self.mask_rate:
                masked_lm_positions = np.random.choice(range(len(its)), int(round(len(its) * self.masked_lm_prob)), replace=False).tolist()
                mask_lm_probs = np.random.choice(range(3), len(masked_lm_positions), replace=True, p=[0.8, 0.1, 0.1])
                masked_lm_ids = []
                for pos, prob in zip(masked_lm_positions, mask_lm_probs):
                    masked_lm_ids.append(its[pos])
                    if prob == 0:
                        its[pos] = 2
                    elif prob == 1:
                        its[pos] = np.random.randint(3, self.item_size)
                    else:
                        continue

            if len(its) < self.max_seq_length:
                masks.extend([0] * (self.max_seq_length - len(its)))
                labels.extend([0] * (self.max_seq_length - len(its)))
                negs.extend([0] * (self.max_seq_length - len(its)))
                its.extend([0] * (self.max_seq_length - len(its)))

            # assert len(its) == self.max_seq_length
            # assert len(masks) == self.max_seq_length
            # assert len(negs) == self.max_seq_length
            # assert len(labels) == self.max_seq_length

            batch_items.append(its)
            batch_masks.append(masks)
            batch_labels.append(labels)
            batch_neg.append(negs)

        return np.asarray(batch_user, np.int), np.asarray(batch_items, np.int), np.asarray(batch_masks, np.int), np.asarray(batch_labels, np.int), np.asarray(batch_neg, np.int)

    def get_valid_batch(self):
        if self.valid_index + self.batch_size > len(self.valid_users):
            batch_user = np.full([self.batch_size], -1, dtype=np.int)
            batch_items = np.zeros([self.batch_size, self.max_seq_length], dtype=np.int)
            batch_masks = np.zeros([self.batch_size, self.max_seq_length], dtype=np.int)
            batch_labels = self.valid_labels[self.valid_index:]
            if self.valid_index < len(self.valid_users):
                batch_user[0: len(self.valid_users) - self.valid_index] = self.valid_users[self.valid_index:]
                batch_items[0: len(self.valid_users) - self.valid_index] = self.valid_items[self.valid_index:]
                batch_masks[0: len(self.valid_users) - self.valid_index] = self.valid_masks[self.valid_index:]
                batch_labels = batch_labels + [[0]] * (self.batch_size - len(self.valid_users) + self.valid_index)
        else:
            batch_user = self.valid_users[self.valid_index: self.valid_index + self.batch_size]
            batch_items = self.valid_items[self.valid_index: self.valid_index + self.batch_size]
            batch_masks = self.valid_masks[self.valid_index: self.valid_index + self.batch_size]
            batch_labels = self.valid_labels[self.valid_index: self.valid_index + self.batch_size]

        self.valid_index += self.batch_size
        return np.asarray(batch_user, np.int), np.asarray(batch_items, np.int), np.asarray(batch_masks, np.int), np.zeros([self.batch_size, self.max_seq_length], np.int), batch_labels, np.zeros([self.batch_size, self.max_seq_length], np.int)

    def get_test_batch(self):
        if self.test_index + self.batch_size > len(self.test_users):
            batch_user = np.full([self.batch_size], -1, dtype=np.int)
            batch_items = np.zeros([self.batch_size, self.max_seq_length], dtype=np.int)
            batch_masks = np.zeros([self.batch_size, self.max_seq_length], dtype=np.int)
            batch_labels = self.test_labels[self.test_index:]
            if self.test_index < len(self.test_users):
                batch_user[0: len(self.test_users) - self.test_index] = self.test_users[self.test_index:]
                batch_items[0: len(self.test_users) - self.test_index] = self.test_items[self.test_index:]
                batch_masks[0: len(self.test_users) - self.test_index] = self.test_masks[self.test_index:]
                batch_labels = batch_labels + [[0]] * (self.batch_size - len(self.test_users) + self.test_index)
        else:
            batch_user = self.test_users[self.test_index: self.test_index + self.batch_size]
            batch_items = self.test_items[self.test_index: self.test_index + self.batch_size]
            batch_masks = self.test_masks[self.test_index: self.test_index + self.batch_size]
            batch_labels = self.test_labels[self.test_index: self.test_index + self.batch_size]

        self.test_index += self.batch_size
        return np.asarray(batch_user, np.int), np.asarray(batch_items, np.int), np.asarray(batch_masks, np.int), np.zeros([self.batch_size, self.max_seq_length], np.int), batch_labels, np.zeros([self.batch_size, self.max_seq_length], np.int)


if __name__ == "__main__":
    # generator = RecommendationDataGenerator('data/Books', 10, 128)
    # for _ in range(10000):
    #     a, b, c, d = generator.get_train_batch()
    import time

    cur = time.time()
    generator = MaskRecommendationDataGenerator('data/LastFM', 16, 128)
    for _ in range(1000):
        a, b, c, d, e = generator.get_train_batch()
        # print(a)
        # print(b)
        # print(c)
        # print(d)
        # print(e)
    print(time.time() - cur)
