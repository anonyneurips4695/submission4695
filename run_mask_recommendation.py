# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import time
import tensorflow as tf
import numpy as np
import math
from create_mask_recommendation_data import MaskRecommendationDataGenerator

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "base_dir", None,
    "Input base dir for files.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("logging_step", 200, "Number of warmup steps.")

flags.DEFINE_integer("valid_step", 1000, "Number of validation steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")

flags.DEFINE_string("gpu", "1", "GPU num.")

flags.DEFINE_bool("use_fix_position", True, "Whether to use fix position embedding")

flags.DEFINE_float("end_learning_rate", 1e-5, "Ending learning rate")

flags.DEFINE_float("weight_decay_rate", 1e-2, "Weight regularization rate")

flags.DEFINE_float("clip_norm", 1.0, "Clip gradient norm")

flags.DEFINE_float("seq_len_rate", 0.7, "Sequence length in training.")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked lm prob")

flags.DEFINE_float("mask_rate", 0.5, "Rate for mask train batch")

flags.DEFINE_integer("mask_step", 0, "Step for mask training")

flags.DEFINE_bool("add_mask", False, "Whether add mask score")


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


class BertMaskModel(object):
    def __init__(self, bert_config, learning_rate, num_train_steps, num_warmup_steps, batch_size, max_seq_length):
        self.train_mode = tf.placeholder(tf.bool)

        self.input_ids = tf.placeholder(tf.int32, [batch_size, max_seq_length])
        self.input_mask = tf.placeholder(tf.int32, [batch_size, max_seq_length])
        self.label_ids = tf.placeholder(tf.int32, [batch_size, max_seq_length])
        self.neg_label_ids = tf.placeholder(tf.int32, [batch_size, max_seq_length])

        self.is_training = tf.placeholder(tf.bool)

        self.total_loss, logits, avg_logits = self.create_model(bert_config, self.input_ids, self.input_mask, self.label_ids, self.neg_label_ids, bert_config.item_size, self.is_training, True)
        self.train_op = optimization.create_optimizer(self.total_loss, learning_rate, num_train_steps, num_warmup_steps, False, FLAGS.weight_decay_rate, FLAGS.end_learning_rate, FLAGS.clip_norm)
        _, self.top_labels = tf.nn.top_k(logits, k=bert_config.item_size)
        _, self.top_labels_2 = tf.nn.top_k(avg_logits, k=bert_config.item_size)

        tf.summary.scalar('loss', self.total_loss)
        self.merged = tf.summary.merge_all()

    def create_model(self, bert_config, input_ids, input_mask, labels, neg_labels, num_labels, is_training, model_large):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=False,
            use_fix_positional_embedding=FLAGS.use_fix_position,
            model_large=model_large,
            diag_mask=True)

        sequence_out = model.get_sequence_output()
        hidden_size = sequence_out.shape[-1].value
        output_layer = tf.layers.dense(sequence_out, hidden_size, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        if FLAGS.add_mask:
            with tf.variable_scope('mask'):
                mask_score = tf.layers.dense(sequence_out, 1, activation=tf.sigmoid, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_layer = output_layer * mask_score

        final_output = output_layer[:, -1, :]
        embedding = model.get_embedding_table()

        with tf.variable_scope("loss"):
            logits_1 = tf.matmul(final_output, embedding, transpose_b=True)
            logits_1 = logits_1 / float(math.sqrt(bert_config.hidden_size))
            avg_output = tf.reduce_mean(output_layer, axis=1)
            logits_2 = tf.matmul(avg_output, embedding, transpose_b=True)
            logits_2 = logits_2 / float(math.sqrt(bert_config.hidden_size))

        labels = tf.reshape(labels, [-1])
        neg_labels = tf.reshape(neg_labels, [-1])
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        is_pad = tf.reshape(tf.to_float(tf.not_equal(labels, 0)), [-1])

        neg_embedding = tf.nn.embedding_lookup(embedding, neg_labels)
        pos_embedding = tf.nn.embedding_lookup(embedding, labels)
        pos_logits = tf.reduce_sum(output_layer * pos_embedding, axis=-1)
        neg_logits = tf.reduce_sum(output_layer * neg_embedding, axis=-1)
        loss = tf.reduce_sum(- tf.log(tf.sigmoid(pos_logits) + 1e-24) * is_pad -
                             tf.log(1 - tf.sigmoid(neg_logits) + 1e-24) * is_pad) / tf.reduce_sum(is_pad)

        return loss, logits_1, logits_2


def get_acc_and_recall(accs, recalls, top_labels, valid_labels, users, its):
    for top_label, valid_label, user in zip(top_labels, valid_labels, users):
        if user == -1:
            continue
        valid_dict = set(valid_label)
        # valid_index, valid_count = np.unique(valid_label, return_counts=True)
        # valid_dict = dict(zip(valid_index, valid_count))
        acc_count, recall_count = 0, 0
        exist = its[user]
        cur_index = 0
        for tt in top_label:
            if cur_index == 20:
                break
            if tt in exist:
                continue
            if tt in valid_dict:
                acc_count += 1
                recall_count += 1
            cur_index += 1
        accs.append(acc_count / cur_index)
        recalls.append(recall_count / len(valid_label))


def decay_generator(total_step):
    step = total_step
    for _ in range(total_step):
        step -= 1
        yield float(step) / total_step
    while True:
        yield 0.0


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    with tf.Session() as sess:
        model = BertMaskModel(
            bert_config=bert_config,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=FLAGS.num_train_steps,
            num_warmup_steps=FLAGS.num_warmup_steps,
            batch_size=FLAGS.batch_size,
            max_seq_length=bert_config.max_seq_length,
        )

        data_gen = MaskRecommendationDataGenerator(FLAGS.base_dir, FLAGS.batch_size, bert_config.max_seq_length, FLAGS.seq_len_rate, bert_config.item_size, FLAGS.masked_lm_prob, FLAGS.mask_rate)
        decay_gen = decay_generator(FLAGS.mask_step)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if FLAGS.init_checkpoint:
            tvars = tf.trainable_variables()
            assignment_map, initialized_variable_names, var_list = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
            restore = tf.train.Saver(var_list)
            restore.restore(sess, FLAGS.init_checkpoint)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        task = FLAGS.base_dir.split('/')[1]
        tf.logging.info(" task = %s, MASK !!!!!!!!!", task)

        saver = tf.train.Saver(max_to_keep=10)
        cur_time = str(time.time())
        train_writer = tf.summary.FileWriter('log/mask/' + task + '/' + str(FLAGS.learning_rate) + '_' + cur_time + '/board/train', sess.graph)
        valid_writer = tf.summary.FileWriter('log/mask/' + task + '/' + str(FLAGS.learning_rate) + '_' + cur_time + '/board/valid')
        test_writer = tf.summary.FileWriter('log/mask/' + task + '/' + str(FLAGS.learning_rate) + '_' + cur_time + '/board/test')

        tf.logging.info(bert_config.to_json_string())
        tf.logging.info("  Running in gpu %s", FLAGS.gpu)
        tf.logging.info("  seq_len_rate = %f, mask_rate = %f, add_mask = %d", FLAGS.seq_len_rate, FLAGS.mask_rate, FLAGS.add_mask)

        tf.logging.info("  learning_rate = %s, use_fix_position_embedding = %d, cur_time = %s", FLAGS.learning_rate, FLAGS.use_fix_position, cur_time)

        tf.logging.info("  model_large = True, end_learning_rate = %s, weight_decay = %s, clip_norm = %s", FLAGS.end_learning_rate, FLAGS.weight_decay_rate, FLAGS.clip_norm)

        tf.logging.info("  mask decay step = %d", FLAGS.mask_step)

        tf.logging.info("  num_train_steps = %d", FLAGS.num_train_steps)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        train_statistics = []

        for step in range(FLAGS.num_train_steps):
            if FLAGS.mask_step != 0:
                data_gen.mask_rate = next(decay_gen)

            train_users, train_items, train_masks, train_labels, train_neg_labels = data_gen.get_train_batch()
            _, summary, loss = sess.run([model.train_op, model.merged, model.total_loss], feed_dict={model.is_training: True,
                                                                                                     model.input_ids: train_items,
                                                                                                     model.input_mask: train_masks,
                                                                                                     model.label_ids: train_labels,
                                                                                                     model.neg_label_ids: train_neg_labels})
            train_writer.add_summary(summary, step)
            train_statistics.append(loss)

            if step and step % FLAGS.logging_step == 0:
                train_statistics = np.mean(np.asarray(train_statistics, np.float))
                tf.logging.info('Step {} Train: loss = {:3.4f}\t '.format(step, train_statistics))
                train_statistics = []

            if step and step % FLAGS.valid_step == 0:
                tf.logging.info("***** Running validation *****")
                accs, recalls = [], []
                accs_2, recalls_2 = [], []
                for _ in range(data_gen.valid_step):
                    valid_users, valid_items, valid_masks, valid_single_labels, valid_labels, valid_neg_labels = data_gen.get_valid_batch()
                    top_labels, top_labels_2 = sess.run([model.top_labels, model.top_labels_2], feed_dict={model.is_training: False,
                                                                                                           model.input_ids: valid_items,
                                                                                                           model.input_mask: valid_masks,
                                                                                                           model.label_ids: valid_single_labels,
                                                                                                           model.neg_label_ids: valid_neg_labels})
                    get_acc_and_recall(accs, recalls, top_labels, valid_labels, valid_users, data_gen.train_its)
                    get_acc_and_recall(accs_2, recalls_2, top_labels_2, valid_labels, valid_users, data_gen.train_its)
                data_gen.valid_index = 0
                accs = np.mean(np.asarray(accs, np.float))
                recalls = np.mean(np.asarray(recalls, np.float))
                accs_2 = np.mean(np.asarray(accs_2, np.float))
                recalls_2 = np.mean(np.asarray(recalls_2, np.float))

                valid_summary = tf.Summary()
                valid_summary.value.add(tag='acc', simple_value=accs)
                valid_summary.value.add(tag='recall', simple_value=recalls)
                valid_summary.value.add(tag='acc2', simple_value=accs_2)
                valid_summary.value.add(tag='recall2', simple_value=recalls_2)
                valid_writer.add_summary(valid_summary, step)

                tf.logging.info('Step {} Valid: accs = {:3.4f} recalls = {:3.4f}'.format(step, accs, recalls))
                tf.logging.info('Step {} Valid: 2accs = {:3.4f} recalls2 = {:3.4f}'.format(step, accs_2, recalls_2))

                if FLAGS.do_predict:
                    tf.logging.info("***** Running prediction *****")
                    accs, recalls = [], []
                    accs_2, recalls_2 = [], []
                    for _ in range(data_gen.test_step):
                        test_users, test_items, test_masks, test_single_labels, test_labels, test_neg_labels = data_gen.get_test_batch()
                        top_labels, top_labels_2 = sess.run([model.top_labels, model.top_labels_2], feed_dict={model.is_training: False,
                                                                                                               model.input_ids: test_items,
                                                                                                               model.input_mask: test_masks,
                                                                                                               model.label_ids: test_single_labels,
                                                                                                               model.neg_label_ids: test_neg_labels})
                        get_acc_and_recall(accs, recalls, top_labels, test_labels, test_users, data_gen.train_val_its)
                        get_acc_and_recall(accs_2, recalls_2, top_labels_2, test_labels, test_users, data_gen.train_val_its)

                    data_gen.test_index = 0
                    accs = np.mean(np.asarray(accs, np.float))
                    recalls = np.mean(np.asarray(recalls, np.float))
                    accs_2 = np.mean(np.asarray(accs_2, np.float))
                    recalls_2 = np.mean(np.asarray(recalls_2, np.float))

                    test_summary = tf.Summary()
                    test_summary.value.add(tag='acc', simple_value=accs)
                    test_summary.value.add(tag='recall', simple_value=recalls)
                    test_summary.value.add(tag='acc2', simple_value=accs_2)
                    test_summary.value.add(tag='recall2', simple_value=recalls_2)
                    test_writer.add_summary(test_summary, step)

                    tf.logging.info('Step {} Test: accs = {:3.4f} recalls = {:3.4f}'.format(step, accs, recalls))
                    tf.logging.info('Step {} Test: 2accs = {:3.4f} recalls2 = {:3.4f}'.format(step, accs_2, recalls_2))

                tf.logging.info("***** Running training *****")
                tf.logging.info("  Batch size = %d", FLAGS.batch_size)

            if step and step % FLAGS.save_checkpoints_steps == 0:
                saver.save(sess, 'log/mask/' + task + '/' + str(FLAGS.learning_rate) + '_' + cur_time + '/tmp/model.ckpt', global_step=step)


if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("base_dir")
    # flags.mark_flag_as_required("init_checkpoint")

    tf.app.run()
