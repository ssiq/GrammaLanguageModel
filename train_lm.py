import numpy as np
import tensorflow as tf
import pandas as pd

from collections import OrderedDict
import os
import functools

from embedding.wordembedding import load_vocabulary
from common import util, tf_util


# def register_predict(model, raw_validation_data, validation_data, basename, train_metrics):
#     validation_register_path = os.path.join("dataset", "error")
#     util.make_dir(validation_register_path)
#     pp = [model.predict(*t[:-1]) for t in util.batch_holder(*validation_data, epoches=1, is_shuffle=False)()]
#     metrics = [model.no_reduced_metrics(*t) for t in util.batch_holder(*validation_data, epoches=1, is_shuffle=False)()]
#     for m in metrics:
#         print(m.shape)
#     pp = np.concatenate(pp, axis=0)
#     metrics = np.concatenate(metrics, axis=0)[:, 0]
#     metrics = list(metrics)
#     pp = list(pp)
#     filer_good = lambda x: filter(lambda t: t[1] > train_metrics, x)
#     e_df = OrderedDict({"raw_feature": list(filer_good(zip(raw_validation_data[1], metrics)))})
#     e_df = OrderedDict({**e_df, **{"feature_{}".format(i): list(filer_good(zip(d, metrics))) for i, d in
#                                    enumerate(validation_data[:-1])}})
#     e_df = OrderedDict({**e_df, **{"result": list(filer_good(zip(validation_data[-1], metrics))),
#                                    "predict": list(filer_good(zip(pp, metrics)))}})
#     df = pd.DataFrame(e_df)
#     df.to_csv(os.path.join(validation_register_path, "{}.csv".format(basename)), index=False)

def train(model_fn,
          train_data,
          validation_data,
          model_param,
          base_name,
          create_word_embedding_layer_func,
          batch_size=32):
    load_pretrained_model = True
    continue_train = False
    only_use_basename = True
    # train_data, validation_data = util.train_test_split(train_data, 1000)
    print("length of train_data:{}".format(len(train_data[0])))
    print("length of validation_data:{}".format(len(validation_data[0])))
    # train_data, test_data = util.train_test_split(train_data, 1000)
    m = model_fn(create_word_embedding_layer_func, **model_param)
    if not only_use_basename:
        checkpoint_path = os.path.join('checkpoints','{}_{}'.format(base_name, util.format_dict_to_string(model_param)), '')
    else:
        checkpoint_path = os.path.join('checkpoints','{}'.format(base_name), '')
    print("checkpoint_path:{}".format(checkpoint_path))
    if load_pretrained_model:
        tf_util.load_check_point(checkpoint_path)
        print("global_step:{}".format(m.global_step))
        print("load_check_point")
        if not continue_train:
            return m
    print_skip_step = 100
    skip_steps = 100
    save_steps = 100
    train_loss = []
    train_metrics = []
    validation_losses = []
    validatopm_metrics = []
    sess = tf_util.get_session()
    train_writer = tf.summary.FileWriter(
        logdir='./graphs/{}/{}'.format(base_name,
                                       (util.format_dict_to_string(model_param) if not only_use_basename else "tf_board")
                                                                                                             + "_train"),
        graph=sess.graph)
    validation_writer = tf.summary.FileWriter(
        logdir='./graphs/{}/{}'.format(base_name,
                                       (util.format_dict_to_string(model_param) if not only_use_basename else "tf_board")
                                                                                                             + "_val"),
        graph=sess.graph)

    saver = tf.train.Saver()
    util.make_dir(checkpoint_path)
    validation_data_itr = util.batch_holder(*validation_data, epoches=None)()
    saved_model_list = []
    last_train_metrics = None
    for i, d in enumerate(util.batch_holder(*train_data, epoches=3, batch_size=batch_size)()):
        # for t in d:
        #     print(np.array(t).shape)
        loss, metrics, _ = m.train(*d)
        train_loss.append(loss)
        train_metrics.append(metrics)
        val_data = next(validation_data_itr)
        loss, metrics = m.metrics(*val_data)
        validation_losses.append(loss)
        validatopm_metrics.append(metrics)
        if i % print_skip_step == 0:
            last_train_metrics = np.mean(train_metrics)
            print("iteration {}: train loss is {}, metrics is {} and validation loss is {}, metrics is {}".format(i,
                                                                                                                  np.mean(train_loss),
                                                                                                                  np.mean(train_metrics),
                                                                                                                  np.mean(validation_losses),
                                                                                                                  np.mean(validatopm_metrics)))
            train_loss = []
            train_metrics = []
            validation_losses = []
            validatopm_metrics = []
        if i % save_steps == 0:
            saver.save(sess, checkpoint_path,
                       m.global_step)
        if i % skip_steps == 0:
            train_summary = m.summary(*d)
            train_writer.add_summary(train_summary, global_step=m.global_step)
            validation_summary = m.summary(*val_data)
            validation_writer.add_summary(validation_summary, global_step=m.global_step)
    saver.save(sess,
              checkpoint_path,
               m.global_step)
    if last_train_metrics is None:
        last_train_metrics = np.mean(train_metrics)


    return m, last_train_metrics


if __name__ == '__main__':
    debug = False
    import parameter_config
    util.set_cuda_devices(0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    train_config = parameter_config.config6
    tokenizer_method = train_config["tokenize_name"]
    create_embedding_fn = train_config["create_embedding_fn"]
    batch_size = train_config.get("batch_size", 32)
    with tf.Session(config=config):
        if not debug:
            train_data = data.load_train_data(tokenizer_method)
        else:
            train_data = [t[:100] for t in data.load_train_data(tokenizer_method)]
        word_embedding = create_embedding_fn(train_data[1])
        train_data, raw_validation_data = util.train_test_split(train_data, 0.1)
        train_data = tuple(word_embedding.parse_fn(train_data[1])) + (train_data[2], )
        validation_data = tuple(word_embedding.parse_fn(raw_validation_data[1])) + (raw_validation_data[2], )
        basename = train_config["base_name"]
        m, train_metrics = train(train_config["model_fn"], train_data, validation_data,
                  train_config["model_parameter"],
                  basename,
                  word_embedding.create_embedding_layer_fn,
                  batch_size)
        # register_predict(m, raw_validation_data, validation_data, basename, train_metrics)
        if not debug:
            ids, texts = data.load_test_data(tokenizer_method)
        else:
            ids, texts = [t[:100] for t in data.load_test_data(tokenizer_method)]
        print("parse the test data")
        texts = word_embedding.parse_fn(texts)
        print("predict on the test data")
        def predict(*args):
            try:
                r = m.predict(*args)
            except Exception as e:
                print("some error happened in the batched predict")
                try:
                    r = np.concatenate([m.predict(*[[tt] for tt in t]) for t in zip(*args)], axis=0)
                    print("the error has been recovered")
                except Exception as e:
                    print("the error recovered failed")
                    r = np.zeros((len(args[0]), 6))
            return r
        res = [predict(*t) for t in util.batch_holder(*texts, epoches=1, is_shuffle=False, batch_size=batch_size)()]
        print("merge predicted data")
        res = np.concatenate(res, axis=0)
        res = OrderedDict(**{'id': ids}, **{k: list(res[:, v]) for k, v in data.target_map.items()})

        df = pd.DataFrame(data=res)
        # print(df.head(10))
        df.to_csv("dataset/result/r_{}.csv".format(basename), index=False)