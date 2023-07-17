from fewshot_re_kit.data_loader import (
    get_loader,
    get_loader_pair,
    get_loader_unsupervised,
)
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import (
    CNNSentenceEncoder,
    BERTSentenceEncoder,
    BERTPAIRSentenceEncoder,
    RobertaSentenceEncoder,
    RobertaPAIRSentenceEncoder,
    BERTWeighted,
    BERTWeightedATT,
)
import models
from models.proto import Proto
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
from models.siamese import Siamese
from models.pair import Pair
from models.d import Discriminator
from models.mtb import Mtb
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os

from argument_parser import parser


def main():
    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length

    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    if encoder_name == "cnn":
        try:
            glove_mat = np.load("./pretrain/glove/glove_mat.npy")
            glove_word2id = json.load(open("./pretrain/glove/glove_word2id.json"))
        except:
            raise Exception(
                "Cannot find glove files. Run glove/download_glove.sh to download glove files."
            )
        sentence_encoder = CNNSentenceEncoder(glove_mat, glove_word2id, max_length)

    pretrain_ckpt = opt.bert_path or "bert-base-uncased"
    if encoder_name == "bert":
        sentence_encoder = BERTSentenceEncoder(
            pretrain_ckpt,
            max_length,
            cat_entity_rep=opt.cat_entity_rep,
            mask_entity=opt.mask_entity,
        )
    elif encoder_name == "weighted":
        sentence_encoder = BERTWeighted(
            pretrain_ckpt,
            max_length,
        )
    elif encoder_name == "weighted-att":
        sentence_encoder = BERTWeightedATT(
            pretrain_ckpt,
            max_length,
        )
    else:
        raise NotImplementedError

    if opt.pair:
        train_data_loader = get_loader_pair(
            opt.train,
            sentence_encoder,
            N=trainN,
            K=K,
            Q=Q,
            na_rate=opt.na_rate,
            batch_size=batch_size,
            encoder_name=encoder_name,
        )
        val_data_loader = get_loader_pair(
            opt.val,
            sentence_encoder,
            N=N,
            K=K,
            Q=Q,
            na_rate=opt.na_rate,
            batch_size=batch_size,
            encoder_name=encoder_name,
        )
        test_data_loader = get_loader_pair(
            opt.test,
            sentence_encoder,
            N=N,
            K=K,
            Q=Q,
            na_rate=opt.na_rate,
            batch_size=batch_size,
            encoder_name=encoder_name,
        )
    else:
        train_data_loader = get_loader(
            opt.train,
            sentence_encoder,
            N=trainN,
            K=K,
            Q=Q,
            na_rate=opt.na_rate,
            batch_size=batch_size,
        )
        val_data_loader = get_loader(
            opt.val,
            sentence_encoder,
            N=N,
            K=K,
            Q=Q,
            na_rate=opt.na_rate,
            batch_size=batch_size,
        )
        test_data_loader = get_loader(
            opt.test,
            sentence_encoder,
            N=N,
            K=K,
            Q=Q,
            na_rate=opt.na_rate,
            batch_size=batch_size,
        )
        if opt.adv:
            adv_data_loader = get_loader_unsupervised(
                opt.adv,
                sentence_encoder,
                N=trainN,
                K=K,
                Q=Q,
                na_rate=opt.na_rate,
                batch_size=batch_size,
            )

    if opt.optim == "sgd":
        pytorch_optim = optim.SGD
    elif opt.optim == "adam":
        pytorch_optim = optim.Adam
    elif opt.optim == "adamw":
        from transformers import AdamW

        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    if opt.adv:
        d = Discriminator(opt.hidden_size)
        framework = FewShotREFramework(
            train_data_loader,
            val_data_loader,
            test_data_loader,
            adv_data_loader,
            adv=opt.adv,
            d=d,
        )
    else:
        framework = FewShotREFramework(
            train_data_loader, val_data_loader, test_data_loader
        )

    prefix = "-".join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.adv is not None:
        prefix += "-adv_" + opt.adv
    if opt.na_rate != 0:
        prefix += "-na{}".format(opt.na_rate)
    if opt.dot:
        prefix += "-dot"
    if opt.cat_entity_rep:
        prefix += "-catentity"
    if len(opt.ckpt_name) > 0:
        prefix += "-" + opt.ckpt_name

    if model_name == "proto":
        model = Proto(sentence_encoder, dot=opt.dot)
    elif model_name == "gnn":
        model = GNN(sentence_encoder, N, hidden_size=opt.hidden_size)
    elif model_name == "snail":
        model = SNAIL(sentence_encoder, N, K, hidden_size=opt.hidden_size)
    elif model_name == "metanet":
        model = MetaNet(N, K, sentence_encoder.embedding, max_length)
    elif model_name == "siamese":
        model = Siamese(
            sentence_encoder, hidden_size=opt.hidden_size, dropout=opt.dropout
        )
    elif model_name == "pair":
        model = Pair(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == "mtb":
        model = Mtb(sentence_encoder, use_dropout=not opt.no_dropout)
    else:
        raise NotImplementedError

    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    ckpt = "checkpoint/{}.pth.tar".format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name in ["bert", "roberta"]:
            bert_optim = True
        else:
            bert_optim = False

        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1

        opt.train_iter = opt.train_iter * opt.grad_iter
        framework.train(
            model,
            prefix,
            batch_size,
            trainN,
            N,
            K,
            Q,
            pytorch_optim=pytorch_optim,
            load_ckpt=opt.load_ckpt,
            save_ckpt=ckpt,
            na_rate=opt.na_rate,
            val_step=opt.val_step,
            fp16=opt.fp16,
            pair=opt.pair,
            train_iter=opt.train_iter,
            val_iter=opt.val_iter,
            bert_optim=bert_optim,
            learning_rate=opt.lr,
            use_sgd_for_bert=opt.use_sgd_for_bert,
            grad_iter=opt.grad_iter,
        )
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print(
                "Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint."
            )
            ckpt = "none"

    acc = framework.eval(
        model,
        batch_size,
        N,
        K,
        Q,
        opt.test_iter,
        na_rate=opt.na_rate,
        ckpt=ckpt,
        pair=opt.pair,
    )
    print("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()
