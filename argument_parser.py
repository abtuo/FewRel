import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train", default="train_wiki", help="train file")
parser.add_argument("--val", default="val_wiki", help="val file")
parser.add_argument("--test", default="val_wiki", help="test file")
parser.add_argument("--adv", default=None, help="adv file")
parser.add_argument("--trainN", default=5, type=int, help="N in train")
parser.add_argument("--N", default=5, type=int, help="N way")
parser.add_argument("--K", default=5, type=int, help="K shot")
parser.add_argument("--Q", default=5, type=int, help="Num of query per class")
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument(
    "--train_iter", default=30000, type=int, help="num of iters in training"
)
parser.add_argument(
    "--val_iter", default=1000, type=int, help="num of iters in validation"
)
parser.add_argument(
    "--test_iter", default=10000, type=int, help="num of iters in testing"
)
parser.add_argument(
    "--val_step", default=2000, type=int, help="val after training how many iters"
)
parser.add_argument("--model", default="proto", help="model name")
parser.add_argument(
    "--encoder", default="weighted", help="encoder: cnn or bert or roberta"
)
parser.add_argument("--max_length", default=128, type=int, help="max length")
parser.add_argument("--lr", default=-1, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")
parser.add_argument("--dropout", default=0.0, type=float, help="dropout rate")
parser.add_argument("--na_rate", default=0, type=int, help="NA rate (NA = Q * na_rate)")
parser.add_argument(
    "--grad_iter",
    default=1,
    type=int,
    help="accumulate gradient every x iterations",
)
parser.add_argument("--optim", default="sgd", help="sgd / adam / adamw")
parser.add_argument("--hidden_size", default=230, type=int, help="hidden size")
parser.add_argument("--load_ckpt", default=None, help="load ckpt")
parser.add_argument("--save_ckpt", default=None, help="save ckpt")
parser.add_argument("--fp16", action="store_true", help="use nvidia apex fp16")
parser.add_argument("--only_test", action="store_true", help="only test")
parser.add_argument("--ckpt_name", type=str, default="", help="checkpoint name.")

# only for bert / roberta
parser.add_argument("--pair", action="store_true", help="use pair model")
parser.add_argument(
    "--pretrain_ckpt", default=None, help="bert / roberta pre-trained checkpoint"
)
parser.add_argument(
    "--cat_entity_rep",
    action="store_true",
    help="concatenate entity representation as sentence rep",
)

# only for prototypical networks
parser.add_argument(
    "--dot", action="store_true", help="use dot instead of L2 distance for proto"
)

# only for mtb
parser.add_argument(
    "--no_dropout",
    action="store_true",
    help="do not use dropout after BERT (still has dropout in BERT).",
)

# experiment
parser.add_argument("--mask_entity", action="store_true", help="mask entity names")
parser.add_argument(
    "--use_sgd_for_bert",
    action="store_true",
    help="use SGD instead of AdamW for BERT.",
)

# parser.add_argument("--weighted", action="store_true", help="use bert weighted")
# parser.add_argument("--att", action="store_true", help="use bert weightedATT")

parser.add_argument(
    "--bert_path",
    default="/home/users/atuo/language_models/bert/bert-base-uncased/",
    help="Bert Model path",
)
# /home/users/atuo/language_models/bert/
# /scratch_global/LANGUAGE_MODELS/bert/ --> Bertgamote
