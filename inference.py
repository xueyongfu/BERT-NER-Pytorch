import argparse
import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.progressbar import ProgressBar
from tools.common import seed_everything
from tools.common import init_logger, logger

from models.transformers import WEIGHTS_NAME,BertConfig,AlbertConfig
from models.bert_for_ner import BertCrfForNer
from models.albert_for_ner import AlbertCrfForNer
from processors.utils_ner import CNerTokenizer,get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn


global _args
global _model
global _tokenizer

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert':(AlbertConfig,AlbertCrfForNer,CNerTokenizer)
}


def load_and_cache_examples(args, task, tokenizer,lines, data_type='train'):
    if args.local_rank not in [-1, 0] :
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()

    label_list = processor.get_labels()
    examples = processor.get_predict_examples(lines)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=args.train_max_seq_length if data_type=='train' \
                                                           else args.eval_max_seq_length,
                                            cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                            pad_on_left=bool(args.model_type in ['xlnet']),
                                            cls_token = tokenizer.cls_token,
                                            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens,all_label_ids)
    return dataset


def predict(args, model, tokenizer,lines, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name,tokenizer,lines, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1,collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)

    results = []
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None,'input_lens':batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            preds, _ = model.crf._obtain_labels(logits, args.id2label, inputs['input_lens'])
        preds = preds[0][1:-1] # [CLS]XXXX[SEP]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(preds)
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    print(results[:3])


def main():

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task_name", default=None, type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir",default=None,type=str,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",)
    parser.add_argument("--model_type",default=None,type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--model_name_or_path",default=None,type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),)
    parser.add_argument("--output_dir",default=None,type=str,
                        help="The output directory where the model predictions and checkpoints will be written.", )

    # Other parameters
    parser.add_argument('--markup',default='bios',type=str,choices=['bios','bio'])
    parser.add_argument( "--labels",default="",type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",)
    parser.add_argument( "--config_name", default="", type=str,
                         help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",default="",type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--cache_dir",default="",type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=128,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--eval_max_seq_length",default=512,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train",default=None, type=bool,help="Whether to run training.")
    parser.add_argument("--do_eval", default=None, type=bool, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", default=None, type=bool, help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training",action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", default=None, type=bool,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument( "--max_steps", default=-1,type=int,
                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints",action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument('--predict_all_checkpoints',action="store_true",
                        help="Predict all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", default=None,type=bool,
                        help="Overwrite the content of the output directory 将输出目录覆写")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16",action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level",type=str,default="O1",\
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html",)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    # 参数修改
    # 主要必须修改labels, 位置在processors文件夹ner_seq文件中
    args.task_name = 'cner'
    args.model_type = 'bert'
    args.model_name_or_path = '/root/models/chinese/bert/pytorch/bert-base-chinese'
    args.do_train = True
    args.do_eval = True
    args.do_predict = True
    args.do_lower_case = True
    args.data_dir = '/root/A/违法主体识别/train_data'
    args.train_max_seq_length = 150
    args.eval_max_seq_length = 150
    args.per_gpu_train_batch_size = 4
    args.per_gpu_eval_batch_size = 4
    args.learning_rate = 2e-5
    args.num_train_epochs = 5.0
    args.logging_steps = 300
    args.saving_steps = 600
    args.output_dir = './outputs'
    args.overwrite_output_dir = True
    args.seed = 42


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir ,args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}-{}-{}.log'.format(args.model_type, args.task_name,
                                                                  time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank,device,args.n_gpu, bool(args.local_rank != -1),args.fp16,)

    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,cache_dir=args.cache_dir if args.cache_dir else None,)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,cache_dir=args.cache_dir if args.cache_dir else None,
                                        label2id=args.label2id,device=args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)


    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    checkpoint = args.output_dir

    logger.info("Predict the following checkpoints: %s", checkpoint)
    model = model_class.from_pretrained(checkpoint,config=config,label2id=args.label2id, device=args.device)
    model.to(args.device)

    global _args, _model, _tokenizer
    _args, _model, _tokenizer = args, model, tokenizer

main()

def get_entity(lines):
    lines = []
    for _ in range(16):
        line = '浦东新区浦东图书馆（前程路）'
        label = 'O' * len(line)
        lines.append({'words': list(line), 'labels': list(label)})

    results = predict(_args, _model, _tokenizer, lines, prefix='')
    return results





if __name__ == "__main__":
    main()
