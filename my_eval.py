
from bert_connector import Bert
import base_connector as base
import argparse
from utils import load_GLUE_data
from tqdm import tqdm
import torch
import numpy as np
def parse_template(template, subject_label, object_label, context):
    print("template:", template)
    HYP_SYMBOL = "[H]"
    PREM_SYMBOL = "[P]"
    SEN_SYMBOL = "[S]"
    sentences = subject_label.split("*%*")
    prem = sentences[0]
    hyp = sentences[1]
    sent_tem = template.split(SEN_SYMBOL)
    context_temp = sent_tem[0]
    sentence_temp = sent_tem[1]
    # SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"

    sentence_temp = sentence_temp.replace(HYP_SYMBOL, hyp)
    sentence_temp = sentence_temp.replace(OBJ_SYMBOL, object_label)
    sentence_temp = sentence_temp.replace(PREM_SYMBOL, prem)

    context_temp = context_temp.replace(HYP_SYMBOL, hyp)
    context_temp = context_temp.replace(OBJ_SYMBOL, object_label)
    context_temp = context_temp.replace(PREM_SYMBOL, prem)

    # CONTEXT PROBING
    if context_temp:
        # template = context + ' ' + template
        # print('TEMPLATE:', template)
        return [context_temp, sentence_temp]
    else:
        return [sentence_temp]

def batchify(data, batch_size):
    msg = ""
    list_samples_batches = []
    list_sentences_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    c = 0

    # sort to group togheter sentences with similar length
    for sample in sorted(
        data, key=lambda k: len(" ".join(k["masked_sentences"]).split())
    ):
        # print('CONTEXT:', sample['context'])
        masked_sentences = sample["masked_sentences"]
        # print('MASKED SENT:', masked_sentences)
        current_samples_batch.append(sample)
        current_sentences_batches.append(masked_sentences)
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            current_samples_batch = []
            current_sentences_batches = []
            c = 0

    # last batch
    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)

    return list_samples_batches, list_sentences_batches, msg


def first(facts, args):
    all_samples = []
    for fact in facts:
        (sub, obj) = fact
        sample = {}
        sample["sub_label"] = sub
        sample["obj_label"] = obj
        # sobstitute all sentences with a standard template
        sample["masked_sentences"] = parse_template(
            args.template.strip(), sample["sub_label"].strip(), base.MASK, None
        )
        all_samples.append(sample)
     # create uuid if not present
    i = 0
    for sample in all_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1
    return all_samples




def second(all_samples, args):
    model = Bert(args)
    print("%%%%%%%%%%%%%%%%%%%%%", args.labels)
    samples_batches, sentences_batches, ret_msg = batchify(all_samples, args.batch_size)
    # logger.info("\n" + ret_msg + "\n")
    index_list = None
    all_count = 0
    correct_prediction = 0
    for i in tqdm(range(len(samples_batches))):
        all_count = all_count + 1
        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]
        # print('SENT B:', sentences_b)
        # print("before get batch")
        original_log_probs_list, token_ids_list, masked_indices_list = model.get_batch_generation(
            sentences_b, logger=None
        )
        # print("after get batch")
        filtered_log_probs_list = original_log_probs_list

        label_index_list = []
        for sample in samples_b:
            obj_label_id = model.get_id(sample["obj_label"])

            # MAKE SURE THAT obj_label IS IN VOCABULARIES
            if obj_label_id is None:
                raise ValueError(
                    "object label {} not in model vocabulary".format(
                        sample["obj_label"]
                    )
                )
            elif model.vocab[obj_label_id[0]] != sample["obj_label"]:
                raise ValueError(
                    "object label {} not in model vocabulary".format(
                        sample["obj_label"]
                    )
                )

            label_index_list.append(obj_label_id)


        arguments = [
            {
                "original_log_probs": original_log_probs,
                "filtered_log_probs": filtered_log_probs,
                "token_ids": token_ids,
                "vocab": model.vocab,
                "label_index": label_index[0],
                "masked_indices": masked_indices,
                "interactive": args.interactive,
                "index_list": index_list,
                "sample": sample,
            }
            for original_log_probs, filtered_log_probs, token_ids, masked_indices, label_index ,sample in zip(
                original_log_probs_list,
                filtered_log_probs_list,
                token_ids_list,
                masked_indices_list,
                label_index_list,
                samples_b,
            )
        ]
        # single thread for debug
        for isx,a in enumerate(arguments):
            experiment_result, sample_MRR, sample_P, sample_perplexity, msg = run_thread(a)
            label_index_l = [2000]*len(args.labels)
            for d in experiment_result.get('topk'):
                for ind,l in enumerate(args.labels):
                    if d.get('token_word_form') == l:
                        label_index_l[ind] = d.get('i')
            min_index = label_index_l.index(min(label_index_l))
            obj_index = args.labels.index(samples_b[isx].get('obj_label'))
            if (min_index == obj_index and min_index<2000):
                correct_prediction = correct_prediction + 1
        if (i %1000 == 0):
            print("accuracy:", correct_prediction/all_count)

    print("accuracy:", correct_prediction/all_count)


def run_thread(arguments):

    msg = ""

    # 1. compute the ranking metrics on the filtered log_probs tensor
    sample_MRR, sample_P, experiment_result, return_msg = get_ranking(
        arguments["filtered_log_probs"],
        arguments["masked_indices"],
        arguments["vocab"],
        label_index=arguments["label_index"],
        index_list=arguments["index_list"],
        print_generation=arguments["interactive"],
        topk=1000,
    )
    msg += "\n" + return_msg

    sample_perplexity = 0.0
    return experiment_result, sample_MRR, sample_P, sample_perplexity, msg


def __print_top_k(value_max_probs, index_max_probs, vocab, mask_topk, index_list, max_printouts = 10):
    result = []
    msg = "\n| Top{} predictions\n".format(max_printouts)
    for i in range(mask_topk):
        filtered_idx = index_max_probs[i].item()

        if index_list is not None:
            # the softmax layer has been filtered using the vocab_subset
            # the original idx should be retrieved
            idx = index_list[filtered_idx]
        else:
            idx = filtered_idx

        log_prob = value_max_probs[i].item()
        word_form = vocab[idx]

        if i < max_printouts:
            msg += "{:<8d}{:<20s}{:<12.3f}\n".format(
                i,
                word_form,
                log_prob
            )
        element = {'i' : i, 'token_idx': idx, 'log_prob': log_prob, 'token_word_form': word_form}
        result.append(element)
    return result, msg


def get_ranking(log_probs, masked_indices, vocab, label_index = None, index_list = None, topk = 1, P_AT = 10, print_generation=True):

    experiment_result = {}

    # score only first mask
    masked_indices = masked_indices[:1]

    masked_index = masked_indices[0]
    log_probs = log_probs[masked_index]

    value_max_probs, index_max_probs = torch.topk(input=log_probs,k=topk,dim=0)
    index_max_probs = index_max_probs.numpy().astype(int)
    value_max_probs = value_max_probs.detach().numpy()

    result_masked_topk, return_msg = __print_top_k(value_max_probs, index_max_probs, vocab, topk, index_list)
    experiment_result['topk'] = result_masked_topk

    if print_generation:
        print(return_msg)

    MRR = 0.
    P_AT_X = 0.
    P_AT_1 = 0.
    PERPLEXITY = None

    if label_index is not None:

        # check if the labe_index should be converted to the vocab subset
        if index_list is not None:
            label_index = index_list.index(label_index)

        query = torch.full(value_max_probs.shape, label_index, dtype=torch.long).numpy().astype(int)
        ranking_position = (index_max_probs==query).nonzero()

        # LABEL PERPLEXITY
        tokens = torch.from_numpy(np.asarray(label_index))
        label_perplexity = log_probs.gather(
            dim=0,
            index=tokens,
        )
        PERPLEXITY = label_perplexity.item()

        if len(ranking_position) >0 and ranking_position[0].shape[0] != 0:
            rank = ranking_position[0][0] + 1

            # print("rank: {}".format(rank))

            if rank >= 0:
                MRR = (1/rank)
            if rank >= 0 and rank <= P_AT:
                P_AT_X = 1.
            if rank == 1:
                P_AT_1 = 1
    experiment_result["MRR"] = MRR
    experiment_result["P_AT_X"] = P_AT_X
    experiment_result["P_AT_1"] = P_AT_1
    experiment_result["PERPLEXITY"] = PERPLEXITY
    #
    # print("MRR: {}".format(experiment_result["MRR"]))
    # print("P_AT_X: {}".format(experiment_result["P_AT_X"]))
    # print("P_AT_1: {}".format(experiment_result["P_AT_1"]))
    # print("PERPLEXITY: {}".format(experiment_result["PERPLEXITY"]))

    return MRR, P_AT_X, experiment_result, return_msg

def main():
    # train_data = load_TREx_data(args, train_file)
    #TODO make RTE as input
    #load_GLUE_data(args, filename, is_train, glue_name, ent_word, cont_word, sentence_size, down_sample = False):
    #(args.data_dir, train_file , True, glue_name = args.dataset , down_sample = False, ent_word = args.ent_word, cont_word = args.cont_word, sentence_size = args.sentence_size)
    facts = load_GLUE_data("/home/yrazeghi/comsence/data/glue_data/", "" , True, "MNLI", down_sample=False,  ent_word="and", cont_word="but", sentence_size=100)
    params = {
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-cased",
        "bert_model_dir": None, #"pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
        "bert_vocab_name" : "vocab.txt",
        "template":"[P] [S] Originally offering quite [Y] [H]",
        "batch_size":1,
        "interactive": False,
        "labels": ['and', 'but']
    }
    args = argparse.Namespace(**params)
    all_samples = first(facts, args)
    second(all_samples, args)


if __name__ == "__main__":
    main()
