import re
import csv
import nltk
import spacy
import string
import random
import argparse
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk.tokenize.treebank import TreebankWordDetokenizer

punc = string.punctuation
nlp = spacy.load("en_core_web_trf")
tagger = SequenceTagger.load('pos')


def subString(template):
    rule = r'<[^<>]+>'
    slotList = re.findall(rule, template)
    final_ls = []
    for tag in slotList:
        final_ls.append(tag[1:-1])
    return final_ls


def nltk_out(this_list):
    list_pos = nltk.pos_tag(this_list)
    pos_tags = [token[1] for token in list_pos]
    return pos_tags


def flair_out(this_list):
    sentence = Sentence(this_list)
    tagger.predict(sentence)
    str_sent = str(sentence.to_tagged_string())
    pos_tags = subString(str_sent)
    return pos_tags


def spacy_out(this_list):
    this_str = TreebankWordDetokenizer().detokenize(this_list)
    doc = nlp(this_str)
    pos_tags = [token.pos_ for token in doc]
    if len(pos_tags) != len(this_list):
        return []
    return pos_tags


def this_main(tool_name, ontonotes5_test_file_path, follow_up_inputs_csv_file_path, violations_csv_file_path):
    f_violation = open(violations_csv_file_path, 'w', newline='', encoding='utf-8')
    csv_writer_violation = csv.writer(f_violation)
    f_100_source = open(follow_up_inputs_csv_file_path, 'w', newline='', encoding='utf-8')
    csv_writer_100_source = csv.writer(f_100_source)
    csv_writer_violation.writerow(["source_input", "follow_input", "target_output", "follow_output"])
    csv_writer_100_source.writerow(["source_input", "follow_input", "target_output", "follow_output"])
    with open(ontonotes5_test_file_path, "r", encoding="utf-8") as f:
        lines_source = f.readlines()
    lines = []
    for line in lines_source:
        if line == "\n" or line[0] != "#":
            lines.append(line)
    source_input = {}
    this_sent = []
    this_tag = []
    this_num = 0
    for num in range(len(lines)):
        this_line = lines[num]
        if this_line == "\n":
            if this_sent[-1] in ["?", "/?"]:
                this_sent = []
                this_tag = []
                continue
            while ("/." in this_sent):
                this_sent.remove("/.")
            source_input.update({this_num: this_sent})
            this_num += 1
            this_sent = []
            this_tag = []
            continue
        rule = r'([^\s]+)'
        slotList = re.findall(rule, this_line)
        this_sent.append(slotList[3])
        this_tag.append(slotList[4])
    follow_target = {}
    follow_input = {}
    follow_input_source = {}
    random_source_inputs = [i for i in range(len(source_input))]
    random.shuffle(random_source_inputs)
    this_num = 0
    for num in tqdm(range(len(random_source_inputs)), desc='source output'):
        if this_num >= 100:
            break
        this_sample = source_input[random_source_inputs[num]]
        if len(this_sample) == 0:
            continue
        if this_sample[0] in ["Because", "because"]:
            if "," not in this_sample:
                continue
            if this_sample[0] == "Because":
                index_because = this_sample.index("Because")
            elif this_sample[0] == "because":
                index_because = this_sample.index("because")
            if tool_name == "flair":
                list_pos = flair_out(this_sample)
            elif tool_name == "nltk":
                list_pos = nltk_out(this_sample)
            elif tool_name == "spacy":
                list_pos = spacy_out(this_sample)
            index_comma = this_sample.index(",")
            if this_sample[-1] in punc or this_sample[-1] == "/.":
                this_because_clause = this_sample[index_because:index_comma]
                this_because_clause[0] = this_because_clause[0].lower()
                this_because_clause_tag = list_pos[index_because:index_comma]
                this_follow_input = this_sample[index_comma + 1:-1]
                this_follow_input.append(",")
                this_follow_input.extend(this_because_clause)
                this_follow_input.append(".")
                this_follow_input[0] = this_follow_input[0][0].upper() + this_follow_input[0][1:]
                this_follow_target = list_pos[index_comma + 1:-1]
                if tool_name == "spacy":
                    this_follow_target.append("PUNCT")
                else:
                    this_follow_target.append(",")
                this_follow_target.extend(this_because_clause_tag)
                if tool_name == "spacy":
                    this_follow_target.append("PUNCT")
                else:
                    this_follow_target.append(".")
                follow_input.update({this_num: this_follow_input})
                follow_input_source.update({this_num: this_sample})
                follow_target.update({this_num: this_follow_target})
                this_num += 1
            else:
                this_because_clause = this_sample[index_because:index_comma]
                this_because_clause[0] = this_because_clause[0].lower()
                this_because_clause_tag = list_pos[index_because:index_comma]
                this_follow_input = this_sample[index_comma + 1:]
                this_follow_input.append(",")
                this_follow_input.extend(this_because_clause)
                this_follow_input.append(".")
                this_follow_input[0] = this_follow_input[0][0].upper() + this_follow_input[0][1:]
                this_follow_target = list_pos[index_comma + 1:]
                if tool_name == "spacy":
                    this_follow_target.append("PUNCT")
                else:
                    this_follow_target.append(",")
                this_follow_target.extend(this_because_clause_tag)
                if tool_name == "spacy":
                    this_follow_target.append("PUNCT")
                else:
                    this_follow_target.append(".")
                follow_input.update({this_num: this_follow_input})
                follow_input_source.update({this_num: this_sample})
                follow_target.update({this_num: this_follow_target})
                this_num += 1
            continue
        if "because" not in this_sample:
            continue
        if this_sample[0] not in ["I", "i"]:
            this_sample[0] = this_sample[0][0].lower() + this_sample[0][1:]
        index_because = this_sample.index("because")
        if this_sample[index_because - 1] in ["is", "are", "'s", "'re", "'re", "was", "were"]:
            continue
        if tool_name == "flair":
            list_pos = flair_out(this_sample)
        elif tool_name == "nltk":
            list_pos = nltk_out(this_sample)
        elif tool_name == "spacy":
            list_pos = spacy_out(this_sample)
        this_orignial_because = this_sample[index_because:]
        if "," in this_orignial_because:
            this_index_comma = this_orignial_because.index(",")
            this_because_clause = this_sample[index_because:index_because + this_index_comma]
            this_because_clause_tag = list_pos[index_because:index_because + this_index_comma]
            if this_sample[index_because - 1] == ",":
                this_follow_input = this_because_clause[:]
                this_follow_input.append(",")
                this_follow_input.extend(this_sample[:index_because - 1])
                this_follow_input.extend(this_sample[index_because + this_index_comma:])
                this_follow_input[0] = this_follow_input[0][0].upper() + this_follow_input[0][1:]
                this_follow_target = this_because_clause_tag[:]
                if tool_name == "spacy":
                    this_follow_target.append("PUNCT")
                else:
                    this_follow_target.append(",")
                this_follow_target.extend(list_pos[:index_because - 1])
                this_follow_target.extend(list_pos[index_because + this_index_comma:])
                follow_input.update({this_num: this_follow_input})
                follow_input_source.update({this_num: this_sample})
                follow_target.update({this_num: this_follow_target})
                this_num += 1
            else:
                this_follow_input = this_because_clause[:]
                this_follow_input.append(",")
                this_follow_input.extend(this_sample[:index_because])
                this_follow_input.extend(this_sample[index_because + this_index_comma:])
                this_follow_input[0] = this_follow_input[0][0].upper() + this_follow_input[0][1:]
                this_follow_target = this_because_clause_tag[:]
                if tool_name == "spacy":
                    this_follow_target.append("PUNCT")
                else:
                    this_follow_target.append(",")
                this_follow_target.extend(list_pos[:index_because])
                this_follow_target.extend(list_pos[index_because + this_index_comma:])
                follow_input.update({this_num: this_follow_input})
                follow_input_source.update({this_num: this_sample})
                follow_target.update({this_num: this_follow_target})
                this_num += 1
        else:
            if this_sample[-1] in punc or this_sample[-1] == "/.":
                if this_sample[index_because - 1] == ",":
                    this_because_clause = this_sample[index_because:-1]
                    this_because_clause_tag = list_pos[index_because:-1]
                    this_follow_input = this_because_clause[:]
                    this_follow_input.append(",")
                    this_follow_input.extend(this_sample[:index_because - 1])
                    this_follow_input.append(".")
                    this_follow_input[0] = this_follow_input[0][0].upper() + this_follow_input[0][1:]
                    this_follow_target = this_because_clause_tag[:]
                    if tool_name == "spacy":
                        this_follow_target.append("PUNCT")
                    else:
                        this_follow_target.append(",")
                    this_follow_target.extend(list_pos[:index_because - 1])
                    if tool_name == "spacy":
                        this_follow_target.append("PUNCT")
                    else:
                        this_follow_target.append(".")
                    follow_input.update({this_num: this_follow_input})
                    follow_input_source.update({this_num: this_sample})
                    follow_target.update({this_num: this_follow_target})
                    this_num += 1
                else:
                    this_because_clause = this_sample[index_because:-1]
                    this_because_clause_tag = list_pos[index_because:-1]
                    this_follow_input = this_because_clause[:]
                    this_follow_input.append(",")
                    this_follow_input.extend(this_sample[:index_because])
                    this_follow_input.append(".")
                    this_follow_input[0] = this_follow_input[0][0].upper() + this_follow_input[0][1:]
                    this_follow_target = this_because_clause_tag[:]
                    if tool_name == "spacy":
                        this_follow_target.append("PUNCT")
                    else:
                        this_follow_target.append(",")
                    this_follow_target.extend(list_pos[:index_because])
                    if tool_name == "spacy":
                        this_follow_target.append("PUNCT")
                    else:
                        this_follow_target.append(".")
                    follow_input.update({this_num: this_follow_input})
                    follow_input_source.update({this_num: this_sample})
                    follow_target.update({this_num: this_follow_target})
                    this_num += 1
            else:
                if this_sample[index_because - 1] == ",":
                    this_because_clause = this_sample[index_because:-1]
                    this_because_clause_tag = list_pos[index_because:-1]
                    this_follow_input = this_because_clause[:]
                    this_follow_input.append(",")
                    this_follow_input.extend(this_sample[:index_because])
                    this_follow_input.append(".")
                    this_follow_input[0] = this_follow_input[0][0].upper() + this_follow_input[0][1:]
                    this_follow_target = this_because_clause_tag[:]
                    if tool_name == "spacy":
                        this_follow_target.append("PUNCT")
                    else:
                        this_follow_target.append(",")
                    this_follow_target.extend(list_pos[:index_because])
                    if tool_name == "spacy":
                        this_follow_target.append("PUNCT")
                    else:
                        this_follow_target.append(".")
                    follow_input.update({this_num: this_follow_input})
                    follow_input_source.update({this_num: this_sample})
                    follow_target.update({this_num: this_follow_target})
                    this_num += 1
                else:
                    this_because_clause = this_sample[index_because:]
                    this_because_clause_tag = list_pos[index_because:]
                    this_follow_input = this_because_clause[:]
                    this_follow_input.append(",")
                    this_follow_input.extend(this_sample[:index_because])
                    this_follow_input.append(".")
                    this_follow_input[0] = this_follow_input[0][0].upper() + this_follow_input[0][1:]
                    this_follow_target = this_because_clause_tag[:]
                    if tool_name == "spacy":
                        this_follow_target.append("PUNCT")
                    else:
                        this_follow_target.append(",")
                    this_follow_target.extend(list_pos[:index_because])
                    if tool_name == "spacy":
                        this_follow_target.append("PUNCT")
                    else:
                        this_follow_target.append(".")
                    follow_input.update({this_num: this_follow_input})
                    follow_input_source.update({this_num: this_sample})
                    follow_target.update({this_num: this_follow_target})
                    this_num += 1
    for num in tqdm(range(len(follow_input)), desc='follow output'):
        source_input = follow_input_source[num]
        source_input[0] = source_input[0][0].upper() + source_input[0][1:]
        this_sent = follow_input[num]
        if tool_name == "flair":
            list_pos = flair_out(this_sent)
        elif tool_name == "nltk":
            list_pos = nltk_out(this_sent)
        elif tool_name == "spacy":
            list_pos = spacy_out(this_sent)
        this_target_tag = follow_target[num]
        this_follow_output = list_pos
        if len(this_follow_output) != len(this_target_tag):
            this_follow_output = [n for n in this_follow_output if n not in [",", "."]]
            this_target_tag = [n for n in this_target_tag if n not in [",", "."]]
        if len(this_follow_output) != len(this_target_tag):
            continue
        if this_follow_output != this_target_tag:
            csv_writer_violation.writerow([source_input, this_sent, this_target_tag, this_follow_output])
        csv_writer_100_source.writerow([source_input, this_sent, this_target_tag, this_follow_output])


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tool_name",
        default=None,
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--ontonotes5_test_file_path",
        default=None,
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--follow_up_inputs_csv_file_path",
        default=None,
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--violations_csv_file_path",
        default=None,
        type=str,
        required=True,
        help=""
    )
    args = parser.parse_args()
    if args.tool_name in ["flair", "nltk", "spacy"]:
        this_main(args.tool_name, args.ontonotes5_test_file_path, args.follow_up_inputs_csv_file_path, args.violations_csv_file_path)
    else:
        print("please input the right tool_name: flair, nltk, spacy!")


if __name__ == "__main__":
    main()

