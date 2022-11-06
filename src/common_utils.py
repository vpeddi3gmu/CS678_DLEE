import json
from spacy.tokens import Doc
from collections import defaultdict
import re

PRONOUN_FILE = 'pronoun_list.txt'
pronoun_set = set()
with open(PRONOUN_FILE, 'r') as f:
    for line in f:
        pronoun_set.add(line.strip())


def check_pronoun(text):
    if text.lower() in pronoun_set:
        return True
    else:
        return False


def clean_mention(text):
    '''
    Clean up a mention by removing 'a', 'an', 'the' prefixes.
    '''
    prefixes = ['the ', 'The ', 'an ', 'An ', 'a ', 'A ']
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


def find_head(arg_start, arg_end, doc):
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <= arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head
            break
        else:
            cur_i = doc[cur_i].head.i

    arg_head = cur_i

    return (arg_head, arg_head)


def extract_args_from_template(ontology_dict, evt_type, pred_template,):
    # extract argument text
    template = ontology_dict[evt_type]['template']
    template_words = template.strip().split()
    predicted_words = pred_template.strip().split()
    # each argname may have multiple participants
    predicted_args = defaultdict(list)
    t_ptr = 0
    p_ptr = 0
    # evt_type = ex['event']['event_type']
    while t_ptr < len(template_words) and p_ptr < len(predicted_words):
        if re.match(r'<(arg\d+)>', template_words[t_ptr]):
            m = re.match(r'<(arg\d+)>', template_words[t_ptr])
            arg_num = m.group(1)
            try:
                arg_name = ontology_dict[evt_type][arg_num]
            except KeyError:
                exit()

            if predicted_words[p_ptr] == '<arg>':
                # missing argument
                p_ptr += 1
                t_ptr += 1
            else:
                arg_start = p_ptr
                while (p_ptr < len(predicted_words)) and ((t_ptr == len(template_words)-1) or (predicted_words[p_ptr] != template_words[t_ptr+1])):
                    p_ptr += 1
                arg_text = predicted_words[arg_start:p_ptr]
                predicted_args[arg_name].append(arg_text)
                t_ptr += 1
                # aligned
        else:
            t_ptr += 1
            p_ptr += 1

    return predicted_args


def load_ontology(dataset, ontology_file=None):
    '''
    Read ontology file for event to argument mapping.
    '''
    ontology_dict = {}
    if not ontology_file:  # use the default file path
        if not dataset:
            raise ValueError
        with open('event_role_{}.json'.format(dataset), 'r') as f:
            ontology_dict = json.load(f)
    else:
        with open(ontology_file, 'r') as f:
            ontology_dict = json.load(f)

    for evt_name, evt_dict in ontology_dict.items():
        for i, argname in enumerate(evt_dict['roles']):
            evt_dict['arg{}'.format(i+1)] = argname
            # argname -> role is not a one-to-one mapping
            if argname in evt_dict:
                evt_dict[argname].append('arg{}'.format(i+1))
            else:
                evt_dict[argname] = ['arg{}'.format(i+1)]

    # add mapping between <argx> and the word before it
    for evt_name, evt_dict in ontology_dict.items():
        evt_dict['arg_to_prev'] = {}
        for i, argname in enumerate(evt_dict['roles']):
            template_words = evt_dict['template'].strip().split()
            try:
                idx = template_words.index("<" + evt_dict[argname][0] + ">")
            except ValueError:
                # import ipdb; ipdb.set_trace() # 'ArtifactExistence.DamageDestroyDisableDismantle.Dismantle' '<arg1> dismantled <arg2> using <arg3> instrument in <arg4> place'
                idx = 0
            if idx == 0:
                evt_dict['arg_to_prev'][argname] = "<s>"
            else:
                evt_dict['arg_to_prev'][argname] = template_words[idx-1]

    return ontology_dict


def find_arg_span(arg, context_words, trigger_start, trigger_end, head_only=False, doc=None):
    match = None
    arg_len = len(arg)
    min_dis = len(context_words)  # minimum distance to trigger
    for i, w in enumerate(context_words):
        if context_words[i:i+arg_len] == arg:
            if i < trigger_start:
                dis = abs(trigger_start-i-arg_len)
            else:
                dis = abs(i-trigger_end)
            if dis < min_dis:
                match = (i, i+arg_len-1)
                min_dis = dis

    if match and head_only:
        assert(doc != None)
        match = find_head(match[0], match[1], doc)
    return match


def get_entity_span(ex, entity_id):
    for ent in ex['entity_mentions']:
        if ent['id'] == entity_id:
            return (ent['start'], ent['end'])
