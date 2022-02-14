from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn
from model import BertSeq2Seq


class Example:
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target
 
class InputFeatures:
    def __init__(self, example_id, source_ids, target_ids, 
                                   source_mask, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def read_examples(query_file, question_file):
    examples = []
    with open(query_file, encoding='utf-8') as query_f, 
         open(question_file, encoding='utf-8') as question_f:
         for idx, (query, question) in enumerate(zip(query_f, question_f)):
            examples.append(
                Example(idx=idx,
                        source=question.strip(),
                        target=query.strip()
                       )
                )
    return examples

def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * len(source_tokens)
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        target_tokens = tokenizer.tokenize('None')

        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask
            )
        )
    return features

tokenizer = BertTokenizer.from_pretrained('razent/spbert-mlm-zero')

enc_config = BertConfig.from_pretrained('bert-base-cased')
encoder = BertModel.from_pretrained('bert-base-cased', config=enc_config)

dec_config = BertConfig.from_pretrained('razent/spbert-mlm-zero')
dec_config.is_decoder = True
dec_config.add_cross_attention = True
decoder = BertModel.from_pretrained('razent/spbert-mlm-zero', config=dec_config)

model = BertSeq2Seq(encoder = encoder, decoder = decoder,
                    config=enc_config, beam_size = 10,
                    max_length = 64, sos_id = tokenizer.cls_token_id, 
                    eos_id = tokenizer.sep_token_id)
device = torch.device('cpu')
model.load_state_dict(torch.load('./pytorch_model.bin',map_location=device),
                      strict = False)

model.to(device)
model = torch.nn.DataParallel(model)

files = ['./NLP2SPARQL_datasets/LCQUAD/dev', 
         './NLP2SPARQL_datasets/LCQUAD/test']


for idx, fl in enumerate(files):
    eval_examples = read_examples(fl + '.en', fl + '.sparql')
    eval_features = convert_examples_to_features(eval_examples,
                                                 tokenizer,
                                                 {},
                                                 stage = 'test')
    source_ids = torch.tensor([f.source_ids for f in eval_features],
                              dtype = torch.long)
    source_masks = torch.tensor([f.source_mask for f in eval_features],
                                dtype = torch.long)
    eval_data = TensorDataset(source_ids, source_masks)

    eval_sampler = SequantialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size = 32)

    model.eval()
    p = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_id, source_mask = batch
        with torch.no_grad():
            preds = model(source_ids=source_id, source_mask=source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces = False)
                p.append(text)
    model.train()
    predictions = []
    pred_str = []
    label_str = []

    








