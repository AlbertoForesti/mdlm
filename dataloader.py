import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers
import numpy as np

import utils
import hydra

from itertools import cycle
from datasets import load_dataset

LOGGER = utils.get_logger(__name__)


def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string


def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
      x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x

class IdentityTokenizer(transformers.PreTrainedTokenizer):
  def __init__(
      self,
      vocab_size):
    self._vocab_size = vocab_size
    super().__init__()
  
  def _tokenize(self, data, **kwargs):
    return data
  
  def _convert_token_to_id(self, token):
    return token
  
  def _convert_id_to_token(self, index):
    return index
  
  def convert_tokens_to_string(self, tokens):
    return ''.join(str(tokens))
  
  def get_vocab(self):
    return {str(i): i for i in range(self.vocab_size)}
  
  @property
  def vocab_size(self):
    return self._vocab_size
    


class Text8Tokenizer(transformers.PreTrainedTokenizer):
  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    **kwargs):
    self.characters = list('abcdefghijklmnopqrstuvwxyz ')
    self._vocab_str_to_int = {
      '[CLS]': 0,
      '[SEP]': 1,
      '[BOS]': 2,
      '[EOS]': 3,
      '[MASK]': 4,
      '[PAD]': 5,
      '[RESERVED]': 6,
      '[UNK]': 7,
      ** {ch: i + 8 for i, ch in enumerate(self.characters)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
      response = requests.get(url, stream=True)
      data_list = []

      # Process each line in the response content
      for line in response.iter_lines(decode_unicode=True):
        if line:
          data = json.loads(line)
          data_list.append(data)

      return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset

def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
  """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
  """
  url = 'http://mattmahoney.net/dc/text8.zip'
  if not crop_train:
    cache_dir = f'{cache_dir}/text8'
  else:
    cache_dir = f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    utils.fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    # Check if raw data exists
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      utils.fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not utils.fsspec_exists(
        os.path.join(raw_cache_dir, 'text8.zip')):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL {}.'.format(url))
        with (urllib.request.urlopen(url) as in_stream,
              open(os.path.join(raw_cache_dir, 'text8.zip'),
                   'wb') as out_file):
          shutil.copyfileobj(in_stream, out_file)

      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read(
          'text8').decode('utf-8')

      # Splits taken from D3PM codebase
      splits = {
        'train': rawdata[:90000000],
        'validation': rawdata[90000000: 95000000],
        'test': rawdata[95000000:],
      }

      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()

    # Chunk and save as datasets.DatasetDict
    def chunks(lst, n):
      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
        yield lst[i:i + n]

    dataset_dict = {}
    for k, v in splits.items():
      if k == 'train' and crop_train == True:
        chunk_size = 2 * max_seq_length
      else:
        chunk_size = max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)

  return dataset


def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result

def get_summeval_dataset(dataset_name, tokenizer, wrap, mode, cache_dir,
    field_size_dict, block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False, p_random=0.0):
  assert sum(list(field_size_dict.values())) == block_size, f"Total field size must be {block_size}, instead got {list(field_size_dict.values())} with sum {sum(list(field_size_dict.values()))}"
  field_length_str = '_'.join(
      [f'{k}{v}' for k, v in field_size_dict.items()])
  if wrap:
    filename = f'{dataset_name.replace("/","")}_{mode}_bs{block_size}_{field_length_str}_wrapped.pt'
  else:
    filename = f'{dataset_name.replace("/","")}_{mode}_bs{block_size}_{field_length_str}_unwrapped.pt'
  _path = os.path.join(cache_dir, filename)
  if utils.fsspec_exists(_path) and False:
    LOGGER.info(f'Loading data from: {_path}')
    return torch.load(_path)
  LOGGER.info(f'Generating new data at: {_path}')

  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]
  print(f"EOS is {EOS}, BOS is {BOS}\n************************************")
  def preprocess_and_tokenize(example, field, max_field_length):
    text = example[field]
      
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      try:
        tokens = tokenizer(text,
                         max_length=max_field_length,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True,)
        # print(len(tokens['input_ids']))
      except:
        raise ValueError(f'Error tokenizing: {text}')
    return tokens
  data = load_dataset('json', data_files=dataset_name)['train']
  tokenized_datasets_by_field = {}
  for field, max_field_length in field_size_dict.items():
    preprocess_and_tokenize_field = functools.partial(
      preprocess_and_tokenize, field=field, max_field_length=max_field_length)
    if streaming:
      tokenized_dataset = data.map(
        preprocess_and_tokenize_field,
        batched=True,
        desc='Tokenizing')
    else:
      tokenized_dataset = data.map(
        preprocess_and_tokenize_field,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc='Tokenizing')
    tokenized_datasets_by_field[field] = tokenized_dataset
  
  tokenized_dataset = ConcatenatedDataset(
    tokenized_datasets_by_field,
    p_random=p_random)
  torch.save(tokenized_dataset, _path)
  return tokenized_dataset


def get_dataset(
    dataset_name, tokenizer, wrap, mode, cache_dir,
    block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False, field_size_dict=None, p_random=0.0, seq_to_seq_exp=False):
  
  if 'aligned' in dataset_name:
    assert field_size_dict is not None, f"field_size_dict must be provided for {dataset_name} dataset"
    return get_summeval_dataset(
      dataset_name, tokenizer, wrap, mode, cache_dir,
      field_size_dict=field_size_dict, block_size=block_size, num_proc=num_proc, streaming=streaming, p_random=p_random)
  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'
  _path = os.path.join(cache_dir, filename)
  
  if utils.fsspec_exists(_path) and False:
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')
  LOGGER.info(f'Generating new data at: {_path}')

  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for sub-sampling
    block_size *= 2
  
  if dataset_name == 'wikitext103':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-103-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'wikitext2':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-2-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'ptb':
    dataset = datasets.load_dataset(
      'ptb_text_only', cache_dir=cache_dir)
  elif dataset_name == 'lambada':
    dataset = get_lambada_test_dataset()
  elif dataset_name == 'text8':
    assert wrap
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size)
  elif dataset_name == 'text8-crop':
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_arxiv':
    dataset = datasets.load_dataset(
      'scientific_papers', 'arxiv',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_pubmed':
    dataset = datasets.load_dataset(
      'scientific_papers', 'pubmed',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'ag_news':
    dataset = datasets.load_dataset(
      'ag_news',
      cache_dir=cache_dir,
      streaming=streaming)
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming)

  if dataset_name in ['lambada', 'openwebtext-train',
                      'openwebtext-valid']:
    data = dataset
  elif 'Genomic' in dataset_name:
    if mode == 'validation':
      return dataset['test']
    else:
      data = dataset[mode]
  else:
    data = dataset[mode]

  if dataset_name.startswith('wikitext'):
    detokenizer = wt_detokenizer
  elif dataset_name == 'ptb':
    detokenizer = ptb_detokenizer
  elif dataset_name == 'lm1b':
    detokenizer = lm1b_detokenizer
  elif dataset_name == 'lambada':
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith('scientific_papers'):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  def _apply_detokenizer(detokenizer):
    def detok(text):
      for i, t in enumerate(text, 0):
        text[i] = detokenizer(t)
      return text
    return detok
  
  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  if "Genomic" in dataset_name:
    num_classes = np.unique(data['label']).shape[0]
    if seq_to_seq_exp:
      encoding_length = block_size // 2 - 1
    else:
      encoding_length = int(np.ceil(np.log(num_classes) / np.log(4)))
    if seq_to_seq_exp:
      indices_by_class = {}
      for idx, lbl in enumerate(data['label']):
        if lbl not in indices_by_class:
          indices_by_class[lbl] = []
        indices_by_class[lbl].append(idx)

  def preprocess_and_tokenize(example):
    if dataset_name == 'ptb':
      text = example['sentence']
    elif 'scientific_papers' in dataset_name:
      text = example['article']
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)
    if 'Genomic' in dataset_name:
      text = example['seq']
      label = example['label']
      label_to_id = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
      def get_label_or_seq(lbl):
        if np.random.rand() < p_random:
          lbl = np.random.randint(0, num_classes)
        if seq_to_seq_exp:
          indices_with_lbl = indices_by_class[lbl]
          try:
            random_idx = np.random.randint(
              0, len(indices_with_lbl), size=(1,)).item()
          except:
            raise ValueError(f"indices_by_class: {indices_by_class}, lbl: {lbl}, indices_with_lbl: {indices_with_lbl}")
          seq = data['seq'][indices_with_lbl[random_idx]]
          if len(seq) < encoding_length:
            seq = seq + (tokenizer.pad_token * (encoding_length - len(seq)))
          else:
            seq = seq[:encoding_length]
          return seq
        else:
          lbl_enc = ''
          while lbl > 0:
            lbl_enc = label_to_id[lbl % 4] + lbl_enc
            lbl = lbl // 4
          if len(lbl_enc) < encoding_length:
            lbl_enc = 'A' * (encoding_length - len(lbl_enc)) + lbl_enc
          return lbl_enc
      label = list(map(get_label_or_seq, label)) + ['[SEP]']
      text = list(map(lambda x: f'{x[0]}{x[1]}', zip(label, text)))


    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      try:
        tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True,)
        # print(len(tokens['input_ids']))
      except:
        raise ValueError(f'Error tokenizing: {text}')
    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      desc='Tokenizing')
  else:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Tokenizing')
  if dataset_name == 'ptb':
    tokenized_dataset = tokenized_dataset.remove_columns(
      'sentence')
  elif 'scientific_papers' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns([
      'article', 'abstract', 'section_names'])
  elif dataset_name == 'ag_news':
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['text', 'label'])
  elif 'multi_news' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['document', 'summary'])
  elif 'Genomic' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['seq', 'label'])
    setattr(tokenized_dataset, 'var_indices', [list(range(encoding_length)), list(range(encoding_length, block_size))])
    print(f"var_indices: {tokenized_dataset.var_indices}")
  else:
    tokenized_dataset = tokenized_dataset.remove_columns(
      'text')

  if not wrap:
    tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format('torch')
  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      desc='Grouping')
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Grouping')
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def get_tokenizer(config):
  if "synthetic" in config.data.train:
    try:
      tokenizer = IdentityTokenizer(
      vocab_size=config.data.random_variable.dim)
    except:
      tokenizer = IdentityTokenizer(
      vocab_size=2)
    return tokenizer
  
  if config.data.tokenizer_name_or_path == 'text8':
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased')
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path, trust_remote_code=True)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer

def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  if "synthetic" in config.data.train:
    return get_synthetic_dataloaders(config, tokenizer)
  if skip_train:
    train_set = None
  else:
    if hasattr(config.data, 'field_size_dict'):
      field_size_dict = config.data.field_size_dict
    else:
      field_size_dict = None
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      field_size_dict=field_size_dict,
      p_random=config.data.p_random,
      seq_to_seq_exp=config.data.seq_to_seq_exp,)
  
  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  elif 'Genomic' in config.data.valid:
    validation_split = 'train'
  else:
    validation_split = 'validation'
  if skip_valid:
    valid_set = None
  else:
    if hasattr(config.data, 'field_size_dict'):
      field_size_dict = config.data.field_size_dict
    else:
      field_size_dict = None
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      field_size_dict=field_size_dict,
      streaming=False,
      p_random=config.data.p_random,
      seq_to_seq_exp=config.data.seq_to_seq_exp)

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=True)
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    if config.eval.compute_mutinfo:
      valid_loader = InfiniteDataLoader(valid_loader)
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer
  return train_loader, valid_loader

class InfiniteDataLoader:
  def __init__(self, dataloader):
      self.dataloader = dataloader
      self.iterator = cycle(dataloader)

  def __iter__(self):
      return self

  def __next__(self):
      return next(self.iterator)
  
  def __len__(self):
      return np.inf
  
  @property
  def dataset(self):
      return self.dataloader.dataset

class ConcatenatedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, p_random=0.0):
        self.datasets = datasets
        self._var_indices = None
        self.p_random = p_random

    def __len__(self):
        return len(next(iter(self.datasets.values())))

    def __getitem__(self, idx):
        input_ids_list = []
        attention_mask_list = []

        for i, dataset in enumerate(self.datasets.values()):
            if i > 0 and np.random.rand() < self.p_random:
              random_idx = torch.randint(
              low=0, high=len(dataset), size=(1,)).item()
              idx = random_idx
            item = dataset[idx]
            input_ids_list.extend(item['input_ids'])
            if 'attention_mask' in item:
                attention_mask_list.append(torch.tensor(item['attention_mask']))

        input_ids = torch.tensor(input_ids_list)
        if attention_mask_list:
            try:
              attention_mask = torch.cat(attention_mask_list, dim=0)
            except:
              raise ValueError(f'attention_mask_list: {attention_mask_list}')
        else:
            attention_mask = torch.ones_like(input_ids)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    @property
    def var_indices(self):
      if self._var_indices is not None:
        return self._var_indices
      self._var_indices = []
      for dataset in self.datasets.values():
        example = dataset[0]['input_ids']
        if len(self._var_indices) == 0:
          self._var_indices.append(list(range(len(example))))
        else:
          self._var_indices.append(list(
            range(self._var_indices[-1][-1] + 1, self._var_indices[-1][-1] + 1 + len(example))))
      return self._var_indices

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data):
      # check data is iterable
      assert hasattr(data, '__iter__'), 'Data must be iterable.'
      self.data = torch.cat(data, dim=1)
      self._var_indices = []
      for var in data:
        if len(self._var_indices) == 0:
          self._var_indices.append(list(range(var.shape[1])))
        else:
          self._var_indices.append(list(
            range(self._var_indices[-1][-1] + 1, self._var_indices[-1][-1] + 1 + var.shape[1])))
    
    def __len__(self):
      return len(self.data)
    
    def __getitem__(self, idx):
      return {'input_ids': self.data[idx], 'attention_mask': torch.ones_like(self.data[idx])}
    
    @property
    def var_indices(self):
      return self._var_indices

def get_synthetic_dataloaders(config, tokenizer):

  random_variable = hydra.utils.instantiate(
    config.data.random_variable)
  
  outputs = random_variable.rvs(
    config.data.train_size)
  
  if isinstance(outputs, tuple):
    xy_train = (
      torch.tensor(outputs[0], dtype=torch.long),
      torch.tensor(outputs[1], dtype=torch.long
      ))
  else:
    xy_train = (torch.tensor(outputs, dtype=torch.long),)
  
  train_loader = torch.utils.data.DataLoader(
    SyntheticDataset(xy_train),
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=True)
  train_loader.tokenizer = tokenizer

  valid_loader = torch.utils.data.DataLoader(
    SyntheticDataset(xy_train),
    batch_size=config.loader.eval_batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False)
  if config.eval.compute_mutinfo:
    valid_loader = InfiniteDataLoader(valid_loader)
  valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader

# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0