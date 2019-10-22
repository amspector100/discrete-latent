import io
import torchtext


class PennTreebankSentence(torchtext.datasets.PennTreebank):
    
    def __init__(self, path, text_field, newline_eos=True,
                 encoding='utf-8', **kwargs):
        """
        Create a LanguageModelingDataset given a path and a field.
        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field)]
        examples = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                text = text_field.preprocess(line)
                examples += [torchtext.data.Example.fromlist(
                                [text],
                                fields
                            )]

        super(torchtext.datasets.LanguageModelingDataset, self).__init__(
            examples, fields, **kwargs)


# TODO : COMMENT
def sort_key_fn(example):
    return len(example.text) + 2
  
def pad_batch(batch, multiple, padding_idx):
    if len(batch) % multiple == 0:
        return batch
    length = (len(batch) // multiple + 1) * multiple
    new_batch = torch.full((length, batch.size(1)), padding_idx, 
                           dtype=torch.long, device=batch.device)
    new_batch[:len(batch)] = batch
    return new_batch


def get_data_bucketiters([splits], batch_size):
    out = []
    for split in splits:
        out.append(torchtext.data.BucketIterator(split, batch_size, 
            sort_key=sort_key_fn, sort_within_batch=True, 
            shuffle=True, device=device))
    return out

def load_PTB(batch_size):
    TEXT = torchtext.data.Field(init_token='<s>', eos_token='</s>')
    train, val, test = PennTreebankSentence.splits(TEXT)
    padding_idx = TEXT.vocab.stoi['<pad>']
    vocab_size = len(TEXT.vocab)
    train_iter, val_iter, test_iter = get_data_bucketiters([train, val, test], batch_size)
    return {'TEXT' : TEXT, 'train' : train_iter, 'val' : val_iter, 
            'test' : test_iter, 'vocab_size' : vocab_size}


# TEXT.vocab.load_vectors(vectors='glove.840B.300d')
# embeddings = TEXT.vocab.vectors
# n_downsize = 2
# pad_to_multiple = 2 ** (n_downsize)
# batch_size = 64

