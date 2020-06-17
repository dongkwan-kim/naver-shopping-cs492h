import argparse
import string
import pickle
from torchtext.data import Field, TabularDataset


def build_vocab(csv_file, tokenizer_path, min_freq, max_size):
    """Build a simple vocabulary wrapper."""
    if tokenizer_path:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        TEXT = Field(
            sequential=True,
            use_vocab=True,
            tokenize=tokenizer.tokenize,
            lower=True,
            batch_first=True,
            include_lengths=False,
        )
    else:
        TEXT = Field(
            sequential=True,
            use_vocab=True,
            lower=True,
            batch_first=True,
            include_lengths=False,
        )

    train_data = TabularDataset(csv_file,
                                format='csv',
                                fields=[('text', TEXT)],
                                skip_header=True,
                                csv_reader_params={"delimiter": '\001'})
    for example in train_data.examples:
        text = vars(example)['text']
        text = [x.replace("\x02", " ") for x in text]  # remove \x02
        text = [''.join(c for c in s if c not in string.punctuation) for s in text]  # remove punctuation. e.g.!"#$%&\'()*+,-./:;
        text = ' '.join(text).split()
        vars(example)['text'] = text
    TEXT.build_vocab(train_data, min_freq=min_freq, max_size=max_size)

    return TEXT.vocab


def main(args):
    vocab = build_vocab(csv_file=args.csv_path, tokenizer_path=args.tokenizer_path, min_freq=args.min_freq, max_size=args.max_size)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./data/fashion_dataset_final.csv', help='path for dataset file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for saving vocabulary wrapper')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='path for saving vocabulary wrapper')
    parser.add_argument('--min_freq', type=int, default=3, help='minimum frequency needed to include a token in the vocabulary')
    parser.add_argument('--max_size', type=int, default=999999, help='maximum size of the vocabulary')
    args = parser.parse_args()
    print(args)
    main(args)