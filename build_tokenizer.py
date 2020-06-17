import math
import pickle
import argparse

from soynlp.utils import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import LTokenizer, MaxScoreTokenizer


def word_score(score):
    return (score.cohesion_forward * math.exp(score.right_branching_entropy))


def main(args):
    # Find patterns and extract words from a given set of documents
    sentences = DoublespaceLineCorpus(args.corpus_fname, iter_sent=True)
    word_extractor = WordExtractor(
        min_frequency=100,
        min_cohesion_forward=0.05,
        min_right_branching_entropy=0.0
    )

    # word extractor
    word_extractor.train(sentences)
    words = word_extractor.extract()
    cohesion_score = {word: score.cohesion_forward for word, score in words.items()}
    print('Word   (Freq, cohesion, branching entropy)\n')
    for word, score in sorted(words.items(), key=lambda x: word_score(x[1]), reverse=True)[:30]:
        print('%s     (%d, %.3f, %.3f)' % (word,
                                           score.leftside_frequency,
                                           score.cohesion_forward,
                                           score.right_branching_entropy))

    # noun extractor
    noun_extractor = LRNounExtractor_v2()
    nouns = noun_extractor.train_extract(args.corpus_fname)  # list of str like
    noun_scores = {noun: score.score for noun, score in nouns.items()}

    # combined score
    combined_scores = {noun: score + cohesion_score.get(noun, 0)
                       for noun, score in noun_scores.items()}
    combined_scores.update(
        {subword: cohesion for subword, cohesion in cohesion_score.items()
         if not (subword in combined_scores)})

    # maxScore tokenizer
    tokenizer = MaxScoreTokenizer(scores=combined_scores)

    # save tokenizer
    with open(args.tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_fname', type=str, default='./data/train_prod_nm.txt', help='path for dataset file')
    parser.add_argument('--tokenizer_path', type=str, default='./data/tokenizer.pkl', help='path for saving tokenizer')
    parser.add_argument('--score', type=str, default='combine', help='path for saving tokenizer')
    parser.add_argument('--tokenize', type=str, default='maxscore', help='path for saving tokenizer')
    args = parser.parse_args()
    print(args)
    main(args)
