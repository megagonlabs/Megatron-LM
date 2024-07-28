import argparse
import json
import logging

import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as model

# see: https://github.com/google/sentencepiece/blob/31656da0c9cccfc47d4f0e69fc32d55faac3e1e9/python/add_new_vocab.ipynb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

def create_new_unigram_model(
        dummy_input_file="sp_dummy_input.txt",
        dummy_model_prefix="sp_dummy_model",
):
    spm.SentencePieceTrainer.train(
        input=dummy_input_file,
        model_prefix=dummy_model_prefix,
        vocab_size=275,
        model_type="unigram",
        normalization_rule_name="identity",
        accept_language="ja,en,zh,ko",
        split_by_unicode_script=False,
        split_by_number=False,
        split_by_whitespace=False,
        split_digits=False,
        allow_whitespace_only_pieces=True,
        # user_defined_symbols=" ",
        byte_fallback=True,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        pad_piece="<PAD>",
        minloglevel=0,
    )
    
    proto = model.ModelProto()
    with open(f"{dummy_model_prefix}.model", "rb") as fb:
        proto.ParseFromString(fb.read())
    while len(proto.pieces) > 0:
        print(proto.pieces.pop())
    return proto


def add_tokens(
        proto,
        tokens: list[str],
        unk_token="<unk>",
        special_tokens=["<s>", "</s>", "<MASK>", "<PAD>", "<CLS>", "<SEP>" ,"<EOD>"],
):
    for token in tokens:
        t = model.ModelProto().SentencePiece()
        t.piece = token.encode('utf8')
        t.score = 0
        if token == unk_token:
            logger.info(f"unk token {token}")
            t.type = t.UNKNOWN
        elif token.startswith('<0x') and token.endswith('>'):
            logger.info(f"byte-fallback token {token}")
            t.type = t.BYTE
        elif token in special_tokens:
            logger.info(f"special token {token}")
            t.type = t.CONTROL
        else:
            t.score = -1.0
        proto.pieces.append(t)


def save(proto, path):
    with open(path, 'wb') as fout:
        fout.write(proto.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', help='vocab json file')
    parser.add_argument('-o', '--output', help='output path')
    args = parser.parse_args()

    logger.info("create new model using dummy trainer")
    proto = create_new_unigram_model()
    
    logger.info(f"loading {args.vocab}")
    with open(args.vocab, "r", encoding="utf8") as fin:
        tokens = json.load(fin)

    logger.info(f"adding {len(tokens)} tokens")
    add_tokens(proto, tokens)

    logger.info(f"saving model to {args.output}")
    save(proto, args.output)


if __name__ == '__main__':
    main()
