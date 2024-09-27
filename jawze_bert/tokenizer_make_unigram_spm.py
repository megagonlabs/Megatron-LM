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
        special_tokens=[],
):
    for token in tokens:
        t = model.ModelProto().SentencePiece()
        if isinstance(token, str):
            piece = token.encode('utf8')
            t.score = -1.0
        else:
            assert len(token) == 2 and isinstance(token[1], float), f"bad entry: {token}"
            piece = token[0].encode('utf8')
            t.score = token[1]
        t.piece = piece
        if piece == unk_token:
            logger.info(f"unk token {piece}")
            t.type = t.UNKNOWN
        elif piece.startswith('<0x') and piece.endswith('>'):
            logger.info(f"byte-fallback token {piece}")
            t.type = t.BYTE
        elif piece in special_tokens:
            logger.info(f"special token {piece}")
            t.type = t.CONTROL
        proto.pieces.append(t)


def save(proto, path):
    with open(path, 'wb') as fout:
        fout.write(proto.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', help='vocab json file')
    parser.add_argument('-o', '--output', help='output path')
    parser.add_argument('-u', '--unk_token', help='unk token', default="<unk>")
    parser.add_argument('-s', '--special_tokens_csv', help='special tokens csv', default="<s>,</s>,<MASK>,<PAD>,<CLS>,<SEP>,<EOD>")
    args = parser.parse_args()
    special_tokens = args.special_tokens_csv.split(",")

    logger.info("create new model using dummy trainer")
    proto = create_new_unigram_model()
    
    logger.info(f"loading {args.vocab}")
    with open(args.vocab, "r", encoding="utf8") as fin:
        tokens = json.load(fin)

    logger.info(f"adding {len(tokens)} tokens")
    add_tokens(proto, tokens, args.unk_token, special_tokens)

    logger.info(f"saving model to {args.output}")
    save(proto, args.output)


if __name__ == '__main__':
    main()
