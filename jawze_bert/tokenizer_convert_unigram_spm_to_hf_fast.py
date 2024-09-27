import argparse
import re

from transformers import PreTrainedTokenizerFast
from tokenizers import decoders, models, normalizers, processors, Regex, Tokenizer


"""Tokenizer convert tool for llmjp-tokenizer.

You need to install some packages beforehand.

```console
$ pip install sentencepiece==0.1.99 "protobuf<3.21.0" transformers
```

And retain `sentencepiece_model_pb2.py` from official site.

```console
$ curl -O https://raw.githubusercontent.com/google/sentencepiece/master/python/sentencepiece_model_pb2.py
```
"""

def get_proto():
    try:
        import sys

        sys.path.append(".")

        import sentencepiece_model_pb2 as model
    except Exception:
        raise Exception(
            "You don't seem to have the required protobuf file, in order to use this function you need to run `pip install protobuf` and `wget https://raw.githubusercontent.com/google/sentencepiece/master/python/sentencepiece_model_pb2.py` for us to be able to read the intrinsics of your spm_file. `pip install sentencepiece` is not required."
        )

    m = model.ModelProto()
    return m


def convert_unigram_spm_to_hf(args) -> Tokenizer:
    proto = get_proto()
    proto.ParseFromString(open(args.input_sp_model_path, "rb").read())
    model_type = proto.trainer_spec.model_type
    assert model_type == 1, f"You're trying to run a `Unigram` model but your file was trained with a different algorithm ({model_type=})"
    vocab = []
    special_tokens = []
    score_replacing_pattern = re.compile(args.score_replacing_pattern)
    for piece in proto.pieces:
        if piece.piece == "":
            continue
        if score_replacing_pattern.search(piece.piece):
            score = args.score_replacing_value
        else:
            score = piece.score
        vocab.append((piece.piece, score))
        if piece.type in [2, 3, 4, 5]:
            special_tokens.append(piece.piece)
    unk_id = proto.trainer_spec.unk_id
    tokenizer = Tokenizer(models.Unigram(vocab, unk_id, byte_fallback=True))
    tokenizer.add_special_tokens(special_tokens)
    normalizer_list = []
    precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
    if precompiled_charsmap:
        normalizer_list.append(normalizers.Precompiled(precompiled_charsmap))
    replacement = "▁"
    """
    # do not use Metaspace pre_tokenizer because all the continuous spaces are divided into single space sequences 
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=replacement, add_prefix_space=True
    )
    """
    # using normalizer to insert "▁" to the beginning of text and to replace space to "▁"
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(Regex("(?<!\\n)^"), replacement),
            normalizers.Replace(Regex(" "), replacement),
        ]
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single=[args.cls_token, "$0", args.sep_token],
        pair=[args.cls_token, "$A", args.sep_token, "$B:1", f"{args.sep_token}:1"],
        special_tokens=[
            (args.cls_token, tokenizer.get_vocab()[args.cls_token]),
            (args.sep_token, tokenizer.get_vocab()[args.sep_token]),
        ],
    )
    """
    # do not use Metaspace decoder because all the heading spaces are removed
    tokenizer.decoder = decoders.Metaspace(
        replacement=replacement, add_prefix_space=True
    )
    """
    # using Replace decoders to remove the extra space char at the beginning of text and replace "▁" to space
    tokenizer.decoder = decoders.Sequence(
        [
            decoders.ByteFallback(),
            decoders.Replace(Regex(replacement), " "),
            decoders.Fuse(),
            decoders.Replace(Regex(f"(?<!\\n)^ "), ""),
        ]
    )
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_sp_model_path",
        required=True,
        type=str,
        help="path for input sentencepiece unigram model file",
    )
    parser.add_argument(
        "-o", "--output_hf_tokenizer_dir",
        required=True,
        type=str,
        help="path for output huggingface tokenizers directory",
    )
    parser.add_argument("-l", "--model_max_length", required=True, type=int)
    parser.add_argument("--score_replacing_pattern", default=r"^<(0x[0-9A-Z]{2}|padded_[0-9]+)>$")
    parser.add_argument("--score_replacing_value", default=-1000.0)
    parser.add_argument("--pad_token", default="<PAD>")
    parser.add_argument("--unk_token", default="<unk>")
    parser.add_argument("--cls_token", default="<CLS>")
    parser.add_argument("--sep_token", default="<SEP>")
    parser.add_argument("--mask_token", default="<MASK>")
    parser.add_argument("--eod_token", default="<EOD>")
    parser.add_argument("--bos_token", default="<s>")
    parser.add_argument("--eos_token", default="</s>")
    args = parser.parse_args()
    print("converting", args.input_sp_model_path, "to", args.output_hf_tokenizer_dir)
    fast_tokenizer = convert_unigram_spm_to_hf(args)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast_tokenizer,
        pad_token = args.pad_token,
        unk_token = args.unk_token,
        cls_token = args.cls_token,
        sep_token = args.sep_token,
        mask_token = args.mask_token,
        eod_token = args.eod_token,
        bos_token = args.bos_token,
        eos_token = args.eos_token,
        additional_special_tokens=[args.eod_token],
        model_max_length=args.model_max_length,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.save_pretrained(args.output_hf_tokenizer_dir)


if __name__ == "__main__":
    main()
