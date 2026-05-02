# register_test.py
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoConfig
from ngram_model import NGramConfigMLM, NGramDebertaV2Model, NGramDebertaV2ForMaskedLM
from ngram_model import LoraConfigMLM, LoraModel, LoraModelForMaskedLM

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--push_to_hub", type=str, default="")


if __name__ == "__main__":
    args = parser.parse_args()
    # Register for auto_map (used when pushing to Hub)
    # NGramConfigMLM.register_for_auto_class()
    # NGramDebertaV2Model.register_for_auto_class("AutoModel")
    # NGramDebertaV2ForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")

    # # Register for local use
    # AutoConfig.register("ngram-deberta-v2", NGramConfigMLM)
    # AutoModel.register(NGramConfigMLM, NGramDebertaV2Model)
    # AutoModelForMaskedLM.register(NGramConfigMLM, NGramDebertaV2ForMaskedLM)

    # # Load model and tokenizer
    # model = NGramDebertaV2ForMaskedLM.from_pretrained(args.model_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # model.config.auto_map = {
    #     "AutoConfig": "ngram_model.NGramConfigMLM",
    #     "AutoModel": "ngram_model.NGramDebertaV2Model",
    #     "AutoModelForMaskedLM": "ngram_model.NGramDebertaV2ForMaskedLM"
    # }

    # model.config.save_pretrained(args.model_path)
    # model.save_pretrained(args.model_path)

    # # Push to Hub
    # model.push_to_hub(args.push_to_hub, private=True)
    # tokenizer.push_to_hub(args.push_to_hub, private=True)


    LoraConfigMLM.register_for_auto_class()
    LoraModel.register_for_auto_class("AutoModel")
    LoraModelForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")

    # Register for local use
    AutoConfig.register("lora-deberta-v2", LoraConfigMLM)
    AutoModel.register(LoraConfigMLM, LoraModel)
    AutoModelForMaskedLM.register(LoraConfigMLM, LoraModelForMaskedLM)

    # Load model and tokenizer
    model = LoraModelForMaskedLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model.config.auto_map = {
        "AutoConfig": "ngram_model.LoraConfigMLM",
        "AutoModel": "ngram_model.LoraModel",
        "AutoModelForMaskedLM": "ngram_model.LoraModelForMaskedLM"
    }

    model.config.save_pretrained(args.model_path)
    model.save_pretrained(args.model_path)

    # Push to Hub
    # model.push_to_hub(args.push_to_hub, private=True)
    # tokenizer.push_to_hub(args.push_to_hub, private=True)
