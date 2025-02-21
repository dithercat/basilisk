from model import ExLlama, ExLlamaCache
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import model_init
import argparse
import torch
import os
import json
from flask import Flask, request
from sentence_transformers import SentenceTransformer

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

# Parse arguments

parser = argparse.ArgumentParser(description = "a very simple API server based on ExLlama")

model_init.add_args(parser)

parser.add_argument("-temp", "--temperature", type = float, help = "temperature", default = 0.5)
parser.add_argument("-topk", "--top_k", type = int, help = "top-k", default = 32)
parser.add_argument("-topp", "--top_p", type = float, help = "top-p", default = 0.2)
parser.add_argument("-minp", "--min_p", type = float, help = "min-p", default = 0.00)
parser.add_argument("-repp",  "--repetition_penalty", type = float, help = "repetition penalty", default = 1.1)
parser.add_argument("-repps", "--repetition_penalty_sustain", type = int, help = "past length for repetition penalty", default = 32)
parser.add_argument("-repdc", "--repetition_penalty_decay", type = int, help = "decay length for repetition penalty", default = 64)
parser.add_argument("-repat", "--positional_repetition_penalty", type = int, help = "positional repetition penalty", default = 1.2)
parser.add_argument("-beams", "--beams", type = int, help = "number of beams for beam search", default = 1)
parser.add_argument("-beamlen", "--beam_length", type = int, help = "number of future tokens to consider", default = 1)

args = parser.parse_args()

# load config

if os.path.exists("config.json"):
    with open("config.json", "rb") as f:
        args.__dict__.update(json.load(f))

model_init.post_parse(args)
model_init.get_model_files(args)

# Some feedback

print(f"the system is coming up. please wait. ^^")
print(f" -- tokenizer: {args.tokenizer}")
print(f" -- model config: {args.config}")
print(f" -- model: {args.model}")

# Instantiate model and generator

lconfig = model_init.make_config(args)

model = ExLlama(lconfig)
cache = ExLlamaCache(model)
tokenizer = ExLlamaTokenizer(args.tokenizer)

model_init.print_stats(model)

generator = ExLlamaGenerator(model, tokenizer, cache)
generator.settings = ExLlamaGenerator.Settings()
generator.settings.temperature = args.temperature
generator.settings.top_k = args.top_k
generator.settings.top_p = args.top_p
generator.settings.min_p = args.min_p
generator.settings.token_repetition_penalty_max = args.repetition_penalty
generator.settings.token_repetition_penalty_sustain = args.repetition_penalty_sustain
generator.settings.token_repetition_penalty_decay = generator.settings.token_repetition_penalty_sustain // 2
generator.settings.beams = args.beams
generator.settings.beam_length = args.beam_length

print("ExLlama initialized!")

# init SentenceTransformers
embedder = None
if args.embedding_model != None:
    embedder = SentenceTransformer(args.embedding_model)
    print("SentenceTransformer initialized!")

# HACK: optionally replace literal <s> and </s> with STX and ETX
#       and then replace STX and ETX with BOS and EOS
bos_tok = tokenizer.tokenizer.Encode("\x02")[1]
eos_tok = tokenizer.tokenizer.Encode("\x03")[1]
colon_tok = tokenizer.tokenizer.Encode(":")[0]
def tokenize_evil(str, special_convert=False):
    if special_convert:
        str = str.replace("<s>", "\x02").replace("</s>", "\x03")
    ids = tokenizer.tokenizer.Encode(str)
    for i in range(0, len(ids)):
        if ids[i] == bos_tok:
            ids[i] = tokenizer.bos_token_id
        if ids[i] == eos_tok:
            ids[i] = tokenizer.eos_token_id
    return ids

# actual app init
app = Flask(__name__)

# validate shared secret
def auth_check():
    secret = args.__dict__.get("shared_secret")
    return secret != None and request.headers.get("authorization") != secret

def err(str, code):
    return { "error": str }, code

# dedicated "see if the config is working" endpoint
# maps to kasumiLLM's ping()
# (which used to be a hack using textgen-webui token count endpoint)
@app.route("/basilisk/ping")
def get_ping():
    if auth_check():
        return err("unauthorized", 403)
    return "pong"

# allow clients to retrieve most of the base config
@app.route("/basilisk/config")
def get_config():
    if auth_check():
        return err("unauthorized", 403)
    
    cfg = {}
    cfg.update(args.__dict__)

    # dont disclose these
    del cfg["shared_secret"]
    del cfg["directory"]
    del cfg["config"]
    del cfg["model"]
    del cfg["tokenizer"]

    return cfg

# generate embeddings for a string using SentenceTransformer
@app.route("/basilisk/embed", methods=["POST"])
def post_embed():
    if auth_check():
        return err("unauthorized", 403)
    
    if embedder == None:
        return err("embedding model not loaded", 501)
    
    body = request.get_json()
    prompt = body.get("prompt")
    if prompt == None:
        return err("prompt required", 400)

    embedding = embedder.encode(prompt).tolist()

    return {
        "embedding": embedding,
        "model": args.embedding_model,
        "dimensions": len(embedding)
    }

# tokenize a string
# primary use is to get token count, but also for positional_repeat_inhibit
@app.route("/basilisk/tokenize", methods=["POST"])
def post_tokens():
    if auth_check():
        return err("unauthorized", 403)
    
    body = request.get_json()
    prompt = body.get("prompt")
    if prompt == None:
        return err("prompt required", 400)
    
    ids = tokenize_evil(prompt, body.get("special_convert"))

    frags = []
    for id in ids:
        frags.append(tokenizer.tokenizer.Decode([colon_tok, id])[1:])

    return {
        "tokens": ids,
        "fragments": frags
    }

STOP_UNKNOWN = -1
STOP_LIMIT = 0
STOP_EOS = 1
STOP_ENDSTR = 2

# actual inference endpoint
# param names are kind of sort of compatible with text-generation-webui
@app.route("/basilisk/infer", methods=["POST"])
def post_infer():
    if auth_check():
        return err("unauthorized", 403)
    
    # get parameters+overrides
    body = {
        "min_length": 4,
        "max_new_tokens": 256,
        "stopping_strings": ["\n"],

        # providing the last inference here attempts to prevent any token from
        # being generated at the same position as in the previous sequence
        "positional_repetition_penalty": 1.2,
        "positional_repeat_inhibit": [],

        # convert "<s>" and "</s>"?
        # you should avoid this if possible by using STX and ETX in your app
        "special_convert": False
    }
    body.update(args.__dict__)
    body.update(request.get_json())

    prompt = body.get("prompt")
    if prompt == None:
        return err("prompt required", 400)
    prompt = prompt.strip()

    # update settings
    generator.settings.temperature = body["temperature"]
    generator.settings.top_k = body["top_k"]
    generator.settings.top_p = body["top_p"]
    generator.settings.min_p = body["min_p"]
    generator.settings.token_repetition_penalty_max = body["repetition_penalty"]
    generator.settings.token_repetition_penalty_sustain = body["repetition_penalty_sustain"]
    generator.settings.token_repetition_penalty_decay = body["repetition_penalty_decay"]
    generator.settings.token_penalized_penalty = body["positional_repetition_penalty"]
    
    # tokenize stopping strings
    stopping_strings_tok = []
    for string in body["stopping_strings"]:
        toked = tokenizer.tokenizer.Encode(string)[1:]
        stopping_strings_tok.append(toked)
    
    # build positional inhibit lists
    pr_toks = []
    for seq in body["positional_repeat_inhibit"]:
        for i in range(len(seq)):
            if i == len(pr_toks):
                pr_toks.append([seq[i]])
            else:
                pr_toks[i].append(seq[i])
    
    idsl = tokenize_evil(prompt, body["special_convert"])
    print("context: {length}, limit: {limit}".format(limit=body["max_new_tokens"], length=len(idsl)))
    if len(idsl) + body["max_new_tokens"] >= 2048:
        return err("prompt length and generation limit must be less than 2048", 400)
    
    stop_reason = STOP_LIMIT

    # begin inference
    ids = torch.tensor([idsl])
    generator.gen_begin(ids)
    initial_len = generator.sequence[0].shape[0]

    generator.begin_beam_search()

    for i in range(body["max_new_tokens"]):
        # penalize inference repeats from last turn
        if i < len(pr_toks):
            generator.penalize_tokens(pr_toks[i])
        else:
            generator.penalize_tokens(None)

        # Disallowing the end condition tokens seems like a clean way to force longer replies.
        if i < body["min_length"]:
            generator.disallow_tokens([
                tokenizer.newline_token_id,
                tokenizer.eos_token_id
            ])
        else:
            generator.disallow_tokens(None)
        
        # Get a token
        token = generator.beam_search()

        # stop on stopping strings
        stophit = False
        for needle in stopping_strings_tok:
            haystack = generator.sequence[0][initial_len:][-len(needle):].tolist()
            if len(needle) > len(haystack):
                continue
            found = True
            for j in range(0, len(needle)):
                if haystack[j] != needle[j]:
                    found = False
                    break
            if found:
                stophit = True
        if stophit:
            stop_reason = STOP_ENDSTR
            break

        # If token is EOS, replace it with newline before continuing
        if token.item() == tokenizer.eos_token_id:
            generator.replace_last_token(tokenizer.newline_token_id)
            stop_reason = STOP_EOS
            break
    
    tokens = generator.sequence[0][initial_len:]
    text = tokenizer.decode(tokens)

    # get fragments
    frags = []
    for id in tokens.tolist():
        frags.append(tokenizer.tokenizer.Decode([colon_tok, id])[1:])

    # clean up stopping strings
    # this sucks
    for string in body["stopping_strings"]:
        l = len(string)
        if string == text[-l:]:
            text = text[:-l]
            break
    
    print("generated {length} tokens".format(length=len(tokens.tolist())))

    return {
        "text": text,
        "tokens": tokens.tolist(),
        "fragments": frags,
        "stop_reason": stop_reason
    }

print("starting flask...")

app.run(threaded=False)