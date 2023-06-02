# basilisk

tiny LLaMA inference server

## setup

1. follow the original ExLlama `README.md`, other than installing `flask` for
   the http server
2. create a `config.json` (see `config_example.json`)
   - `shared_secret` is optional; if set, the `Authorization` header of all
     requests will be checked against this
3. just run `basilisk.py`

## notes

- HACK: BOS and EOS are internally encoded as STX and ETX (`\x02` and `\x03`
  respectively) and converted to their actual representations after tokenizing
  - you can use the `special_convert` parameter to lazily replace literal `<s>`
    and `</s>` with their counterparts, but this isnt recommended
  - keep in mind that tokens generally have any spaces in the *front*, and
    HF Transformers tokenizers (*not* basilisk) usually eat whitespace
    surrounding these tokens. basilisk, on the other hand, treats STX/ETX as a
    literal BOS/EOS, and does no preprocessing to make it friendlier for the
    model
    - GOOD: `"ASSISTANT:<s> Hello,"` (tokenizes as `[..., ":", "<s>", " Hello", ","]`)
    - BAD: `"ASSISTANT: <s>Hello, "` (tokenizes as `[..., ":", " ", "<s>", "Hello", ",", " "]`)

## thank you

- turboderp for [ExLlama](https://github.com/turboderp/exllama)