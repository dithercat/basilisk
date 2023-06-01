# basilisk

tiny LLaMA inference server

## setup

1. follow the original ExLlama `README.md`, other than installing `flask` for
   the http server
2. create a `config.json` (see `config_example.json`)
   - `shared_secret` is optional; if set, the `Authorization` header of all
     requests will be checked against this
3. just run `basilisk.py`

## thank you

- turboderp for [ExLlama](https://github.com/turboderp/exllama)