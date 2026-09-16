def register() -> None:
    # The module imports vLLM internals, so import it lazily once vLLM has loaded.
    from vllm_ngram_pager import embedding

    embedding.register()
