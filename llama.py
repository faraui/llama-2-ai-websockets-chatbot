import llama_cpp
from pprint import pprint

gpt = llama_cpp.Llama(
      model_path = 'llama-2-7b-chat.ggmlv3.q4_0.bin',
      n_ctx = 4096,
      n_parts = -1,
      n_gpu_layers = 0,
      seed = 1337,
      f16_kv = True,
      logits_all = False,
      vocab_only = False,
      use_mmap = True,
      use_mlock = False,
      embedding = False,
      n_threads = 4,
      n_batch = 29,
      last_n_tokens_size = 64,
      lora_base = None,
      lora_path = None,
      verbose = True
     )

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def chat_completion_generator(
    dialog,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len = None,
    logprobs: bool = False,
):
    if max_gen_len is None:
        max_gen_len = 2048 - 1
    print('DEBUG: line 42')
    pprint(dialog)
    if dialog[0]["role"] != "system":
        dialog = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + dialog
    print('DEBUG: line 51')
    pprint(dialog)
    dialog = [
        {
            "role": dialog[1]["role"],
            "content": B_SYS
            + dialog[0]["content"]
            + E_SYS
            + dialog[1]["content"],
        }
    ] + dialog[2:]
    print('DEBUG: line 62')
    pprint(dialog)
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    prompt_tokens = sum(
        [
            [gpt.token_bos()] + gpt.tokenize(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ".encode(), add_bos=False
            ) + [gpt.token_eos()]
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ],
        [],
    )
    print('DEBUG: line 82')
    pprint(prompt_tokens)
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    prompt_tokens += [gpt.token_bos()] + gpt.tokenize(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}".encode(), add_bos=True
    )
    print('DEBUG: line 90')
    pprint(gpt.detokenize(prompt_tokens).decode())
    generator = gpt.generate(prompt_tokens, top_k=40, top_p=top_p, temp=temperature, repeat_penalty=1.1, reset=False)
    return generator


gpt.__call__(
         prompt,
         suffix = None,
         max_tokens = -1,
         temperature = 0.8,
         top_p = 0.95,
         logprobs = None,
         echo = False,
         stop = [],
         repeat_penalty = 1.1,
         top_k = 40,
         stream = False
)


chat = [
    {
        'role': 'user',
        'content': 'Hi'
    },
    {
        'role': 'assistant',
        'content': 'Hello. I am an AI chatbot. Would you like to talk?'
    },
    {
        'role': 'user',
        'content': 'Sure!'
    },
    {
        'role': 'assistant',
        'content': 'What would you like to talk about?'
    },
    {
        'role': 'user',
        'content': ''
    }
]
gpt.create_chat_completion(
    messages = chat,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    stream=False,
    stop=[],
    max_tokens=256,
    repeat_penalty=1.1
)

t = gpt.tokenize(b'You: Hi\nAI: Hello. I am an AI chatbot. Would you like to talk?\nYou: Sure!\nAI: What would you like to talk about?\nYou:')
print(gpt.detokenize(t).decode(), end='', flush=True)
while True:
    t += gpt.tokenize(f' {input(" ")}\n'.encode())[1:]
    g = gpt.generate(t, top_k=40, top_p=0.95, temp=0.8, repeat_penalty=1.1, reset=False)
    t = []
    y = False
    while True:
        n = next(g)
        print(gpt.detokenize([n]).decode(), end='', flush=True)
        if n == 3492: y = True
        if (n == 29901) and y: break
