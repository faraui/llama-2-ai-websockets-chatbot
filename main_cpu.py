pimport config

import time
import json
import requests
import websocket
import llama_cpp
from pprint import pprint
import threading

import sys

gpt = llama_cpp.Llama(
      model_path = sys.argv[1],
      n_ctx = 4096,
      n_parts = -1,
      n_gpu_layers = 0,
      seed = 1337,
      f16_kv = True,
      logits_all = False,
      vocab_only = False,
      use_mmap = False,
      use_mlock = False,
      embedding = False,
      n_threads = 4,
      n_batch = int(sys.argv[2]) if len(sys.argv) > 2 else 29,
      last_n_tokens_size = 64,
      lora_base = None,
      lora_path = None,
      verbose = True
     )

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant. Always answer as helpfully as possible.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def tokenize_dialog(dialog):
    print('# DEBUG (line 42): tokenize_dialog: dialog:')
    pprint(dialog)
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    prompt_tokens = sum([
        [gpt.token_bos()] + gpt.tokenize(
            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ".encode(), add_bos=False
        ) + [gpt.token_eos()]
        for prompt, answer in zip(
            dialog[::2],
            dialog[1::2],
        )
    ], [])
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    prompt_tokens += [gpt.token_bos()] + gpt.tokenize(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}".encode(), add_bos=False
    )
    print('# DEBUG (line 62): tokenize_dialog: dialog:')
    pprint(gpt.detokenize(prompt_tokens).decode())
    return prompt_tokens

def append_system_prompt(dialog):
    return [
        {
            "role": dialog[0]["role"],
            "content": B_SYS
            + DEFAULT_SYSTEM_PROMPT
            + E_SYS
            + dialog[0]["content"],
        }
    ] + dialog[1:]

chat_dialog = []
chat_context = []

def onMessage(ws, event):
    global chat_dialog
    if not event['echo']:
        chat_dialog.append({
            'role': 'user',
            'content': event['content']
        })
        if len(chat_dialog) == 1:
            chat_dialog = append_system_prompt(chat_dialog)
        ws.send(json.dumps({
            'type': 'new_message',
            'chat_id': event['chat_id'],
            'content': 'Thinking...'
        }))
    else:
        threading.Thread(target=think, args=(ws, event)).start()

def think(ws, event):
    print('# DEBUG (line 98): think: chat_dialog:')
    pprint(chat_dialog)
    context = tokenize_dialog([chat_dialog[-1]])
    gpt.eval(context)
    c = b''
    while True:
        token = gpt.sample(top_p=0.9, top_k=40, temp=0.6, repeat_penalty=1.1)
        gpt.eval([token])
        is_end = (token == gpt.token_eos())
        c += gpt.detokenize([token])
        try:
            content = c.decode()
        except UnicodeDecodeError:
            continue
        ws.send(json.dumps({
            'type': 'edit_message',
            'chat_id': event['chat_id'],
            'message_id': event['message_id'],
            'content': content + ('' if is_end else ' █')
        }))
        if is_end:
            chat_dialog.append({
                'role': 'assistant',
                'content': content
            })
            break
    print('# DEBUG (line 124): think: chat_dialog:')
    pprint(chat_dialog)

def on_message(ws, event):
    event = json.loads(event)
    print(f'>>> {event["type"].rjust(12)} > {event}')
    if event['type'] == 'new_message':
        onMessage(ws, event)

if __name__ == '__main__':
    token = requests.post(
        f'http://{config.host}/auth',
        data=json.dumps({'login': 'ai', 'pass': 'pass'})
    ).json()["token"]
    print(f'main: token    = {token}')
    ws_token = requests.post(f'http://{config.host}/auth/websocket', data=json.dumps({'token': token})).json()["token"]
    print(f'main: ws_token = {ws_token}')
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(f'ws://{config.host}/ws/{ws_token}', on_message=on_message)
    print('Running...')
    ws.run_forever()
