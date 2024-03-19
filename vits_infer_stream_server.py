import os
import sys
import numpy as np
import time
import fire
import signal
import json

import torch
import utils
import argparse
import base64

from scipy.io import wavfile
#from vits_chinese.text.symbols import symbols
#from vits_chinese.text import cleaned_text_to_sequence
#from vits_chinese.vits_pinyin import VITS_PinYin
from text.symbols import symbols
from text import cleaned_text_to_sequence
from vits_pinyin import VITS_PinYin

from multiprocessing import Process, Queue
#from utils.communicate import to_thread
# asyncio.to_thread is only supported after Python 3.10
import asyncio
import contextvars, functools
async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)

import websockets
import asyncio
import rich
from rich.console import Console 
console = Console(highlight=False)
console._log_render.omit_repeated_times = False

import logging
logging.getLogger("websockets").setLevel(logging.INFO)

addr = '0.0.0.0'
port = '6009'
RATE = 16000
DELTA = 1

def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))

stop_stream = False

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

def stream_synthesizer(text):
    global model
    global tts_front
    global device
    global stop_stream
    phonemes, char_embeds = tts_front.chinese_to_phonemes(text)
    input_ids = cleaned_text_to_sequence(phonemes)
    with torch.no_grad():
        x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
        x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
        audio = []
        audio_generator = model.infer_stream_generator(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5,length_scale=1)
        t0 = time.time()
        num = 0
        for chunk, finish in audio_generator:
            num += 1
            sys.stdout.write(f'chunk {num}: {len(chunk)/RATE}s\n')
            #chunk *= 32767 / max(0.01, np.max(np.abs(chunk))) * 0.6
            chunk *= 32767
            audio.extend(chunk)
            chunk = np.array(chunk, dtype=np.int16).tobytes()
            sys.stdout.flush()
            #time.sleep(1.5)
            if stop_stream:
                stop_stream = False
                break
            else:
                yield chunk, finish
        duration = len(audio) / RATE
        sys.stdout.write(f'RTF: {(time.time() - t0) / duration:.4f}, Total {num} chunks, {duration:.3f}s\n')

def init_synthesizer(queue_in: Queue, queue_out: Queue):
    global args
    global model
    global tts_front
    global device

    console.log('[green4]Loading TTS model', end='\n\n')
    t0 = time.time()
    # device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'gpu' or args.device == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # pinyin
    tts_front = VITS_PinYin("./bert", device)

    # config
    hps = utils.get_hparams_from_file(args.config)

    # model
    model = utils.load_class(hps.train.eval_class)(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)

    utils.load_model(args.model, model)
    model.eval()
    model.to(device)
    t1 = time.time()
    console.log(f'[yellow]Model loaded ({t1-t0}s)', end='\n')

    signal.signal(signal.SIGINT, signal_handler)
    
    queue_out.put(True) # 通知主进程加载完了

    while True:
        text = queue_in.get()       # 从队列中获取任务消息
        synthesizer = stream_synthesizer(text)
        for chunk, finish in synthesizer:
            message = {'audio': base64.b64encode(chunk).decode('utf-8'),
                       'finish': finish}
            queue_out.put(message)

async def ws_serve(websocket, path):
    global loop
    global stop_stream
    global queue_in, queue_out

    console.log(f'Receive connection: {websocket}', style='yellow')

    try:
        async for text in websocket:
            console.log(f'Text received. {text} | start generating audio')

            queue_in.put(text)
            while True:
                message = await to_thread(queue_out.get)
                await websocket.send(json.dumps(message))
                if message['finish']:
                    break

    except websockets.ConnectionClosed:
        console.log("ConnectionClosed...", )
    except websockets.InvalidState:
        console.log("InvalidState...")
    except Exception as e:
        console.log("Exception:", e)

async def main():
    global args
    global loop; loop = asyncio.get_event_loop()
    global queue_in, queue_out

    queue_in = Queue()
    queue_out = Queue()
    tts_process = Process(target=init_synthesizer, args=(queue_in, queue_out), daemon=True)
    tts_process.start()
    queue_out.get() # 等待新进程加载完成

    console.rule('[green3] server started'); console.line()
    start_server = websockets.serve(ws_serve, 
                                addr, 
                                port, 
                                subprotocols=["binary"], 
                                max_size=None)
    try:
        await start_server
    except OSError as e:            # 有时候可能会因为端口占用而报错，捕获一下
        console.log(f'ERROR: {e}', style='bright_red'); console.input('...')
        sys.exit()
    await asyncio.Event().wait()    # 持续运行


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.log('See you')
        sys.exit()
        
if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description='Inference code for bert vits models')
    #parser.add_argument('--config', type=str, required=False, default='vits_chinese/configs/bert_vits.json')
    #parser.add_argument('--model', type=str, required=False, default='vits_chinese/vits_chinese_pretrained_model/vits_bert_model.pth')
    parser.add_argument('--config', type=str, required=False, default='configs/bert_vits.json')
    parser.add_argument('--model', type=str, required=False, default='vits_chinese_pretrained_model/vits_bert_model.pth')
    parser.add_argument('--device', type=str, required=False, default='auto')
    args = parser.parse_args()

    fire.Fire(run)
