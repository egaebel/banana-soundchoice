from potassium import Potassium, Request, Response
from tqdm import tqdm
from typing import List

import itertools
import more_itertools
import time
import torch

app = Potassium("soundchoice-g2p")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    init_start_time = int(time.time())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Init with device: '{device}'", flush=True)
    from speechbrain.pretrained import GraphemeToPhoneme

    model = GraphemeToPhoneme.from_hparams(
        "speechbrain/soundchoice-g2p", run_opts={"device": device}
    )
    context = {
        "model": model,
        "soundchoice_batch_size": 32,
    }
    init_end_time = int(time.time())
    print(
        f"Finished running init in: '{init_end_time - init_start_time}' seconds!",
        flush=True,
    )

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    inference_start_time = int(time.time())
    text_list = request.json.get("text_list")

    model = context.get("model")
    soundchoice_batch_size: int = context.get("soundchoice_batch_size")

    device = "cuda:0"
    model = model.to(device)

    text_list_str: str = str(text_list)[:100]
    print(
        f"Running soundchoice on device: '{str(device)}' "
        f"on text_list with: '{len(text_list)}' "
        f"items with batch size: '{soundchoice_batch_size}' "
        f"and text_list_str: '{text_list_str}'....."
    )
    phoneme_list: List[List[str]] = list(
        itertools.chain.from_iterable(
            model(text_list_chunk)
            for text_list_chunk in tqdm(
                list(more_itertools.chunked(text_list, soundchoice_batch_size))
            )
        )
    )
    inference_end_time = int(time.time())
    print(
        f"Finished running soundchoice on device: '{str(device)}' "
        f"in '{inference_end_time - inference_start_time}' seconds!\n"
        f"Ran on text_list with: '{len(text_list)}' "
        f"items with batch size: '{soundchoice_batch_size}' "
        f"and text_list_str: '{text_list_str}'!"
    )

    return Response(json={"outputs": phoneme_list}, status=200)


if __name__ == "__main__":
    app.serve()
