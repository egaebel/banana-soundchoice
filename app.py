from potassium import Potassium, Request, Response
from tqdm import tqdm
from typing import List

import itertools
import more_itertools
import torch

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    from speechbrain.pretrained import GraphemeToPhoneme

    g2p = GraphemeToPhoneme.from_hparams(
        "speechbrain/soundchoice-g2p", run_opts={"device": device}
    )
    context = {
        "device": device,
        "g2p": g2p,
        "model": g2p.mods.model,
        "soundchoice_batch_size": 8,
    }

    return context


def g2p_wrapper_before(g2p, text):
    """Performs the Grapheme-to-Phoneme conversion
    Arguments
    ---------
    text: str or list[str]
        a single string to be encoded to phonemes - or a
        sequence of strings
    Returns
    -------
    result: list
        if a single example was provided, the return value is a
        single list of phonemes
    """

    model_inputs = g2p.encode_input({"txt": text})
    g2p._update_graphemes(model_inputs)
    return model_inputs


def g2p_wrapper_after(g2p, model_outputs):
    """Performs the Grapheme-to-Phoneme conversion
    Arguments
    ---------
    text: str or list[str]
        a single string to be encoded to phonemes - or a
        sequence of strings
    Returns
    -------
    result: list
        if a single example was provided, the return value is a
        single list of phonemes
    """
    decoded_output = g2p.decode_output(model_outputs)
    phonemes = decoded_output["phonemes"]
    return phonemes


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    text_list = request.json.get("text_list")

    device = context.get("device")
    g2p = context.get("g2p")
    model = context.get("model")
    soundchoice_batch_size: int = context.get("soundchoice_batch_size")

    text_list_str: str = str(text_list)[:100]
    print(
        f"Running soundchoice on device: '{str(device)}' "
        f"on text_list with: '{len(text_list)}' "
        f"items with batch size: '{soundchoice_batch_size}' "
        f"and text_list_str: '{text_list_str}'....."
    )
    phoneme_list: List[List[str]] = list(
        itertools.chain.from_iterable(
            g2p_wrapper_after(g2p, model(**g2p_wrapper_before(g2p, text_list_chunk)))
            for text_list_chunk in tqdm(
                list(more_itertools.chunked(text_list, soundchoice_batch_size))
            )
        )
    )
    print(
        f"Finished running soundchoice  on device: '{str(device)}' "
        f"on text_list with: '{len(text_list)}' "
        f"items with batch size: '{soundchoice_batch_size}' "
        f"and text_list_str: '{text_list_str}'!"
    )

    return Response(json={"outputs": phoneme_list}, status=200)


if __name__ == "__main__":
    app.serve()
