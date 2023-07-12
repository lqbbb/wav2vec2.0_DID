from tfcompat.hparam import HParams
import numpy as np

# Default hyperparameters:
hparams = HParams(
    name="olr",
    sample_rate=16000,
    # num_mels=80,
    n_fft=int(0.04*16000),
    # n_fft=2048,
    hop_length=int(0.02*16000),
    win_length=int(0.04*16000),
    deltas=False,

    # training testing evaluating
    model_type='dtdnnss',

    use_cuda=True,
    max_epoch=200,
    batch_size=32,
    print_intervals=1000,

    lang=[
        "Minnan",
        "Shanghai",
        "Sichuan",
        "unknown",

    ]

)

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
