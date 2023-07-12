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
    model_type='ecapa-tdnn',

    use_cuda=True,
    # use_cuda=False,
    max_epoch=301,
    batch_size=32,
    print_intervals=10000,
    lang=[
        "ALG",
        "EGY",
        "IRA",
        "JOR",
        "KSA",
        "KUW",
        "LEB",
        "LIB",
        "MAU",
        "MOR",
        "OMA",
        "PAL",
        "QAT",
        "SUD",
        "SYR",
        "UAE",
        "YEM"
    ],

)

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
