from text import symbols

class hparams:

    # text Parameters
    cleaner_names = ['english_cleaners']

    # Audio Parameters
    sampling_rate = 22050
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    n_mel_channels = 80
    mel_fmin = 0.0
    mel_fmax = 8000.0

    # Model Parameters
    # embedding parameters
    n_symbols = len(symbols)
    character_embedding_dim = 512

    # encoder parameters
    encoder_kernel_size = 5
    encoder_embedding_dim = 512
    encoder_n_convolutions = 3

    # decoder parameters
    n_frames_per_step = 3










