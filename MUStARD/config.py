class Config:

    model = "SVM"
    runs = 1  # No. of runs of experiments

    # Training modes
    use_context = False # whether to use context information or not (default false)
    use_author = False  # add author one-hot encoding in the input

    use_bert = True # if False, uses glove pooling

    use_target_text = False
    use_target_audio = False # adds audio target utterance features.
    use_target_video = False # adds video target utterance features.

    speaker_independent = False  # speaker independent experiments

    embedding_dim = 300  # GloVe embedding size
    word_embedding_path = "/home/sacastro/glove.840B.300d.txt"
    max_sent_length = 20
    max_context_length = 4  # Maximum sentences to take in context
    num_classes = 2  # Binary classification of sarcasm
    epochs = 15
    batch_size = 16
    val_split = 0.1  # Percentage of data in validation set from training data

    svm_c = 10.0
    svm_scale = True


class SpeakerDependentTConfig(Config):
    use_target_text = True
    svm_c = 1.0


class SpeakerDependentAConfig(Config):
    use_target_audio = True
    svm_c = 1.0


class SpeakerDependentVConfig(Config):
    use_target_video = True
    svm_c = 1.0


class SpeakerDependentTAConfig(Config):
    use_target_text = True
    use_target_audio = True
    svm_c = 1.0


class SpeakerDependentTVConfig(Config):
    use_target_text = True
    use_target_video = True
    svm_c = 10.0


class SpeakerDependentAVConfig(Config):
    use_target_audio = True
    use_target_video = True
    svm_c = 30.0


class SpeakerDependentTAVConfig(Config):
    use_target_text = True
    use_target_audio = True
    use_target_video = True
    svm_c = 10.0


class SpeakerDependentTPlusContext(SpeakerDependentTConfig):
    use_context = True
    svm_c = 1.0


class SpeakerDependentTPlusAuthor(SpeakerDependentTConfig):
    use_author = True
    svm_c = 10.0


class SpeakerDependentTVPlusContext(SpeakerDependentTVConfig):
    use_context = True
    svm_c = 10.0


class SpeakerDependentTVPlusAuthor(SpeakerDependentTVConfig):
    use_author = True
    svm_c = 10.0


class SpeakerIndependentTConfig(Config):
    svm_scale = False
    use_target_text = True
    svm_c = 10.0
    speaker_independent = True


class SpeakerIndependentAConfig(Config):
    svm_scale = False
    use_target_audio = True
    svm_c = 1000.0
    speaker_independent = True


class SpeakerIndependentVConfig(Config):
    svm_scale = False
    use_target_video = True
    svm_c = 30.0
    speaker_independent = True


class SpeakerIndependentTAConfig(Config):
    svm_scale = False
    use_target_text = True
    use_target_audio = True
    svm_c = 500.0
    speaker_independent = True


class SpeakerIndependentTVConfig(Config):
    svm_scale = False
    use_target_text = True
    use_target_video = True
    svm_c = 10.0
    speaker_independent = True


class SpeakerIndependentAVConfig(Config):
    svm_scale = False
    use_target_audio = True
    use_target_video = True
    svm_c = 500.0
    speaker_independent = True


class SpeakerIndependentTAVConfig(Config):
    svm_scale = False
    use_target_text = True
    use_target_audio = True
    use_target_video = True
    svm_c = 1000.0
    speaker_independent = True


class SpeakerIndependentTPlusContext(SpeakerIndependentTConfig):
    use_context = True
    svm_c = 10.0


class SpeakerIndependentTPlusAuthor(SpeakerIndependentTConfig):
    use_author = True
    svm_c = 10.0


class SpeakerIndependentTAPlusContext(SpeakerIndependentTAConfig):
    use_context = True
    svm_c = 1000.0


class SpeakerIndependentTAPlusAuthor(SpeakerIndependentTAConfig):
    use_author = True
    svm_c = 1000.0


CONFIG_BY_KEY = {
    '': Config(),
    't': SpeakerDependentTConfig(),
    'a': SpeakerDependentAConfig(),
    'v': SpeakerDependentVConfig(),
    'ta': SpeakerDependentTAConfig(),
    'tv': SpeakerDependentTVConfig(),
    'av': SpeakerDependentAVConfig(),
    'tav': SpeakerDependentTAVConfig(),
    't-c': SpeakerDependentTPlusContext(),
    't-author': SpeakerDependentTPlusAuthor(),
    'tv-c': SpeakerDependentTVPlusContext(),
    'tv-author': SpeakerDependentTVPlusAuthor(),
    'i-t': SpeakerIndependentTConfig(),
    'i-a': SpeakerIndependentAConfig(),
    'i-v': SpeakerIndependentVConfig(),
    'i-ta': SpeakerIndependentTAConfig(),
    'i-tv': SpeakerIndependentTVConfig(),
    'i-av': SpeakerIndependentAVConfig(),
    'i-tav': SpeakerIndependentTAVConfig(),
    'i-t-c': SpeakerIndependentTPlusContext(),
    'i-t-author': SpeakerIndependentTPlusAuthor(),
    'i-ta-c': SpeakerIndependentTAPlusContext(),
    'i-ta-author': SpeakerIndependentTAPlusAuthor(),
}
