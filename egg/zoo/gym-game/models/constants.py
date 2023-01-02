WORLD_DIMENSIONALITY = 2
MOVEMENT_EMBED_SIZE = 8
MAP_FEATURE_SIZE = 7 # the binary features only.
MAP_POS_AND_FEATURE_SIZE = 9
OWN_MOVEMENT_EMBED_SIZE = 3 # vx, vy, a

# FOL encoder
EMBEDDING_DIM = 9
VOCAB_SIZE = 23
LSTM_HIDDEN_DIM = 9
FOL_ENC_SIZE = 9
MAX_TEACH_ITERS = 20

# SENTENCE PRETRAINED EMBEDS
SENTENCE_EMBED_DIM = 768

INPUT_SIZE = {
    'default': {
        'teach': {
            'teacher': MAP_POS_AND_FEATURE_SIZE + SENTENCE_EMBED_DIM + MOVEMENT_EMBED_SIZE,
            'student': MAP_POS_AND_FEATURE_SIZE + MOVEMENT_EMBED_SIZE
        },
        'test': {
            'student': MAX_TEACH_ITERS
        }
    },
    'student-no-teacher': {
        'test': {
            'student': MAP_POS_AND_FEATURE_SIZE + OWN_MOVEMENT_EMBED_SIZE
        }
    }
}

ACTION_SPACE_SIZE = {
    'test': {
        'student': 4 # up down left right done
    },
    'teach': {
        'teacher': 5, # up down left right no-op
        'student': 5 # up down left right no-op
    }
}
