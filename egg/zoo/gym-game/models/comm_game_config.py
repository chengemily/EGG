import pdb
from typing import NamedTuple, Any, List
import numpy as np
import constants

DEFAULT_BATCH_SIZE = 30 #512
DEFAULT_VALID_BATCH_SIZE = 10
DEFAULT_NUM_EPOCHS = 100000
DEFAULT_LR = 5e-4
SAVE_MODEL = True
DEFAULT_MODEL_FILE = 'latest.pt'

DEFAULT_HIDDEN_SIZE = 16
DEFAULT_DROPOUT = 0.1
DEFAULT_FEAT_VEC_SIZE = 9#256
DEFAULT_TIME_HORIZON = 16

DEFAULT_WORLD_DIM = 1
NUM_AGENTS = 2
MAX_TEACH_ITERS = 20
MAX_TEST_ITERS = 30
NUM_TEST_ROUNDS = 1
USE_PRETRAINED_EMBEDDINGS = True
AUTO_DONE = False

TrainingConfig = NamedTuple('TrainingConfig', [
    ('num_epochs', int),
    ('learning_rate', float),
    ('load_model', bool),
    ('load_model_file', str),
    ('save_model', bool),
    ('save_model_file', str),
    ('log_image', bool),
    ('use_cuda', bool)
    ])

GameConfig = NamedTuple('GameConfig', [
    ('batch_size', int),
    ('num_agents', int),
    ('max_teach_iters', int),
    ('max_test_iters', int),
    ('num_test_rounds', int),
    ('memory_size', int),
    ('use_cuda', bool),
    ('use_pretrained_embeddings', bool),
    ('auto_done', bool)
])

# GRU boilerplate.
GRUModuleConfig = NamedTuple('GRUModuleConfig', [
    ('input_size', int),
    ('hidden_size', int),
    ('dropout', float),
    ('out_size', int)
    ])

# LSTM boilerplate.
LSTMModuleConfig = NamedTuple('LSTMModuleConfig', [
    ('embedding_dim', int),
    ('vocab_size', int),
    ('hidden_dim', int)
    ])

# FC boilerplate.
FCModuleConfig = NamedTuple('FCModuleConfig', [
    ('input_size', int),
    ('hidden_size', int),
    ('dropout', float),
    ('out_size', int)
    ])

# Main GRU
ActionModuleConfig = NamedTuple("ActionModuleConfig", [
    ('attention_input', int),
    ('recurrent_processor', GRUModuleConfig),
    ('action_decoder', FCModuleConfig),
    ('hidden_size', int),
    ('dropout', float),
    ('action_space_size', int),
    ('use_cuda', bool)
    ])

AgentModuleConfig = NamedTuple("AgentModuleConfig", [
    ('time_horizon', int),
    ('feat_vec_size', int),
    ('physical_processor', FCModuleConfig),
    ('fol_processor', LSTMModuleConfig),
    ('action_processors', dict),
    ('use_cuda', bool)
    ])

default_training_config = TrainingConfig(
        num_epochs=DEFAULT_NUM_EPOCHS,
        learning_rate=DEFAULT_LR,
        load_model=False,
        load_model_file="",
        save_model=SAVE_MODEL,
        save_model_file=DEFAULT_MODEL_FILE,
        log_image=True,
        use_cuda=False)

default_game_config = GameConfig(
        DEFAULT_BATCH_SIZE,
        NUM_AGENTS,
        MAX_TEACH_ITERS,
        MAX_TEST_ITERS,
        NUM_TEST_ROUNDS,
        DEFAULT_HIDDEN_SIZE,
        False,
        USE_PRETRAINED_EMBEDDINGS,
        AUTO_DONE
)

feat_size = 9

default_lstm_config = LSTMModuleConfig(
    embedding_dim=constants.EMBEDDING_DIM,
    vocab_size=constants.VOCAB_SIZE,
    hidden_dim=constants.LSTM_HIDDEN_DIM
)

def get_GRU_config_with_input_size(stage, role, ablation='default'):
    return GRUModuleConfig(
        input_size=constants.INPUT_SIZE[ablation][stage][role],
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT,
        out_size=constants.ACTION_SPACE_SIZE[stage][role]
    )

def get_fc_module_config_with_input_size(feat_size, out_size=constants.MAP_FEATURE_SIZE):
    return FCModuleConfig(
        input_size=feat_size,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT,
        out_size=out_size
    )

default_action_module_config = ActionModuleConfig(
        attention_input=None,
        recurrent_processor=get_GRU_config_with_input_size('teach', 'teacher'),
        action_decoder=get_fc_module_config_with_input_size(DEFAULT_HIDDEN_SIZE),
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT,
        action_space_size=constants.ACTION_SPACE_SIZE,
        use_cuda=False)

default_agent_config = AgentModuleConfig(
        time_horizon=DEFAULT_TIME_HORIZON,
        feat_vec_size=DEFAULT_FEAT_VEC_SIZE,
        physical_processor=get_fc_module_config_with_input_size(constants.MAP_FEATURE_SIZE),
        fol_processor=default_lstm_config,
        action_processors={},
        use_cuda=False)


def get_training_config(kwargs):
    return TrainingConfig(
            num_epochs=kwargs['n_epochs'] or default_training_config.num_epochs,
            learning_rate=kwargs['learning_rate'], #or default_training_config.learning_rate,
            load_model=bool(kwargs['load_model_weights']),
            load_model_file=kwargs['load_model_weights'] or default_training_config.load_model_file,
            save_model=default_training_config.save_model,
            save_model_file=kwargs['save_model_weights'] or default_training_config.save_model_file,
            log_image=default_training_config.log_image,
            use_cuda=kwargs['use_cuda'])


def get_game_config(kwargs):
    ablation = kwargs['ablation']

    if ablation == 'default':
        return GameConfig(
            batch_size=DEFAULT_BATCH_SIZE,
            num_agents=NUM_AGENTS,
            max_teach_iters=MAX_TEACH_ITERS,
            max_test_iters=MAX_TEST_ITERS,
            num_test_rounds=NUM_TEST_ROUNDS,
            memory_size=default_game_config.memory_size,
            use_cuda=kwargs['use_cuda'],
            use_pretrained_embeddings=USE_PRETRAINED_EMBEDDINGS,
            auto_done=AUTO_DONE
        )
    elif ablation == 'random':
        return GameConfig(
            batch_size=DEFAULT_BATCH_SIZE,
            num_agents=1,
            max_teach_iters=0,
            max_test_iters=MAX_TEST_ITERS,
            num_test_rounds=NUM_TEST_ROUNDS,
            memory_size=default_game_config.memory_size,
            use_cuda=kwargs['use_cuda'],
            use_pretrained_embeddings=False,
            auto_done=AUTO_DONE
        )
    elif ablation == 'student-no-teacher':
        return GameConfig(
            batch_size=DEFAULT_BATCH_SIZE,
            num_agents=1,
            max_teach_iters=0,
            max_test_iters=MAX_TEST_ITERS,
            num_test_rounds=NUM_TEST_ROUNDS,
            memory_size=default_game_config.memory_size,
            use_cuda=kwargs['use_cuda'],
            use_pretrained_embeddings=False,
            auto_done=AUTO_DONE
        )

def get_valid_game_config(kwargs):
    ablation = kwargs['ablation']

    if ablation == 'default':
        return GameConfig(
                batch_size=DEFAULT_VALID_BATCH_SIZE,
                num_agents=NUM_AGENTS,
                max_teach_iters=MAX_TEACH_ITERS,
                max_test_iters=MAX_TEST_ITERS,
                num_test_rounds=NUM_TEST_ROUNDS,
                memory_size=default_game_config.memory_size,
                use_cuda=kwargs['use_cuda'],
                use_pretrained_embeddings=USE_PRETRAINED_EMBEDDINGS,
                auto_done=AUTO_DONE
                )
    elif ablation == 'random':
        return GameConfig(
            batch_size=DEFAULT_VALID_BATCH_SIZE,
            num_agents=1,
            max_teach_iters=0,
            max_test_iters=MAX_TEST_ITERS,
            num_test_rounds=NUM_TEST_ROUNDS,
            memory_size=default_game_config.memory_size,
            use_cuda=kwargs['use_cuda'],
            use_pretrained_embeddings=False,
            auto_done=AUTO_DONE
        )
    elif ablation == 'student-no-teacher':
        return GameConfig(
            batch_size=DEFAULT_VALID_BATCH_SIZE,
            num_agents=1,
            max_teach_iters=0,
            max_test_iters=MAX_TEST_ITERS,
            num_test_rounds=NUM_TEST_ROUNDS,
            memory_size=default_game_config.memory_size,
            use_cuda=kwargs['use_cuda'],
            use_pretrained_embeddings=False,
            auto_done=AUTO_DONE
        )


def get_agent_config(kwargs):
    use_cuda = kwargs['use_cuda']
    ablation = kwargs['ablation']

    # feat_vec_size = DEFAULT_FEAT_VEC_SIZE*2
    feat_vec_size = 9

    if ablation == 'default':
        action_processors = {
            'test': {
                'student': ActionModuleConfig(
                                attention_input=DEFAULT_HIDDEN_SIZE + constants.OWN_MOVEMENT_EMBED_SIZE + constants.MAP_POS_AND_FEATURE_SIZE,
                                recurrent_processor=get_GRU_config_with_input_size('test', 'student'),
                                action_decoder=get_fc_module_config_with_input_size(DEFAULT_HIDDEN_SIZE, out_size=constants.ACTION_SPACE_SIZE['test']['student']),
                                hidden_size=DEFAULT_HIDDEN_SIZE,
                                dropout=DEFAULT_DROPOUT,
                                action_space_size=constants.ACTION_SPACE_SIZE['test']['student'],
                                use_cuda=use_cuda
                            )
            },
            'teach': {
                'teacher': ActionModuleConfig(
                                attention_input=default_action_module_config.attention_input,
                                recurrent_processor=get_GRU_config_with_input_size('teach', 'teacher'),
                                action_decoder=get_fc_module_config_with_input_size(DEFAULT_HIDDEN_SIZE, out_size=constants.ACTION_SPACE_SIZE['teach']['teacher']),
                                hidden_size=DEFAULT_HIDDEN_SIZE,
                                dropout=DEFAULT_DROPOUT,
                                action_space_size=constants.ACTION_SPACE_SIZE['teach']['teacher'],
                                use_cuda=use_cuda
                            ),
                'student': ActionModuleConfig(
                                attention_input=default_action_module_config.attention_input,
                                recurrent_processor=get_GRU_config_with_input_size('teach', 'student'),
                                action_decoder=get_fc_module_config_with_input_size(DEFAULT_HIDDEN_SIZE, out_size=constants.ACTION_SPACE_SIZE['teach']['student']),
                                hidden_size=DEFAULT_HIDDEN_SIZE,
                                dropout=DEFAULT_DROPOUT,
                                action_space_size=constants.ACTION_SPACE_SIZE['teach']['student'],
                                use_cuda=use_cuda
                            )
            }
        }
    elif ablation == 'student-no-teacher':
        action_processors = {
            'test': {
                'student': ActionModuleConfig(
                                attention_input=default_action_module_config.attention_input,
                                recurrent_processor=get_GRU_config_with_input_size('test', 'student', ablation=ablation),
                                action_decoder=get_fc_module_config_with_input_size(DEFAULT_HIDDEN_SIZE, out_size=constants.ACTION_SPACE_SIZE['test']['student']),
                                hidden_size=DEFAULT_HIDDEN_SIZE,
                                dropout=DEFAULT_DROPOUT,
                                action_space_size=constants.ACTION_SPACE_SIZE['test']['student'],
                                use_cuda=use_cuda
                            )
            }
        }

    return AgentModuleConfig(
                time_horizon=kwargs['n_timesteps'] or default_agent_config.time_horizon,
                feat_vec_size=default_agent_config.feat_vec_size,
                physical_processor=default_agent_config.physical_processor, # featurize maps
                fol_processor=default_agent_config.fol_processor,
                action_processors=action_processors, # GRU and decoder
                use_cuda=use_cuda
            )

