class SpecialCharacters:
    def __init__(
        self,
        subtoken_pad_token="<PAD>",
        subtoken_unk_token="<UNK>",
        node_pad_token="<PAD>",
        node_unk_token="<UNK>",
        target_pad_token="<PAD>",
        target_sos_token="<SOS>",
        target_eos_token="<EOS>",
        target_unk_token="<UNK>",
    ):

        # Special characters for the encoder
        self.subtoken_pad_token = subtoken_pad_token
        self.subtoken_unk_token = subtoken_unk_token
        self.node_pad_token = node_pad_token
        self.node_unk_token = node_unk_token
        self.target_pad_token = target_pad_token
        self.target_sos_token = target_sos_token
        self.target_eos_token = target_eos_token
        self.target_unk_token = target_unk_token


class Vocabularies:
    def __init__(
        self,
        subtoken_to_index={},
        index_to_subtoken={},
        node_to_index={},
        index_to_node={},
        target_to_index={},
        index_to_target={},
    ):

        # dictionaries
        self.node_to_index = node_to_index
        self.index_to_node = index_to_node
        self.subtoken_to_index = subtoken_to_index
        self.index_to_subtoken = index_to_subtoken
        self.target_to_index = target_to_index
        self.index_to_target = index_to_target


class ModelConfig:
    def __init__(
        self,
        subtoken_embedding_size=128,
        subtoken_embedding_vocab_size=128,
        ast_hidden_size=128,
        ast_embedding_vocab_size=128,
        ast_embedding_size=128,
        ast_bidirectional=True,
        encoder_hidden_dim=320,
        encoder_dropout=0.25,
        decoder_hidden_dim=320,
        decoder_output_dim=128,
        decoder_embedding_vocab_size=30,
        decoder_dropout=0.25,
    ):
        # Encoder Settings
        #   subtoken encoder
        self.subtoken_embedding_size = subtoken_embedding_size
        self.subtoken_embedding_vocab_size = subtoken_embedding_vocab_size

        #   ast encoder
        self.ast_embedding_vocab_size = ast_embedding_vocab_size
        self.ast_embedding_size = ast_embedding_size
        self.ast_bidirectional = ast_bidirectional
        self.ast_hidden_size = ast_hidden_size

        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_dropout = encoder_dropout

        # Decoder Settings
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_output_dim = decoder_output_dim
        self.decoder_embedding_vocab_size = decoder_embedding_vocab_size
        self.decoder_dropout = decoder_dropout


class GeneralParams:
    def __init__(
        self,
        batch_size=256,
        max_contexts=200,
        subtoken_length=5,
        ast_path_length=9,
        target_length=5,
        train_path="",
        test_path="",
        val_path="",
    ):
        # General Settings
        self.batch_size = batch_size
        self.max_contexts = max_contexts
        self.subtoken_length = subtoken_length
        self.ast_path_length = ast_path_length
        self.target_length = target_length
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path

class Config:
    def __init__(
        self,
        vocabularies: Vocabularies,
        general_params: GeneralParams = None,
        special_characters: SpecialCharacters = None,
        model_config: ModelConfig = None,
    ):
        self.vocabularies = vocabularies
        self.general_params = general_params if general_params else GeneralParams()
        self.special_characters = (
            special_characters if special_characters else SpecialCharacters()
        )
        self.model_config = model_config if model_config else ModelConfig()
