from decouple import Config, RepositoryEnv

DOTENV_FILE = ".env.local"
config = Config(RepositoryEnv(DOTENV_FILE))

QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"
SMOL_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
AMBER_DATASET_PATH = config("AMBER_DATASET_PATH", default="./data/amber")
BATCH_SIZE = config("BATCH_SIZE", default=4, cast=int)
MAX_SEQUENCE_LENGTH = config("MAX_SEQUENCE_LENGTH", default=1024, cast=int)
SAVE_EVERY_N_STEPS = config("SAVE_EVERY_N_STEPS", default=5000, cast=int) # step is finished when optimizer is called
ACCUMULATE_GRADIENT_FOR_N_SAMPLES = 16
RUN_NAME_SUFFIX = "_LIM_100_most"
EPSILON = 1e-6
INITIAL_LR = 3e-4
PERCENTAGE_OF_LAYERS_TO_QUANTIZE = 1.
