from decouple import Config, RepositoryEnv

DOTENV_FILE = ".env.local"
config = Config(RepositoryEnv(DOTENV_FILE))

QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"
SMOL_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
AMBER_DATASET_PATH = config("AMBER_DATASET_PATH", default="./data/amber")
EPSILON = 1e-6
BATCH_SIZE = config("BATCH_SIZE", default=4, cast=int)
MAX_SEQUENCE_LENGTH = config("MAX_SEQUENCE_LENGTH", default=1024, cast=int)
SAVE_EVERY_N_STEPS = 300 # step is when gradient is accumulated
ACCUMULATE_GRADIENT_FOR_N_SAMPLES = 16
