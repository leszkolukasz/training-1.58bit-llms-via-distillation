from decouple import Config, RepositoryEnv

DOTENV_FILE = ".env.local"
config = Config(RepositoryEnv(DOTENV_FILE))

QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"
SMOL_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
AMBER_DATASET_PATH = config("AMBER_DATASET_PATH", default="./data/amber_small_100000")
BATCH_SIZE = config("BATCH_SIZE", default=4, cast=int)
MAX_SEQUENCE_LENGTH = config("MAX_SEQUENCE_LENGTH", default=1024, cast=int)
SAVE_EVERY_N_STEPS = config(
    "SAVE_EVERY_N_STEPS", default=2000, cast=int
)  # step is finished when optimizer is called
ACCUMULATE_GRADIENT_FOR_N_SAMPLES = 16
RUN_NAME_SUFFIX = ""
EPSILON = 1e-6
INITIAL_LR = 1e-3
PERCENTAGE_OF_LAYERS_TO_QUANTIZE = 0.25

EXPERIMENT_NAME = "nlp_project"
TRACKING_URI = config("TRACKING_URI", default="file:mlruns")

ORG_NAME = "nlp-project-uw"
HF_USERNAME = config("HF_USERNAME", default=None)
HF_TOKEN = config("HF_TOKEN", default=None)

# Evaluation
HF_MODEL_CKPT = "./mlruns/770830031765675480/ed074b3860e9461e862fda19c0b4a880/checkpoints/epoch=2-step=17999.ckpt"
HF_CONVERTED_OUT_DIR = "./data/hf_converted"
HF_QUANTIZATION = "1_58b"
HF_BITLINEAR_IMPL = "FBI"
HF_LOSS_FUNCTION = "CrossEntropy"
HF_MODEL_NAME = f"quant_{HF_QUANTIZATION}_impl_{HF_BITLINEAR_IMPL}_loss_{HF_LOSS_FUNCTION}"
HARNESS_TASK = "mathqa"
BENCHMARK_OUTPUT_FILE = f"./data/benchmarks/{HF_MODEL_NAME}_{HARNESS_TASK}.pkl"