VOCALSET_10 = [
    "vibrato",
    "straight",
    "belt",
    "breathy",
    "lip_trill",
    "inhaled",
    "spoken",
    "trill",
    "trillo",
    "vocal_fry",
]
PMT = ["vibrato", "straight", "trill", "trillo"]
TMT = ["vocal_fry", "belt", "breathy", "inhaled", "spoken"]

TRAIN_SINGERS = [
    "female1",
    "female3",
    "female4",
    "female5",
    "female6",
    "female7",
    "female9",
    "male1",
    "male11",
    "male2",
    "male4",
    "male6",
    "male7",
    "male8",
    "male9",
]
VAL_SINGERS = ["female2", "female8", "male3", "male5", "male10"]

GENDERS = ["female", "male"]

RAW_AUDIO_DATA_DIR = "/home/suvihaara/Documents/PhD/DATA/VocalSet/VocalSet_fixed/FULL"
# Non-overlapping 3 s embeddings (with trimming, zero-padded tails).
EMBEDDINGS_DATA_DIR = "/home/suvihaara/Documents/PhD/DATA/VocalSet/VocalSet_fixed/embeddings/3sec_nonoverlap_silencetrim/"
