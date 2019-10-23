# file path (load_data.py,  main.py, and compute_relation_vectors.py)
_SOURCE_DATA = '../data/source.csv'
_TARGET_DATA = '../data/target.csv'
_RESULT_FILE = '../results/results.csv'
_MEAN_RELATION = '../results/relation_vectors_before.csv'
_MODIFIED_MEAN_RELATINON = '../results/relation_vectors_after.csv'
_COUNT_RELATINON = '../results/count_ver_relation_vectors.csv'

# the number of labels
_SOURCE_DIM_NUM=9
_TARGET_DIM_NUM=2

# for learning
_FOLD_NUM = 10
_SORUCE_LATENT_TRAIN = True
_TARGET_LATENT_TRAIN = True
_DROPOUT_RATE = 0.01
_L2_REGULARIZE_RATE = 0.00001

# for setting models (models.py and main.py)
_BATCH_SIZE=32
_OUT_DIM=188

# epoch (main.py)
_SOURCE_EPOCH_NUM=5
_TARGET_EPOCH_NUM=5

# for compute_relation_vectors.py
_ITERATION = 1000
_SAMPLING = 100
_EPS=1.0e-10

# method flag (main.py, and compute_relation_vectors.py))
_SCRATCH=0
_CONV_TRANSFER=1
_COUNT_ATDL=2
_MEAN_ATDL=3
_MODIFIED_MEAN_ATDL=4
