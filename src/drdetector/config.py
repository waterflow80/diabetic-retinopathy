# CNN Configuration
BATCH_SIZE = 16
#MAX_EPOCHS_NUM = 15
MAX_EPOCHS_NUM = 2  # Just for testing purposes
FREEZE_BACKBONE = False
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
TRAIN_CLASSES_FILE = "train.csv"
TEST_CLASSES_FILE = "test.csv"
#BACKBONE = 'resnet18'
BACKBONE = 'alexnet'
MODEL_DIR = './models/'
PLOTS_DIR = './plots/'