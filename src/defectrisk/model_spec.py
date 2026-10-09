"""Frozen selected specification; no tuning or policy decisions."""

MODEL_VERSION = 'defectrisk-rf-sigmoid-v1'
RANDOM_STATE = 42
RF_PARAMETERS = {
    'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 3,
    'max_features': 'sqrt', 'class_weight': 'balanced_subsample',
    'random_state': RANDOM_STATE,
}
FEATURES = (
    'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't',
    'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
    'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount',
)
CALIBRATION_METHOD = 'sigmoid'
SIGMOID_PARAMETERS = {'C': float('inf'), 'solver': 'lbfgs', 'random_state': RANDOM_STATE,
                      'max_iter': 5000, 'class_weight': None}
TRAINING_OOF = 'docs/calibration-and-uncertainty-results/oof-probabilities.csv'
TRAINING_OOF_SHA256 = '294a2572f87a64faae280b56b0569ffad2f66a1c16a5e0a4c83f3bfe8918b217'
