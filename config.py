# Modifyed according to the implementation of Temporal ensembling:
#     https://github.com/smlaine2/tempens

#----------------------------------------------------------------------------
# Base directories.
#----------------------------------------------------------------------------

# Hostname and user.
# - Used for reporting, as well as specializing paths on a per-host basis.

import os, socket, getpass
host = socket.gethostname().lower()
user = getpass.getuser()

# Base directory for input data.

data_dir = (
    os.environ['TEMPENS_DATA_DIR'] if 'TEMPENS_DATA_DIR' in os.environ else
    'data')

# Directory for storing the results of individual training runs.

result_dir = (
    os.environ['TEMPENS_RESULT_DIR'] if 'TEMPENS_RESULT_DIR' in os.environ else
    'results')

#----------------------------------------------------------------------------
# Theano configuration.
#----------------------------------------------------------------------------

theano_flags = "device=gpu,floatX=float32,assert_no_cpu_op=warn,allow_gc=False,nvcc.fastmath=True,dnn.conv.algo_fwd=small,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic,print_active_device=0"

#----------------------------------------------------------------------------
# Training defaults.
#----------------------------------------------------------------------------

from collections import OrderedDict

run_desc                            = 'baseline'                # Name the results directory to be created for current run.
network_type                        = 'pi'                      # Valid values: 'pi', 'tempens'.
dataset                             = 'svhn'                    # Valid values: 'cifar-10', 'svhn'.
aux_tinyimg                         = None                      # Valid values: None, 'c100', # for any.
whiten_inputs                       = 'norm'                    # Valid values: None, 'norm', 'zca'.
augment_noise_stddev                = 0.15                      # Controls the Gaussian noise added inside network during training.
augment_mirror                      = False                     # Enable horizontal flip augmentation.
augment_translation                 = 2                         # Maximum translation distance for augmentation. Must be an integer.
num_labels                          = 500                       # Total number of labeled inputs (1/10th of this per class). Value 'all' uses all labels.
corruption_percentage               = 0                         # How big percentage of input labels to corrupt.
num_epochs                          = 300                       # Number of epochs to train.
max_unlabeled_per_epoch             = None                      # Set this to use at most n unlabeled inputs per epoch.
minibatch_size                      = 100                       # Samples per minibatch.
batch_normalization_momentum        = 0.999                     # Mean-only batch normalization momentum.
learning_rate_max                   = 0.003                     # Maximum learning rate.
rampup_length                       = 80                        # Ramp learning rate and unsupervised loss weight up during first n epochs.
rampdown_length                     = 50                        # Ramp learning rate and Adam beta1 down during last n epochs.
rampdown_beta1_target               = 0.5                       # Target value for Adam beta1 for rampdown.
adam_beta1                          = 0.9                       # Default value.
adam_beta2                          = 0.999                     # Default value.
adam_epsilon                        = 1e-8                      # Default value.
prediction_decay                    = 0.6                       # Ensemble prediction decay constant (\alpha in paper).
unsup_weight_max                    = 100.0                     # Unsupervised loss maximum (w_max in paper). Set to 0.0 -> supervised loss only.
load_network_filename               = None                      # Set to load a previously saved network.
start_epoch                         = 0                         # Which epoch to start training from. For continuing a previously trained network.
cuda_device_number                  = 0                         # Which GPU to use.
random_seed                         = 1000                      # Randomization seed.


# My settings
ema_decay                           = 0.001                     # a + ema_decay*(p-a)
large_net                           = True                      # Use the origial network architecture or smaller one.
coeff_entropy                       = False                     # postive numbers like 0.00001 or False. Defalut False.
consis                              = True                      # If True, add consistent loss else do not add consistent loss. Default True.

# specified settings for embed
coeff_embed                         = 0.4                       # For embed, default 1.0
merge                               = True                      # Merge the target predictions with true labels. Default True.
margin                              = 1.0                       # Default: 1.0


suffix = ''
if ema_decay != 0.001:
    suffix += '_ema{}'.format(ema_decay)
if large_net is False:
    suffix += '_small'
suffix += '_embed{}'.format(coeff_embed)
if merge is False:
    suffix += '_nomerge'
if margin != 1.0:
    suffix += '_margin{}'.format(margin)
if coeff_entropy is not False:
    suffix += '_ent{}'.format(coeff_entropy)

#----------------------------------------------------------------------------
# Individual run customizations.
#----------------------------------------------------------------------------

# MNIST: Pi.
run_desc                        = 'run-pi'
network_type                    = 'pi'
dataset                         = 'mnist'
whiten_inputs                   = 'norm'
augment_mirror                  = False
augment_translation             = 0
num_labels                      = 100
learning_rate_max               = 0.0001
unsup_weight_max                = 100.0
large_net                       = True


# MNIST: Temporal ensembling.
# run_desc                        = 'run-tempens'
# network_type                    = 'tempens'
# dataset                         = 'mnist'
# whiten_inputs                   = 'norm'
# augment_mirror                  = False
# augment_translation             = 0
# num_labels                      = 100
# learning_rate_max               = 0.0001
# unsup_weight_max                = 100.0


# SVHN: Pi.
# run_desc                        = 'run-pi_'
# network_type                    = 'pi'
# dataset                         = 'svhn'
# whiten_inputs                   = 'norm'
# augment_mirror                  = False
# augment_translation             = 2
# num_labels                      = 1000
# learning_rate_max               = 0.003
# unsup_weight_max                = 100.0


# SVHN: Temporal ensembling.
# run_desc                        = 'run-tempens_epoch150'
# network_type                    = 'tempens'
# dataset                         = 'svhn'
# whiten_inputs                   = 'norm'
# augment_mirror                  = False
# augment_translation             = 2
# num_labels                      = 'all'
# learning_rate_max               = 0.001
# unsup_weight_max                = 30.0
# num_epochs                      = 150

# CIFAR-10: Pi.
# run_desc                        = 'run-pi'
# network_type                    = 'pi'
# dataset                         = 'cifar-10'
# whiten_inputs                   = 'zca'
# augment_mirror                  = True
# augment_translation             = 2
# num_labels                      = 4000
# learning_rate_max               = 0.003
# unsup_weight_max                = 100.0


# CIFAR-10: Temporal ensembling.
# run_desc                        = 'run-tempens_current'
# network_type                    = 'tempens'
# dataset                         = 'cifar-10'
# whiten_inputs                   = 'zca'
# augment_mirror                  = True
# augment_translation             = 2
# num_labels                      = 4000
# learning_rate_max               = 0.003
# unsup_weight_max                = 30.0


# CIFAR-100: Pi.
# run_desc                        = 'run-pi'
# network_type                    = 'pi'
# dataset                         = 'cifar-100'
# whiten_inputs                   = 'zca'
# augment_mirror                  = True
# augment_translation             = 2
# num_labels                      = 10000
# learning_rate_max               = 0.003
# unsup_weight_max                = 100.0

# CIFAR-100: Temporal ensembling.
# run_desc                        = 'run-tempens'
# network_type                    = 'tempens'
# dataset                         = 'cifar-100'
# whiten_inputs                   = 'zca'
# augment_mirror                  = True
# augment_translation             = 2
# num_labels                      = 'all'
# learning_rate_max               = 0.003
# unsup_weight_max                = 100.0

# CIFAR-100 plus Tiny Images: Pi.
#run_desc                        = 'run-pi'
#network_type                    = 'pi'
#dataset                         = 'cifar-100'
#aux_tinyimg                     = 500000
#whiten_inputs                   = 'zca'
#augment_mirror                  = True
#augment_translation             = 2
#num_labels                      = 'all'
#learning_rate_max               = 0.003
#unsup_weight_max                = 300.0
#max_unlabeled_per_epoch         = 50000

# CIFAR-100 plus Tiny Images: Temporal ensembling.
# run_desc                        = 'run-tempens'
# network_type                    = 'tempens'
# dataset                         = 'cifar-100'
# aux_tinyimg                     = 500000  #'c100'
# whiten_inputs                   = 'zca'
# augment_mirror                  = True
# augment_translation             = 2
# num_labels                      = 'all'
# learning_rate_max               = 0.003
# unsup_weight_max                = 1000.0
# max_unlabeled_per_epoch         = 50000

# SVHN with label corruption: Temporal ensembling.
# run_desc                        = 'run-tempens'
# network_type                    = 'tempens'
# dataset                         = 'svhn'
# whiten_inputs                   = 'norm'
# augment_mirror                  = False
# augment_translation             = 2
# num_labels                      = 'all'
# learning_rate_max               = 0.001
# corruption_percentage           = 20
# unsup_weight_max                = 300.0 if (corruption_percentage < 50) else 3000.0
# consis                          = True

#----------------------------------------------------------------------------
# Disable mirror and translation augmentation.
#----------------------------------------------------------------------------

# if True:
#    augment_mirror = False
#    augment_translation = 0
#    run_desc = run_desc + '_noaug'

#----------------------------------------------------------------------------
# Automatically append dataset, label count, and random seed to run_desc.
#----------------------------------------------------------------------------

if consis is False:
    run_desc += '-no_consis'

if corruption_percentage != 0:
    run_desc += '-corrupt%d' % corruption_percentage

if aux_tinyimg == 'c100':
    run_desc += '-auxcif'
elif aux_tinyimg == 500000:
    run_desc += '-aux500k'
else:
    assert(aux_tinyimg is None)

if num_labels == 'all':
    num_labels_str = 'all'
elif (num_labels % 1000) == 0:
    num_labels_str = '%dk' % (num_labels / 1000)
else:
    num_labels_str = '%d' % num_labels

if dataset == 'cifar-10':
    dataset_str = 'cifar'
elif dataset == 'cifar-100':
    dataset_str = 'cifar100'
else:
    dataset_str = dataset

# run_desc = run_desc + ('_%s%s_%04d' % (dataset_str, num_labels_str, random_seed))

#----------------------------------------------------------------------------
