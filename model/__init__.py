from .heads import *
from .heads_ablation import *

heads_repo={
    # Rush Heads
    'RushH': RushHead,

    # ablation study
    'RHnoHyper': RH_no_hypernet,
    'RHnoM': RH_no_modulation,
    'RHnoP': RH_no_pos,
    'RHnoC': RH_no_complement,
    'RHnoR': RH_no_reg,
}