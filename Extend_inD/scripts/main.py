import ame_trainer # AMEnet
import Trans_trainer # DCEnet
import angelos
import contextlib
import extract_ind_traj

# It is assumed that ame_trainer.main.parser and Trans_trainer.main.parser
# have train_mode==False.
with contextlib.redirect_stdout(None):
    extract_ind_traj.main() # Pre-process raw data
    ame_trainer.main() # Evaluate the pre-trained AMEnet model
    Trans_trainer.main() # Evaluate the pre-trained DCEnet model
angelos.main('AMENet')
angelos.main('DCENet')