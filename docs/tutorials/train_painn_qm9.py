import argparse
import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl


class BoolArg(argparse.Action):
    """
    Take an argparse argument that is either a boolean or a string and return a boolean.
    """
    def __init__(self, default=None, nargs=None, *args, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")

        # Set default
        if default is None:
            raise ValueError("Default must be set!")

        default = _arg_to_bool(default)

        super().__init__(*args, default=default, nargs='?', **kwargs)

    def __call__(self, parser, namespace, argstring, option_string):

        if argstring is not None:
            # If called with an argument, convert to bool
            argval = _arg_to_bool(argstring)
        else:
            # BoolArg will invert default option
            argval = True

        setattr(namespace, self.dest, argval)

def _arg_to_bool(arg):
    # Convert argument to boolean

    if type(arg) is bool:
        # If argument is bool, just return it
        return arg

    elif type(arg) is str:
        # If string, convert to true/false
        arg = arg.lower()
        if arg in ['true', 't', '1']:
            return True
        elif arg in ['false', 'f', '0']:
            return False
        else:
            return ValueError('Could not parse a True/False boolean')
    else:
        raise ValueError('Input must be boolean or string! {}'.format(type(arg)))


parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--nf', type=int, default=128, metavar='N',
                    help='feature_dim')
parser.add_argument('--cutoff', type=float, default=5.0, metavar='N',
                    help='cut off')
parser.add_argument('--rbf', type=int, default=20, metavar='N',
                    help='radial basis functions')
parser.add_argument('--num_block', type=int, default=3, metavar='N',
                    help='number of interaction blocks')
parser.add_argument('--property', type=str, default='homo', metavar='N',
                    help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--num_workers', type=int, default=16, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--outf', type=str, default='/mnt/nfs-mnj-hot-01/tmp/i22_yzhang/painn_new/', metavar='N',
                    help='folder to output results')
parser.add_argument('--agg_mode', type=str, default='sum', metavar='N',
                    help='aggregation of atomic predictions')
parser.add_argument('--remove_atomrefs', action=BoolArg, default=False,
                    help='remove atomrefs for H, U, U0, G')
args = parser.parse_args()


property = getattr(QM9, args.property)

qm9tut = args.outf + property + "-" + str(args.lr) + "-" + args.agg_mode
if not os.path.exists(qm9tut):
    os.makedirs(qm9tut)
inference_model = property + 'best_inference_model'


cutoff = args.cutoff
n_atom_basis = args.nf


qm9data = QM9(
    os.path.join(qm9tut, 'qm9' + property + '.db'),
    batch_size=args.batch_size,
    num_train=100000,
    num_val=17748,
    transforms=[
        trn.ASENeighborList(cutoff=args.cutoff),
        trn.RemoveOffsets(property, remove_mean=True, remove_atomrefs=args.remove_atomrefs),
        trn.CastTo32()
    ],
    # property_units = {property: 'eV'},
    # property_units= None if property == "r2" or "mu" or "alpha" or "cv" else {property: 'eV'},
    num_workers=args.num_workers,
    split_file=os.path.join(qm9tut, "split.npz"),
    pin_memory=True, # set to false, when not using a GPU
    load_properties=[property], #only load homo property
    remove_uncharacterized = True)

qm9data.prepare_data()
qm9data.setup()

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=args.rbf, cutoff=cutoff)

painn = spk.representation.PaiNN(
    n_atom_basis=n_atom_basis, n_interactions=args.num_block,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)

pred_prop = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=property, aggregation_mode=args.agg_mode)

nnpot = spk.model.NeuralNetworkPotential(
    representation=painn,
    input_modules=[pairwise_distance],
    output_modules=[pred_prop],
    postprocessors=[trn.CastTo64(), trn.AddOffsets(property, add_mean=True, add_atomrefs=False)]
)

output_prop = spk.task.ModelOutput(
    name=property,
    loss_fn=torch.nn.L1Loss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_prop],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": args.lr}
)

logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(qm9tut, inference_model),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    callbacks=callbacks,
    logger=logger,
    default_root_dir=qm9tut,
    max_epochs=100000, # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=qm9data)