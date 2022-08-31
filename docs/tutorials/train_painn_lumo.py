import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl
import os

lr = 1e-3

props= ['mu', 'alpha', 'homo', 'lumo',
        'gap', 'r2', 'zpve', 'U0', 
        'U', 'H', 'G', 'Cv']
prop = props[3]

qm9tut = './train_painn_' + prop
if not os.path.exists(qm9tut):
    os.makedirs(qm9tut)
    
inference_model = prop + 'best_inference_model'

# os.system('rm split.npz')

qm9data = QM9(
    os.path.join(qm9tut, 'qm9' + prop + '.db'),
    batch_size=100,
    num_train=100000,
    num_val=17748,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets(QM9.lumo, remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    property_units={QM9.lumo: 'eV'},
    num_workers=16,
    split_file=os.path.join(qm9tut, "split.npz"),
    pin_memory=True, # set to false, when not using a GPU
    load_properties=[QM9.lumo], #only load lumo property
    remove_uncharacterized = True

)
qm9data.prepare_data()
qm9data.setup()

cutoff = 5.
n_atom_basis = 128

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)

painn = spk.representation.PaiNN(
    n_atom_basis=n_atom_basis, n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)


# schnet = spk.representation.SchNet(
#     n_atom_basis=n_atom_basis, n_interactions=3,
#     radial_basis=radial_basis,
#     cutoff_fn=spk.nn.CosineCutoff(cutoff)
# )

pred_lumo = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.lumo,aggregation_mode = 'avg')

nnpot = spk.model.NeuralNetworkPotential(
    representation=painn,
    input_modules=[pairwise_distance],
    output_modules=[pred_lumo],
    postprocessors=[trn.CastTo64(), trn.AddOffsets(QM9.lumo, add_mean=True, add_atomrefs=False)]
)

output_lumo = spk.task.ModelOutput(
    name=QM9.lumo,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_lumo],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": lr}
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
    max_epochs=10000, # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=qm9data)

