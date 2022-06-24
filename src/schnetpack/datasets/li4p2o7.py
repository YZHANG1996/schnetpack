import logging
import os
from typing import List, Optional, Dict

from ase import Atoms

import torch
from schnetpack.data import *
from schnetpack.data import AtomsDataModuleError, AtomsDataModule
import tempfile
import shutil
import wget
import gzip
import pickle
import numpy as np
from tqdm import tqdm


__all__ = ["Li4P2O7"]


class Li4P2O7(AtomsDataModule):
    """
    Materials Project (MP) database of bulk crystals.
    This class adds convenient functions to download Materials Project data into
    pytorch.

    References:

        .. [#matproj] https://materialsproject.org/

    """

    # properties
    energy = "energy"
    forces = "forces"

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 2,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        apikey: Optional[str] = None,
        timestamp: Optional[str] = None,
        **kwargs
    ):
        """

        Args:
            datapath: path to dataset
            batch_size: (train) batch size
            num_train: number of training examples
            num_val: number of validation examples
            num_test: number of test examples
            split_file: path to npz file with data partitions
            format: dataset format
            load_properties: subset of properties to load
            val_batch_size: validation batch size. If None, use test_batch_size, then batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then batch_size.
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides num_workers).
            num_test_workers: Number of test data loader workers (overrides num_workers).
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
        """
        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            **kwargs
        )
        self.apikey = apikey
        self.timestamp = timestamp

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                Li4P2O7.energy: "eV",
                Li4P2O7.forces: "eV/Ang",
            }

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
            )

            self._download_data(dataset)
        else:
            dataset = load_dataset(self.datapath, self.format)

    def _download_data(self, dataset: BaseAtomsData):
        """
        Downloads dataset provided it does not exist in self.path
        Returns:
            works (bool): true if download succeeded or file already exists
        """
        # download data
        tmp_dir = tempfile.mkdtemp()
        file_urls = [
            f"https://files.codeocean.com/files/verified/d4a3801d-1342-4f84-8139-cf7a82fc4df3_v1.0/" \
            f"data/Structure_Database/SystemB_Li4P2O7/RAW_VASP_Structures/part_{i}_extract.pkl.gz"
            for i in range(11)
        ]
        for i, file_url in tqdm(enumerate(file_urls), "downloading data", total=11):
            response = wget.download(file_url, os.path.join(tmp_dir, f"part_{i}.pkl.gz"))

        # collect data
        atms_list = []
        properties_list = []
        atom_types = np.array([3] * 64 + [8] * 112 + [15] * 32)
        for i in tqdm(range(11), "parsing parts"):
            with gzip.open(os.path.join(tmp_dir, f"part_{i}.pkl.gz"), "rb") as file:
                data = pickle.load(file, encoding='latin1')
                for energy, forces, cell, positions in zip(data["free energy"], data["forces"], data["lattice"], data["atoms"]):
                    atms_list.append(Atoms(numbers=atom_types, positions=positions, cell=cell, pbc=True))
                    properties_list.append(dict(energy=energy, forces=forces))

        # remove tmp dir
        shutil.rmtree(tmp_dir)

        # write systems to database
        logging.info("Write atoms to db...")
        dataset.add_systems(
            atoms_list=atms_list,
            property_list=properties_list,
        )
        logging.info("Done.")


class MyUracil(AtomsDataModule):
    """
    Tutorial dataset for uracil.

    Args:
        datapath: path to dataset
        batch_size: (train) batch size

    """

    energy = "energy"
    forces = "forces"
    atomrefs = {}

    def __init__(
            self,
            datapath: str,
            batch_size: int,
            **kwargs,
    ):
        super(MyUracil, self).__init__(
            datapath=datapath,
            batch_size=batch_size,
            **kwargs,
        )

    def prepare_data(self):
        # download data if not present at location
        if not os.path.exists(self.datapath):
            # create empty dataset
            property_unit_dict = {
                MyUracil.energy: "eV",
                MyUracil.forces: "eV/Ang",
            }
            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
                atomrefs=MyUracil.atomrefs,
            )
            # download data and fill dataset
            self._download_data(dataset)

        # load dataset if data is available
        else:
            dataset = load_dataset(self.datapath, self.format)

         # checks
        if len(dataset) == 0:
            raise AtomsDataModuleError(
                f"The dataset located at {self.datapath} is empty."
            )

    def _download_data(self, dataset):
        # load raw data
        tmp_dir = tempfile.mkdtemp()
        _ = wget.download(
            "http://quantum-machine.org/gdml/data/npz/uracil_dft.npz",
            os.path.join(tmp_dir, f"uracil_dft.npz"),
        )
        data = np.load(os.path.join(tmp_dir, f"uracil_dft.npz"))
        shutil.rmtree(tmp_dir)

        # parse atoms and properties
        atoms_list = []
        property_list = []
        numbers = data["z"]
        for positions, energies, forces in tqdm(zip(data["R"], data["E"], data["F"])):
            ats = Atoms(positions=positions, numbers=numbers)
            properties = {'energy': energies, 'forces': forces}
            property_list.append(properties)
            atoms_list.append(ats)

        # write data to dataset
        dataset.add_systems(property_list, atoms_list)


if __name__ == "__main__":
    dataset = MyUracil(datapath="test.db", batch_size=10)
    dataset.prepare_data()
    print(len(dataset))