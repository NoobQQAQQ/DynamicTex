import importlib
import os

from utils.string_util import underscore2camelcase
from datasets.base_dataset import BaseDataset


def find_dataset_class_by_name(name):
    """
    Input
    name: string with underscore representation

    Output
    dataset: a dataset class with class name {camelcase(name)}Dataset

    Searches for a dataset module with name {name}_dataset in current
    directory, returns the class with name {camelcase(name)}Dataset found in
    the module.
    """

    cls_name = underscore2camelcase(name) + 'Dataset'
    pkg_name = os.path.dirname(__file__)
    filename = f"{name}_dataset"

    module = importlib.import_module("datasets." + filename)

    assert cls_name in module.__dict__, f'Cannot find dataset class named "{cls_name}" in "{filename}.py"'
    cls = module.__dict__[cls_name]

    assert issubclass(cls, BaseDataset), f'Dataset class "{cls_name}" must inherit from BaseDataset'
    return cls


def create_dataset(args):
    dataset = find_dataset_class_by_name(args.dataset_name)
    instance = dataset(args)
    print(f"dataset [{instance.name()}] was created")
    return instance
