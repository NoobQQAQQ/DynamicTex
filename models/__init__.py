import importlib
import os

from utils.string_util import underscore2camelcase
from models.base_model import BaseModel


def find_model_class_by_name(name):
    cls_name = underscore2camelcase(name) + 'Model'
    pkg_name = os.path.dirname(__file__)
    filename = f"{name}_model"

    module = importlib.import_module('models.' + filename)

    assert cls_name in module.__dict__, f'Cannot find model class named "{cls_name}" in "{filename}.py"'
    cls = module.__dict__[cls_name]

    assert issubclass(cls, BaseModel), f'Model class "{cls_name}" must inherit from BaseModel'
    return cls


def create_model(args):
    model = find_model_class_by_name(args.model_name)
    instance = model(args)
    print(f"model [{instance.name()}] was created")
    return instance
