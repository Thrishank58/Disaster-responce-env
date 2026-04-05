import os
import asyncio
import json
import importlib.util
from openai import OpenAI
from env import DisasterEnv
from grader import grade


# 🔥 FIND BASE DIRECTORY AUTOMATICALLY
def find_base_dir():
    current = os.getcwd()

    # check current + one level down
    if os.path.exists(os.path.join(current, "tasks")):
        return current

    for folder in os.listdir(current):
        possible = os.path.join(current, folder)
        if os.path.isdir(possible) and os.path.exists(os.path.join(possible, "tasks")):
            return possible

    raise Exception("tasks folder not found")


BASE_DIR = find_base_dir()


# 🔥 SAFE MODULE LOADER
def load_module(name, relative_path):
    full_path = os.path.join(BASE_DIR, relative_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 🔥 LOAD TASKS
easy = load_module("easy", "tasks/easy.py")
medium = load_module("medium", "tasks/medium.py")
hard = load_module("hard", "tasks/hard.py")