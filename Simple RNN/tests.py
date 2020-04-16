import pandas as pd
import json
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

with open(os.path.join(__location__, 'class_to_action.json')) as json_file:
    CtA = json.load(json_file)

def class_to_action (CtA, class_int):
    action = CtA[class_int]
    return action

prueba = class_to_action(CtA,"0")

print(prueba[0]+ prueba[1]+1)
