# test_calculator.py
import pytest, os
#from calculator import add

import os

def truthy(value):
    return bool(value)
def falsy(value):
    return not bool(value)

def check_files(directory, filename):
    # Get all items in the directory
    items = os.listdir(directory)
    
    # Loop through each item
    for item in items:
        if item.startswith('.') or item.startswith('__'):
            continue
        item_path = os.path.join(directory, item)
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Construct the path to the potential simulation.input file
            simulation_input_path = os.path.join(item_path, filename)
            
            # Check if simulation.input exists in this directory
            if not os.path.exists(simulation_input_path): 
                print(f"Could not find {filename} in {item_path}\n")
                return False
    return True

def test_file():
    current_directory = os.getcwd() + '/'
    assert  check_files(current_directory, 'simulation.input')
    assert not check_files(current_directory, 'force_field.def')
    #assert not check_files(current_directory, 'asdasd.def')