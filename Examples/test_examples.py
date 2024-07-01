# test_calculator.py
import pytest, os, re
#from calculator import add

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

def check_energy_drift(relative_file_path):
    
    finish = False
    try:
        # Construct the absolute file path using the relative path
        file_path = os.path.join(os.getcwd(), relative_file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))

        with open(file_path, 'r') as file:
            content = file.read()

        # Search for "ENERGY DRIFT" pattern
        energy_drift_pattern = re.compile(r'ENERGY DRIFT')
        total_energy_pattern = re.compile(r'Total Energy:\s+([-+]?[0-9]*\.?[0-9]+)')

        matches = list(energy_drift_pattern.finditer(content))
        if not matches:
            return False

        for match in matches:
            # Search for the nearest next match for pattern "Total Energy:"
            search_start = match.end()
            total_energy_match = total_energy_pattern.search(content, search_start)
            if total_energy_match:
                # Read the first float value in that line
                energy_value = float(total_energy_match.group(1))
                print(f"This simulation in {folder_name} folder has a total energy drift of {energy_value} (internal unit, 10 J/mol)\n")
                if energy_value < 1e-3:
                    finish = True

        # Search for "Work took" and "seconds" in the same line
        work_took_pattern = re.compile(r'Work took\s+([-+]?[0-9]*\.?[0-9]+)\s+seconds')
        work_took_match = work_took_pattern.search(content)
        if work_took_match:
            time_taken = float(work_took_match.group(1))
            print(f"This simulation in {folder_name} folder took {time_taken} seconds")
        else:
            print(f"There is something wrong with the {folder_name} folder")

        return finish

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def test_file():
    current_directory = os.getcwd() + '/'
    assert  check_files(current_directory, 'simulation.input')
    assert  os.path.exists(f"{current_directory}/Tail-Correction/force_field.def")
    paths = ['CO2-MFI',
             'Methane-TMMC',
             'Bae-Mixture',
             'NU2000-pX-LinkerRotations',
             'Tail-Correction']
    for path in paths:
        print(f"checking {path}/output.txt\n")
        assert  check_energy_drift(f"{path}/output.txt")
    #assert not check_files(current_directory, 'asdasd.def')
