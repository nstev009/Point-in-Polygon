# point_in_polygon

## Description

Open source point in polygon analysis using geodata warehouse

Analysis works but it is still somewhat slow. Querying of the geodata warehouse is slow.

## Installation

Requires oracle driver. Please email the geo help desk for installation and help.
Use the `create_environment.py` and the `requirement.txt` file to create a new environment with all the packages installed.

## Usage

Uses keyring. Please set your domain, username, and password prior to using the script.

The `set_keyring.py` script can be run up your keyring authentication. Please ensure the name of the domain is the same as the one use in your code.

Test data for the program can be found at: \\fld6filer\Record_Linkage\Team\wan\Other\PiP_data

## Roadmap

Optimizing query to geodata warehouse.
Caching geodata and check if update of table has occured.

## Project status

Active development
