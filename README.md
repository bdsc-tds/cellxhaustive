# cellxhaustive

A python package for performing an exhaustive search of cell phenotypes.

## Installation

### Using `pip`

Note: not yet implemented.

```bash
pip install cellxhaustive
```

### Using `mamba`

You can install the package requirements in a dedicated `mamba` environment. To do so, follow the next steps:

```bash
git clone git@github.com:bernibra/cellxhaustive.git # Clone repository
cd cellxhaustive  # Go into cloned repository
mamba env create -f mamba_env.yaml  # Create computing environment
```

This will create a specific environment called `cellxhaustive`, that can later be activated with `mamba activate cellxhaustive`.

Note: if you don't have `mamba` installed, you can find instructions [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) explaining how to install it.

## Quickstart

To quickly check what the package does and how it works, you can use it with the provided test dataset:

```bash
cd cellxhaustive  # Go into repository
mamba activate cellxhaustive  # Activate computing environment
python cellxhaustive.py -i test/data/test_data.tsv -m test/data/test_markers.txt -o test/results/test_data_annotated.tsv  # Run analyses
```

More information on the detailed functioning of the package can be found in the [next section](#usage).

## Usage

`cellxhaustive` has a mix of mandatory and optional parameters, detailed in the next paragraphs.

### Mandatory parameters

`cellxhaustive` has 3 mandatory parameters, namely:
- `-i`, `--input INPUT_PATH`: path to the input table with expression data and samples/batch/cell_type information. Must be a string
- `-m`, `--markers MARKER_PATH`: path to the input file with list of markers of interest that will be used. Must be a string
- `-o`, `--output OUTPUT_PATH`: path to output file where data will be written. Must be a string

These parameters as well as the file contents will be explained in more details in the following sections.

#### Input file

TODO

#### Markers file

TODO

#### Output

TODO

### Optional parameters

All the following parameters are optional and are preset with default values. As such, they do not strictly need to be specified to run the analyses, but they can be very useful to tweak the analyses or reduce the computational burden.

- `-l`, `--log LOG_PATH`: path to the file containing the log messages generated by the script. By default, it is the same path than the output file, except it finishes with the `.log` extension. Must be a string
- `-ll`, `--log-level LOG_LEVEL`: verbosity level of log file. Set to `info` by default. Must be `debug`, `info` or `warning`
    - `debug` will often produce a very heavy file (>100 MB) and should be used if you are interested in the detailed progression of the analyses
    - `warning` will produce a very empty file and should be used only if you don't need the basic analysis information
- `-t`, `--two-peak-threshold TWO_PEAK_THRESHOLD`: threshold used to determine whether a marker with two-peaks is negative or positive. Set to 3 by default. Must be a positive float number
    - Expression below this threshold means the marker will be considered negative
    - Expression above this threshold means the marker will be considered positive
- `-tp`, `--three-peaks THREE_PEAK_MARKERS`: path to the file containing a list of markers that have three peaks. If specified, file must contain one marker per line. Empty by default. Must be a string
- `-tpl`, `--three-peak-low THREE_PEAK_LOW`: threshold used to determine whether a marker with three-peaks is negative or low positive. Set to 2 by default. Must be a positive float number
    - Expression below this threshold means marker will be considered negative
    - See description of `three_peak_high` for more information on low positive markers
- `-tph`, `--three-peak-high THREE_PEAK_HIGH`: threshold used to determine whether a marker with three-peaks is low positive or positive. Set to 4 by default. Must be a positive float number
    - Expression above this threshold means marker will be considered positive
    - Expression between `THREE_PEAK_LOW` and `THREE_PEAK_HIGH` means the marker will be considered low positive
- `-c`, `--cell-type-definition CELL_TYPE_PATH`: path to the file in **.json format** containing information on relationships between marker status (positive or negative) and cell type ontology. The file that is provided by default was built using data from the [rcellontologymapping](https://github.com/RGLab/rcellontologymapping) package. Must be a string
- `-mm`, `--max-markers MAX_MARKERS`: maximum number of relevant markers to select among the total list of markers. Must be less than or equal to the number of markers available in `INPUT_PATH`. Set to 15 by default. Must be a positive integer
- `-ms`, `--min-samplesxbatch MIN_SAMPLESXBATCH`: minimum proportion of samples within each batch with at least `MIN_CELLXSAMPLE` cells for a new annotation to be considered. Set to 0.5 by default. Must be a positive float number in `[0; 1]`
    - In other words, by default, an annotation needs to be assigned to at least 10 cells/sample (see description of next parameter) in at least 50% of samples within a batch to be considered
- `-mc`, `--min-cellxsample MIN_CELLXSAMPLE`: minimum number of cells within each sample in `MIN_SAMPLESXBATCH` %% of samples within each batch for a new annotation to be considered. Set to 10 by default. Must be a positive integer in `[0; 100]`
    - In other words, by default, an annotation needs to be assigned to at least 10 cells/sample in at least 50% of samples (see description of previous parameter) within a batch to be considered
- `-nk`, `--no-knn`: if present, do not refine annotations with a KNN classifier. Missing by default, which means the annotations will be improved using a KNN-classifier
    - This parameter is a boolean, which means that you do not need to write a value after calling it:
- `-knn`, `--knn-min-probability KNN_MIN_PROBABILITY`: confidence threshold for a KNN-classifier to reassign a new cell type to previously undefined cells. Set to 0.5 by default. Must be a positive float number in `[0; 1]`
    - A KNN probability below this threshold means the classifier will not reassign a new cell type to an undefined cell
    - A KNN probability above this threshold means the classifier will reassign a new cell type to an undefined cell

### Help

You can access the package help and get a summary of the parameters with:

```
python cellxhaustive.py -h
```


## Contributing

Interested in contributing? Check out the [contributing guidelines](CONTRIBUTING.md). Please note that this project is released with a [Code of Conduct](CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## License

`cellxhaustive` was created by Bernat Bramon Mora and Antonin Thiébaut. It is licensed under the terms of the [MIT license](LICENSE).

## Credits

`cellxhaustive` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
