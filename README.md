# cellxhaustive

A python package for performing an exhaustive search of cell phenotypes.

## Installation

### Using `pip` (recommended)

You can install `cellxhaustive` from TestPyPI (for now):

```bash
pip install -i https://test.pypi.org/simple/ cellxhaustive --extra-index-url https://pypi.python.org/simple
```

> [!NOTE]
> We recommend to install the package in a dedicated environment, created with [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or [conda/mamba](https://github.com/conda-forge/miniforge) to avoid dependency conflicts with other packages you may have installed.

### Using `mamba`

You can install the package requirements in a dedicated `conda`/`mamba` environment. To do so, follow the next steps:

```bash
git clone git@github.com:bernibra/cellxhaustive.git # Clone repository
cd cellxhaustive  # Go into cloned repository
mamba env create -f mamba_env.yaml  # Create computing environment
```

This will create a specific environment called `cellxhaustive`, that can later be activated with `mamba activate cellxhaustive`.

> [!TIP]
> If you don't have `mamba` installed, you can find instructions [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) explaining how to install it.

## Quickstart

To quickly check what the package does and how it works, you can use it with the provided test dataset:

- If you installed the package with `pip`:

    ```bash
    cd cellxhaustive  # Go into repository
    cellxhaustive -i test/data/test_data.tsv -m test/data/test_markers.txt -o test/results/test_data_annotated.tsv  # Run analysis
    ```

- If you installed the package with `mamba`:

    ```bash
    cd cellxhaustive  # Go into repository
    mamba activate cellxhaustive  # Activate computing environment
    python src/cellxhaustive/cellxhaustive.py -i test/data/test_data.tsv -m test/data/test_markers.txt -o test/results/test_data_annotated.tsv  # Run analysis
    ```

More information on the detailed functioning of the package can be found in the [next section](#usage).

## Usage

`cellxhaustive` is run from the command line with the syntax `cellxhaustive [options]`.

Example:

 ```bash
cellxhaustive -i data.tsv -m markers.txt -o results/results.tsv --log logs/results.log -a 9 -e CD4 -q 0.9 -r 10 -t 4  # PyPi install
python src/cellxhaustive/cellxhaustive.py -i data.tsv -m markers.txt -o results/results.tsv --log logs/results.log -a 9 -e CD4 -q 0.9 -r 10 -t 4  # Conda/mamba install
 ```

`cellxhaustive` has 3 mandatory parameters and several optional ones that can be used to tweak the analyses. They are described below.

### Mandatory parameters

`cellxhaustive` requires 3 parameters to run, namely:
- `-i`, `--input INPUT_PATH`: path to the input table with expression data and samples/batch/cell_type information. Must be a string
- `-m`, `--markers MARKER_PATH`: path to the input file with list of markers of interest that will be used during the analyses. Must be a string
- `-o`, `--output OUTPUT_PATH`: path to the output file where results will be written. Must be a string

The file contents and formats will be detailed in the following sections.

#### Input file

The input file should be a `Tab-Separated Values` (`.tsv`) table with:
- Cells as rows
    - First row must be the **header row** containing the column names
    - Other rows must contain **markers expression**, with one row per cell. Values should preferably be normalized
- Markers and metadata as columns
    - **Marker name**, with one column per marker
    - **Metadata**: `sample` and `batch`
        - A `batch` may contain multiple `samples`, but each `sample` must belong to only one `batch`
        - You can also add an optional `cell_type` column containing prior information on cell populations (for example, if the cells are already split into sub-populations)

Example:

| | CD3 | CD4 | CD8 | sample | batch | cell_type (optional) |
| --- | --- | --- | --- | --- | --- | --- |
| Cell 1| 1.4 | 3.4 | 3.2 | sample1 | batch1 | CD4 |
| Cell 2 | 6.0 | 4.4 | 0.2 | sample1 | batch1 | CD8 |
| Cell 3 | 4.3 | 2.4 | 1.9 | sample2 | batch1 | CD4 |

#### Markers file

The markers file should be a `.txt` file listing marker names (one per line).
Theses markers should be present as column headers in the `input file`.

Example:

```txt
CD3
CD4
CD8
```

#### Output file

The output file is a `Tab-Separated Values` (`.tsv`) table which includes:
- All the columns from the input file
- If one or more marker combinations were found by `cellxhaustive`, new columns:
    - `Annotations_i`: predicted cell type for the _i-th_ marker combination
    - `Phenotypes_i`: predicted marker phenotype derived from `Annotations_i`
    - If KNN re-assignment is enabled:
        - `KNN_annotations_i`: cell type prediction after KNN re-classification for the _i-th_ marker combination
        - `KNN_phenotype_i`: marker phenotype after KNN re-classification derived from `KNN_annotations_i`
        - `KNN_proba_i`: probability associated with the KNN prediction

### Optional parameters

All the following parameters are optional and are preset with default values. As such, they do not need to be specified to run the analyses, but they can be very useful to tweak the analyses or reduce the computational burden.

- `-a`, `--max-markers MAX_MARKERS`: maximum number of relevant markers to select among the total list of markers. Must be less than or equal to the number of markers available in `INPUT_PATH`. Set to 15 by default. Must be a positive integer
- `-mi`, `--markers-interest MARKERS_INTEREST`: comma-separated list of markers of interest that must appear in the final combination (_i.e._ each resulting marker combination will include all specified markers). Global setting that applies to all cell types. Empty by default. Can also be specified in a config file allowing tuning at cell type level (see `-b` parameter below)
- `-dm`, `--detection-method 'auto'|INT`: method used to determine the length of the best marker combination. Can be:
  - `'auto'`: use the default heuristic algorithm (default). Global setting that applies to all cell types
  - Integer: set a fixed combination length manually. Must be less than `MAX_MARKERS`. Global setting that applies to all cell types
  - Can also be specified in a config file allowing tuning at cell type level (see `-b` parameter below)
- `-c`, `--config CONFIG`: path to a config file in [`.yaml`](https://yaml.org/) format that provides cell type–specific detection methods and `MARKER_INTEREST` settings. Empty by default. Example of configuration:
    ```yaml
    # Use 'all' or specific cell types as in the 'cell_type' column of the input file
    markers_interest:
        # Parameters for all cell populations
        all:
            - CD3
            - CD4
            - CD8
            - CD19
            - CD20
            - CD45RA
        # Cell populations specific parameters
        # CD4T:
        #     - CD3
        #     - CD4
        #     - CD8
        # CD8T:
        #     - CD3
        #     - CD4
        #     - CD8

    detection_method:
        # Parameters for all cell populations
        all: 'auto'
        # Cell populations specific parameters
        # CD4T: 5
        # CD8T: 7
    ```
- `-b`, `--cell-type-definition CELL_TYPE_PATH`: path to the `.yaml` file specifying relationships between marker status and cell type ontology. Default configuration is shipped with the package (file can be found [here](`https://github.com/bdsc-tds/cellxhaustive/blob/main/src/cellxhaustive/config/major_cell_types.yaml`))
- `-l`, `--log LOG_PATH`: path to the file containing the log messages generated by the package. By default, it is the same path than the output file but with the `.log` extension. Must be a string
- `-n`, `--log-level LOG_LEVEL`: verbosity level of log file. Must be `debug`, `info` or `warning`. `info` by default
    - `warning` will often produce a very empty file containing only warnings and should only be used if you don't need any information about the analysis
    - `info` will provide a good summary of the analysis and is a good compromise between verbosity and file size
    - `debug` will often produce a very heavy file (>100 MB) and should only be used if you are interested in the detailed progression of the analysis
- `-e`, `--three-peaks THREE_PEAK_MARKERS`: path to the file containing a list of markers that have three peaks or comma separated list of markers that have three peaks. If specified, file must contain one marker per line. Empty by default
- `-j`, `--thresholds THRESHOLDS`: comma seperated list of 3 float numbers defining thresholds to determine whether markers are negative or positive.
    - 1st number is for two peaks markers. Set to 3 by default. Expression **below** this threshold means the marker will be considered **negative**. Expression **above** this threshold means the marker will be considered **positive**
    - Last 2 numbers are for three peaks markers. Set to 2 and 4 by default. Expression **below** the **first threshold** means the marker will be considered **negative**. Expression **above** the **second threshold** means the marker will be considered **positive**. Expression in **between the thresholds** means the marker will be considered **low positive**
- `-q`, `--min-samplesxbatch MIN_SAMPLESXBATCH`: minimum proportion of samples within each batch with at least `MIN_CELLXSAMPLE` cells for a new annotation to be considered. Set to 0.5 by default. Must be a positive float number in `[0.01; 1]`
    - In other words, by default, an annotation needs to be assigned to at least 10 cells/sample (see description of next parameter) in at least 50% of samples within a batch to be considered
- `-r`, `--min-cellxsample MIN_CELLXSAMPLE`: minimum number of cells within each sample in `MIN_SAMPLESXBATCH` proportion of samples within each batch for a new annotation to be considered. Set to 10 by default. Must be a positive integer in `[1; 100]`
    - In other words, by default, an annotation needs to be assigned to at least 10 cells/sample in at least 50% of samples (see description of previous parameter) within a batch to be considered
- `-p`, `--knn-min-probability KNN_MIN_PROBABILITY`: confidence threshold for a KNN-classifier to reassign a new cell type to previously undefined cells. Set to 0.5 by default. Must be a positive float number in `[0; 1]`
    - A KNN probability **below** this threshold means the classifier **will not** reassign a new cell type to an undefined cell
    - A KNN probability **above** this threshold means the classifier **will** reassign a new cell type to an undefined cell
    - Set this parameter to **0** to disable KNN re-assignment
- `-t`, `--threads THREADS`: number of CPU cores to use. Using more than one core allows parallel processing. `1` by default. Must be a strictly positive integer
- `-d`, `--dry-run`: boolean. If present, activate dry-run mode to check all inputs and configuration without running the full analysis

### Help

You can access the package help and get a summary of the parameters:

- If you installed the package with `pip`:

    ```bash
    cellxhaustive -h  # Display help
    ```

- If you installed the package with `mamba`:

    ```bash
    cd cellxhaustive  # Go into repository
    mamba activate cellxhaustive  # Activate computing environment
    python src/cellxhaustive/cellxhaustive.py -h  # Display help
    ```

## Contributing

Interested in contributing? Check out the [contributing guidelines](CONTRIBUTING.md). Please note that this project is released with a [Code of Conduct](CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## License

`cellxhaustive` was created by Bernat Bramon Mora and Antonin Thiébaut. It is licensed under the terms of the [MIT license](LICENSE).

## Credits

`cellxhaustive` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
