# Indel-Insight
A tool for characterising and analysing the indels within the Pf7 dataset from MalariaGEN.

## Environment
Requires:
- numpy
- pandas
- pysam
- json
- zarr
- matplotlib
- sklearn

## Configuration
A config file is required for usage, an example file is included but must be customised before use.

## Usage
### Multiregion command
~~~
python interface.py multiregion [chromomsome]:[start position]-[end position] [etc...]
~~~
-e.g. python interface.py multiregion 8:547896-548199 8:548336-548511 8:550381-550500 8:550617-551057 - For analysis of the noncoding regions of dhps.

### Position command
~~~
python interface.py position [chromosome]:[position]
~~~
- e.g. python interface.py position 13:1726571 - For analysis of indel within kelch13
- This command allows for in depth analysis of a particular position, and will generate a graph of variants at that position and their distribution in space and time.

### Bams command
~~~
python interface.py bams [chromosome]:[position]
~~~
- For the given position, gives a summary of the reads which are used to call the genotype at that position in Pf7.
