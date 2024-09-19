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
- ```data_path``` - No longer used, replaced with specific data paths.
- ```results_path``` - The directory to which results will be saved.
- ```callset_path``` - The path which the package will search for the "Pf7.zarr.zip" callset of Pf7.
- ```metadata_path``` - The path which the package will search for the "Pf7_samples.txt" metadata of Pf7.
- ```fws_path``` - The path which the package will search for the "Pf7_fws.txt" fws matrix of Pf7.
- ```reference_genome_path``` - The path which the package will search for the reference genome of P. falciparum.
- ```bam_paths_list_path``` - The path which the package will search for the "pf7_bam_gvcf_paths_20230823.tsv" list of BAM locations of Pf7.

- ```gff_path``` - The path which the package will search for a gff to define the regions used in the multiregion command.
- ```regions_of_focus``` - The regions in the gff which will be used by the multiregion command, e.g. "CDS" for coding regions.

- ```chromosome_prefix``` - The prefix for chromosomes within the Pf7 callset, before the chromosome number.
- ```chromosome_suffix``` - The suffix for chromosomes within the Pf7 callset, after the chromosome number.

- ```position_frequency_grouping``` - When finding the frequencies of groups with the position command, the way the data will be grouped:
  - "Population" to group by populations defined in the Pf7 paper.
  - "Country" to group by country.
- ```genotypes_list``` - The list of genotypes to include in analyses of freuqncies in the position command.
- ```low_sample_threshold``` - The threshold for the number of samples needed to include a population in the distributions generated by the population command.
- ```low_fws_sample_threshold``` - The threshold for an fws score fo a sample, for the sample to be counted as having a low (non-clonal) fws.

- ```kmer_size_limit``` - When compressing sequences, defines the size limit of the kmer that can be searched for (larger limit may allow more efficient compression, and therefore better characterisation of similar indels when grouping via compressed sequences)
- ```variant_focus``` - The variant type which will be characterised by the multiregion command.
- ```frequency_spectrum_number_plotted``` - The number of variants within a set of regions which will be plotted in the allele frequency spectrum for the multiregion command.
- ```plotted_position_genotypes``` - Which genotypes are plotted when using the position command.
- ```categories_of_interest``` - Which types of mutations to include when using the multiregion command. (e.g. "is_snp", "is_frameshift_indel", "is_not_frameshift_indel")
- ```heuristics_of_interest``` - Which heuristics to look at when using the multiregion command, in trying to detect erroneous mutations. (e.g. "altlen", "hom_frequency", "het_frequency", "het_over_hom_frequency", "clonal_hom_frequency", "clonal_het_frequency", "clonal_het_over_hom_frequency", "gc_content", "allele_depths_std")

- ```upstream_window_length``` - The number of bases upstream of a given mutation to include as contextual sequence.
- ```downstream_window_length``` - The number of bases downstream of a given mutation to include as contextual sequence.
  
- ```drop_hard_clipped_reads``` - When considering the reads at a position, whether to include hard clipped reads.
- ```drop_soft_clipped_reads``` - When considering the reads at a position, whether to include soft clipped reads.

## Usage
### Multiregion command
~~~
python interface.py multiregion
python interface.py multiregion [chromomsome]:[start position]-[end position] [etc...]
~~~
-e.g. python interface.py multiregion 8:547896-548199 8:548336-548511 8:550381-550500 8:550617-551057 - For analysis of the noncoding regions of dhps.
This command will characterise and analyse the regions given, if no regions are given the gff file given in the config file will be used to choose regions instead.

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
