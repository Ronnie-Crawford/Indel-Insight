import sys
sys.path.append("./../../../env")

import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyranges as pr
from sklearn.decomposition import PCA

from region import Region
from config import Config

class Multiregion:
    
    def __init__(self, *regions: Region, config: Config):
        
        """
        Initializes the Multiregion object, sets configuration values, and processes regions.

        Parameters:
        regions (str): Variable length region strings.
        config (Config): A Config object containing configuration settings.
        """
    
        self.set_config_values(config)
        self.callset = self.read_callset(self.callset_path)
        self.genotypes_df, self.variants_df, self.metadata_df, self.allele_depths_df = self.iterate_regions(regions, config, self.callset)
        self.frequencies_df = self.find_frequencies(self.genotypes_df)
        self.fws_df = self.read_fws(self.fws_path)
        self.allele_list = self.generate_allele_list(self.variants_df, self.frequencies_df)
        
    def plot_plots(self) -> None:

        """
        Generates and saves plots for the regions.
        """
        
        self.plot_allele_frequency_spectrum(self.allele_list)
        self.plot_heuristics(self.allele_list)
        #self.plot_heurisitcs_pca(self.allele_list)
        
    def save_lists(self) -> None:

        """
        Saves lists of processed data for the regions.
        """
        
        self.allele_list.to_csv(f"{self.results_path}/allele_list.tsv", sep='\t')
        
    def set_config_values(self, config: Config) -> None:
        
        """
        Reads values from the config object and sets them as attributes of the Multiregion object.

        Parameters:
        config (Config): A Config object containing configuration settings.
        """
        
        print("Setting properties from config...")
        
        for attribute in vars(config).keys():
            
            setattr(self, attribute, getattr(config, attribute))
            
    def read_callset(self, callset_path: str) -> zarr.hierarchy.Group:
        
        """
        Reads the callset from the specified path and loads it into memory.

        Parameters:
        callset_path (str): The path to the callset file.

        Returns:
        zarr.hierarchy.Group: The loaded callset.
        """
        
        print("Reading callset...")
        
        return zarr.open(callset_path, "r")
            
    def iterate_regions(self, regions: list, config: Config, callset: zarr.hierarchy.Group) -> pd.DataFrame:

        """
        Iterates through the provided regions, processing each one to extract genotypic, variant, and allele depth data.

        Parameters:
        regions (list): List of regions to process.
        config (Config): A Config object containing configuration settings.
        callset (zarr.hierarchy.Group): The loaded callset.

        Returns:
        tuple: DataFrames containing genotypic, variant, metadata, and allele depth data.
        """
        
        genotypes_df = pd.DataFrame()
        variants_df = pd.DataFrame()
        allele_depths_df = pd.DataFrame()
        metadata_df = pd.DataFrame()
        
        for region in regions:
            
            print(f'\nFinding alleles in region {regions.index(region) + 1} of {len(regions)}')
        
            chromosome = region.split(':')[0]
            coordinates = region.split(':')[1]
            start = int(coordinates.split('-')[0])
            end = int(coordinates.split('-')[1])
            
            region_instance = Region(chromosome, start, end, config, callset)
            genotypes_df = pd.concat([genotypes_df, region_instance.genotypes_df]).reindex()
            variants_df = pd.concat([variants_df, region_instance.variants_df]).reindex()
            allele_depths_df = pd.concat([allele_depths_df, region_instance.allele_depths_df]).reindex()
            
            if metadata_df.empty:
                
                metadata_df = region_instance.metadata_df
                
            print('\n--------------------------------------')
            
        return genotypes_df, variants_df, metadata_df, allele_depths_df
    
    def find_frequencies(self, genotypes_df: pd.DataFrame) -> pd.DataFrame:
        
        """
        Counts the number of each genotype at each position.

        Parameters:
        genotypes_df (pd.DataFrame): DataFrame containing genotypic data.

        Returns:
        pd.DataFrame: DataFrame containing frequencies of each genotype.
        """
        
        print("Finding frequencies of genotypes...")
        
        frequencies_df = genotypes_df.astype(str).apply(lambda position : position.value_counts(), axis = 1)
        frequencies_df['total_count'] = frequencies_df.sum(axis = 1)
        frequencies_df['total_nonmissing_count'] = frequencies_df.drop(['[-1, -1]', 'total_count'], axis = 1).sum(axis = 1)
        frequencies_df['missing_frequency'] = frequencies_df['[-1, -1]'] / frequencies_df['total_count']
        frequencies_df = frequencies_df.reindex(frequencies_df.columns.union(self.genotypes_list, sort=False), axis=1).fillna(0)
        
        return frequencies_df
    
    def read_fws(self, fws_path) -> pd.DataFrame:

        """
        Reads the Fws data from the specified path.

        Parameters:
        fws_path (str): The path to the Fws file.

        Returns:
        pd.DataFrame: DataFrame containing Fws data.
        """
        
        return pd.read_csv(fws_path, sep="\t")
        
    def generate_allele_list(self, variants_df: pd.DataFrame, frequencies_df: pd.DataFrame) -> pd.DataFrame:
        
        """
        Reformats frequencies data as a list of all alleles within the given regions and adds heuristics for each allele.

        Parameters:
        variants_df (pd.DataFrame): DataFrame containing variant data.
        frequencies_df (pd.DataFrame): DataFrame containing frequencies of each genotype.

        Returns:
        pd.DataFrame: DataFrame containing allele list with added heuristics.
        """
        
        print("Generating allele list...")
        
        # Generate list of reference alles for each position
        reference_list = variants_df.drop(['FILTER_PASS', 'ALT'], axis=1)
        reference_list['ALT'] = reference_list['REF']
        reference_list['allele_index'] = 0
        
        # Generate list of alternative alleles for each position
        alternate_list = variants_df.drop(['FILTER_PASS'], axis=1)
        alternate_list['allele_index'] = alternate_list.apply(lambda x: [1, 2, 3, 4, 5, 6], axis=1)
        #alternate_list['ALT'] = alternate_list['ALT'].apply(lambda x: x[2:-2].split("', '"))
        alternate_list = alternate_list.explode(['ALT', 'allele_index'])
        
        # Join reference and alternative lists into list of all alleles
        allele_list = pd.concat([alternate_list, reference_list])
        allele_list = allele_list[allele_list['ALT'] != '']
        allele_list['type'] = allele_list.apply(lambda allele: self.find_allele_type(allele), axis=1)
        allele_list["is_reference"] = np.where(allele_list["type"] == "reference", True, False)
        allele_list["is_snp"] = np.where(allele_list["type"] == "snp", True, False)
        
        # Apply frequencies to allele list
        allele_list = allele_list.merge(frequencies_df, right_index = True, left_index = True).fillna(0)
        allele_list['hom_count'] = allele_list.apply(lambda allele : self.count_homozygous(allele), axis = 1)
        allele_list['het_count'] = allele_list.apply(lambda allele : self.count_heterozygous(allele), axis = 1)
        allele_list['hom_frequency'] = allele_list['hom_count'] / allele_list['total_nonmissing_count']
        allele_list['het_frequency'] = allele_list['het_count'] / allele_list['total_nonmissing_count']
        allele_list["het_over_hom_frequency"] = allele_list['het_frequency'] / allele_list['hom_frequency']
        allele_list['allele_count'] = allele_list['hom_count'] + allele_list['het_count']
        allele_list['allele_frequency'] = (allele_list['het_frequency'] / 2) + allele_list['hom_frequency']
        
        # Frameshift heuristic
        allele_list['altlen'] = allele_list.apply(lambda allele : len(allele['ALT']) - len(allele['REF']), axis = 1)
        allele_list['is_indel'] = np.where(allele_list["altlen"] != 0, True, False)
        allele_list["is_frameshift_indel"] = np.where(allele_list["altlen"] % 3 != 0, True, False)
        allele_list["is_not_frameshift_indel"] = np.where((allele_list["is_indel"] == True) & (allele_list["is_frameshift_indel"] == False), True, False)
        #allele_list["is_synonymous"] = allele_list.apply(lambda allele : self.check_synonymous(allele), axis = 1)
        
        # GC content
        allele_list = self.find_contextual_sequence(allele_list, self.upstream_window_length, self.downstream_window_length, self.reference_genome_path)
        allele_list['contextual_sequence'] = allele_list['upstream_reference'] + allele_list['ALT'] + allele_list['downstream_reference']
        allele_list["gc_content"] = allele_list.apply(lambda allele : self.find_gc_content(allele['contextual_sequence']), axis = 1)
        
        # Homopolymer heuristic
        allele_list['depolymered_reference'] = (allele_list['upstream_reference'] + allele_list['REF'] + allele_list['downstream_reference']).apply(lambda reference : self.depolymer_sequence(reference))
        allele_list['depolymered_alternative'] = (allele_list['upstream_reference'] + allele_list['ALT'] + allele_list['downstream_reference']).apply(lambda alternative : self.depolymer_sequence(alternative))
        allele_list['is_homopolymer_expansion'] = np.where((allele_list["depolymered_reference"] == allele_list["depolymered_alternative"]) & (allele_list["type"] != "reference") & (allele_list["type"] != "snp"), True, False)
        
        # STR heuristic
        allele_list['compressed_reference_sequence'] = (allele_list['upstream_reference'] + allele_list['REF'] + allele_list['downstream_reference']).apply(lambda sequence : self.compress_sequence(sequence))
        allele_list['compressed_alternative_sequence'] = (allele_list['upstream_reference'] + allele_list['ALT'] + allele_list['downstream_reference']).apply(lambda sequence : self.compress_sequence(sequence))
        allele_list['is_str'] = allele_list.apply(lambda allele: self.find_strs(allele), axis = 1)
        
        # Clonal frequencies
        allele_list[["clonal_hom_frequency", "clonal_het_frequency", "total_nonmissing_count_clonal"]] = self.find_frequencies_in_clonals(self.genotypes_df, self.fws_df, self.clonal_sample_fws_threshold, allele_list)
        allele_list["clonal_het_over_hom_frequency"] =  allele_list["clonal_het_frequency"] / allele_list["clonal_hom_frequency"]
        
        # Allele depths
        allele_list = self.find_allele_depths(self.allele_depths_df, allele_list)
        allele_list = self.find_clonal_allele_depths(self.allele_depths_df, self.fws_df, self.genotypes_df, allele_list, self.low_fws_sample_threshold)
        
        return allele_list
    
    def find_allele_type(self, allele: pd.Series) -> str:
        
        """
        Identifies the type of an allele based on the relationship between alternate and reference sequences.

        Parameters:
        allele (pd.Series): A series representing an allele.

        Returns:
        str: The type of the allele.
        """
        
        if allele['ALT'] == "*":
            
            return "spanning_deletion"
        
        elif len(allele['REF']) < len(allele['ALT']):
            
            return "insertion"
        
        elif len(allele['REF']) > len(allele['ALT']):
            
            return "deletion"
        
        elif allele['REF'] == allele['ALT']:
            
            return "reference"
        
        elif len(allele['REF']) == 1:
            
            return "snp"
        
        else:
            
            return "hidden_snp"
        
    def count_homozygous(self, allele: pd.Series) -> pd.Series:
        
        """
        For a given allele, extracts the number of homozygous calls of that allele from the data.

        Parameters:
        allele (pd.Series): A series representing an allele.

        Returns:
        int: The count of homozygous calls.
        """
    
        return allele[f'[{allele.allele_index}, {allele.allele_index}]']
        
    def count_heterozygous(self, allele: pd.Series) -> pd.Series:
        
        """
        For a given allele, extracts the number of heterozygous calls in which that allele appears.

        Parameters:
        allele (pd.Series): A series representing an allele.

        Returns:
        int: The count of heterozygous calls.
        """
        
        het_columns = [column for column in allele.index if (column.count(str(allele.allele_index)) == 1)]
        return allele[het_columns].sum()
    
    def depolymer_sequence(self, sequence: str) -> str:
        
        """
        Given a sequence, reduces consecutive repeats of bases to single bases.

        Parameters:
        sequence (str): The input sequence.

        Returns:
        str: The depolymerized sequence.
        """
        
        return ''.join([sequence[i] for i in range(len(sequence)) if (i==0) | (sequence[i] != sequence[i-1])])
    
    def compress_sequence(self, sequence: str) -> str:
        
        """
        Compresses repeats of kmers in a sequence to a compressed representation.

        Parameters:
        sequence (str): The input sequence.

        Returns:
        str: The compressed sequence.
        """
        
        if sequence == "" : return sequence

        sequence = sequence.lower()
        
        kmers = self.iterate_possible_kmers(sequence)
        best_kmer = self.find_best_kmer(kmers)
        
        if best_kmer[1] not in sequence: return sequence
        
        prefix = sequence[: best_kmer[0]]
        suffix = sequence[((best_kmer[0] + len(best_kmer[1]) * best_kmer[2])) :]
        
        return f"{self.compress_sequence(prefix)}[{best_kmer[1]}:{best_kmer[2]}]{self.compress_sequence(suffix)}"        

    def iterate_possible_kmers(self, sequence: str) -> list:
        
        """
        Iterates through unique kmers in the given sequence, finding the number of times they occur consecutively.

        Parameters:
        sequence (str): The input sequence.

        Returns:
        list: A list of kmers and their counts.
        """
        
        kmers = []
        
        if len(sequence) == 1 : return [(0, sequence, 1)]
            
        if len(sequence) > self.kmer_size_limit: kmer_length_max = self.kmer_size_limit
        else: kmer_length_max = len(sequence)
        
        for kmer_length in range(1, kmer_length_max + 1) :
                    
            for sequence_position in range(kmer_length, len(sequence) + 1) :
                    
                kmer = "".join(sequence[(sequence_position - kmer_length) : sequence_position])
                kmer_count = 1
                
                while kmer == "".join(sequence[(sequence_position - kmer_length) + (kmer_count * kmer_length): sequence_position + (kmer_count * kmer_length)]):
            
                    kmer_count += 1
                        
                kmers.append(((sequence_position - kmer_length), kmer, kmer_count))
                        
        return kmers

    def find_best_kmer(self, kmers: list) -> tuple:
        
        """
        Identifies the longest kmer with the most repeats that compresses the sequence the most.

        Parameters:
        kmers (list): A list of kmers and their counts.

        Returns:
        tuple: The best kmer and its count.
        """
        
        best_kmer = kmers[0]
        
        if len(kmers) == 1 : return best_kmer
        
        for kmer_index in range(0, len(kmers)):
            
            if kmers[kmer_index][2] > best_kmer[2] :
                
                best_kmer = kmers[kmer_index]
                
            if kmers[kmer_index][2] == best_kmer[2] and len(kmers[kmer_index][1]) > len(best_kmer[1]) :
                
                best_kmer = kmers[kmer_index]       
                
        return best_kmer
    
    def find_score_compression(self, allele: pd.Series) -> float:

        """
        Calculates the compression score of the contextual sequence.

        Parameters:
        allele (pd.Series): A series representing an allele.

        Returns:
        float: The compression score.
        """
        
        sequence = allele['compressed_contextual_sequence'][1:-1]
        units = sequence.split('][')
        number_of_units = len(units)
        most_repetitions = max(units[:][-1])
        
        return len(most_repetitions) / number_of_units
    
    def find_strs(self, allele: pd.Series) -> bool:

        """
        Identifies if an allele is a short tandem repeat (STR) by comparing compressed reference to compressed alternative allele.

        Parameters:
        allele (pd.Series): A series representing an allele.

        Returns:
        bool: True if the allele is an STR, False otherwise.
        """
        
        reference_representation = [element for element in allele["compressed_reference_sequence"] if not element.isdigit()]
        alternate_representation = [element for element in allele["compressed_alternative_sequence"] if not element.isdigit()]
        
        if (reference_representation == alternate_representation) & (allele["type"] != "reference") & (allele["type"] != "snp"): return True
        else: return False
    
    def find_frequencies_in_clonals(self, genotypes_df: pd.DataFrame, fws_df: pd.DataFrame, clonal_sample_fws_threshold: int, allele_list: pd.DataFrame) -> pd.DataFrame:
        
        """
        Finds allele frequencies in clonal samples.

        Parameters:
        genotypes_df (pd.DataFrame): DataFrame containing genotypic data.
        fws_df (pd.DataFrame): DataFrame containing Fws data.
        clonal_sample_fws_threshold (int): Threshold for identifying clonal samples.
        allele_list (pd.DataFrame): DataFrame containing allele data.

        Returns:
        pd.DataFrame: DataFrame containing clonal allele frequencies.
        """
        
        clonal_samples_df = np.where(fws_df["Fws"] >= clonal_sample_fws_threshold, True, False)
        clonal_genotypes_df = genotypes_df.transpose()[clonal_samples_df].transpose().astype(str)
        clonal_frequencies_df = self.find_frequencies(clonal_genotypes_df)
        allele_list = allele_list.merge(clonal_frequencies_df, left_index = True, right_index = True, suffixes = (None, "_clonal"))
        
        allele_list["clonal_hom_count"] = allele_list.apply(lambda allele : self.count_homozygous(allele), axis = 1)
        allele_list["clonal_het_count"] = allele_list.apply(lambda allele : self.count_heterozygous(allele), axis = 1)
        
        allele_list["clonal_hom_frequency"] = allele_list["clonal_hom_count"] / allele_list["total_nonmissing_count"]
        allele_list["clonal_het_frequency"] = allele_list["clonal_het_count"] / allele_list["total_nonmissing_count"]
        
        return allele_list[["clonal_hom_frequency", "clonal_het_frequency", "total_nonmissing_count_clonal"]]
    
    def find_contextual_sequence(self, allele_list: pd.DataFrame, upstream_window_length: int, downstream_window_length: int, reference_genome_path: str) -> pd.DataFrame:
        
        """
        Finds the contextual reference sequences for each allele.

        Parameters:
        allele_list (pd.DataFrame): DataFrame containing allele data.
        upstream_window_length (int): Length of upstream sequence to extract.
        downstream_window_length (int): Length of downstream sequence to extract.
        reference_genome_path (str): Path to the reference genome.

        Returns:
        pd.DataFrame: DataFrame containing alleles with contextual sequences.
        """
        
        print("Finding contextual reference sequences...")
        
        reference_allele_df = allele_list[["REF", "ALT"]]
        reference_allele_df.reset_index(inplace = True)

        reference_allele_df.loc[:, 'reference_length'] = reference_allele_df.apply(lambda allele : len(allele["REF"]), axis = 1)
        upstream_df = pd.DataFrame({
            'Chromosome' : reference_allele_df["CHROM"],
            'Start' : (reference_allele_df["POS"] - 1) - upstream_window_length,
            'End' : reference_allele_df["POS"] - 1
        })

        downstream_df = pd.DataFrame({
            'Chromosome' : reference_allele_df["CHROM"],
            'Start' : (reference_allele_df['POS'] - 1) + reference_allele_df['reference_length'],
            'End' : (reference_allele_df['POS'] - 1) + reference_allele_df['reference_length'] + downstream_window_length
        })

        upstream_pr = pr.PyRanges(upstream_df)
        downstream_pr = pr.PyRanges(downstream_df)
        
        allele_list['upstream_reference'] = pd.Series(pr.get_sequence(upstream_pr, reference_genome_path)).values
        allele_list['downstream_reference'] = pd.Series(pr.get_sequence(downstream_pr, reference_genome_path)).values
        
        return allele_list
    
    def find_gc_content(self, sequence: str) -> float:

        """
        Calculates the GC content of a sequence.

        Parameters:
        sequence (str): The input sequence.

        Returns:
        float: The GC content.
        """
        
        sequence = sequence.lower()
        c_count = sequence.count('c')
        g_count = sequence.count('g')
        
        return (c_count + g_count) / (len(sequence))
    
    def find_allele_depths(self, allele_depths_df: pd.DataFrame, allele_list: pd.DataFrame) -> pd.DataFrame:
        
        """
        Finds the average depth of alleles at each position.

        Parameters:
        allele_depths_df (pd.DataFrame): DataFrame containing allele depth data.
        allele_list (pd.DataFrame): DataFrame containing allele data.

        Returns:
        pd.DataFrame: DataFrame containing alleles with average depth information.
        """
        
        print("Finding average depth of alleles at each position...")
        
        for i in range(1, 7):
            
            allele_depths = allele_depths_df.apply(lambda position : position.str[(i - 1)])
            allele_depths.replace(-1, 0, inplace = True)
            allele_list[f"allele_{i}_average_depth"] = allele_depths.mean(axis = 1)
        
        average_depth_columns = [column for column in allele_list.columns if "_average_depth" in column]
        
        allele_list["allele_depths_std"] = allele_list[average_depth_columns].std(axis = 1)
        
        return allele_list
    
    def find_clonal_allele_depths(self,  allele_depths_df: pd.DataFrame, fws_df: pd.DataFrame, genotypes_df: pd.DataFrame, allele_list: pd.DataFrame, low_fws_threshold: float) -> pd.DataFrame:
        
        """
        Finds the allele depths for clonal samples.

        Parameters:
        allele_depths_df (pd.DataFrame): DataFrame containing allele depth data.
        fws_df (pd.DataFrame): DataFrame containing Fws data.
        genotypes_df (pd.DataFrame): DataFrame containing genotypic data.
        allele_list (pd.DataFrame): DataFrame containing allele data.
        low_fws_threshold (float): Threshold for identifying low Fws samples.

        Returns:
        pd.DataFrame: DataFrame containing alleles with clonal allele depth information.
        """
        
        low_fws_samples_df = np.where(fws_df["Fws"] <= low_fws_threshold, True, False)
        low_fws_genotypes_df = genotypes_df.transpose()[low_fws_samples_df].transpose().astype(str)
        low_fws_allele_depths_df = allele_depths_df.transpose()[low_fws_samples_df].transpose().astype(str)
        
        #print(low_fws_allele_depths_df)
        #print(low_fws_genotypes_df)
        
        #print(low_fws_allele_depths_df.index.get_level_values(1)[0])
        
        #print(low_fws_genotypes_df.index.get_level_values(1)[low_fws_allele_depths_df.index.get_level_values(1)[0]])
        
        allele_list["low_fws_het_allele_depth_mean_difference"] = low_fws_allele_depths_df.apply(lambda position : self.find_average_allele_depths_of_hets(position, low_fws_genotypes_df), axis = 1)
        
        return allele_list
    
    def find_average_allele_depths_of_hets(self, position: pd.Series, low_fws_genotypes_df: pd.DataFrame):

        """
        Finds the average allele depths for heterozygous calls in low Fws samples.

        Parameters:
        position (pd.Series): Series representing allele depths at a position.
        low_fws_genotypes_df (pd.DataFrame): DataFrame containing genotypic data for low Fws samples.

        Returns:
        float: The mean difference between major and minor allele depths.
        """
        
        position_low_fws_samples = low_fws_genotypes_df.loc[(position.name[0], position.name[1])]
        
        #print(position)
        #print(position_low_fws_samples)
        
        position_low_fws_samples = position_low_fws_samples.to_frame().merge(position, right_index = True, left_index = True, suffixes=("_genotype", "_allele_depths")).transpose()
        position_low_fws_samples.index = ["Genotype", "Allele_depth"]
        position_low_fws_samples = position_low_fws_samples.transpose()
        
        position_low_fws_samples = pd.DataFrame(position_low_fws_samples["Genotype"].tolist()).merge(pd.DataFrame(position_low_fws_samples["Allele_depth"].tolist()), right_index = True, left_index = True)
        #position_low_fws_samples = position_low_fws_samples[position_low_fws_samples["0_x"] != "[-1, -1]"]
        position_low_fws_samples["zygosity"] = position_low_fws_samples["0_x"].apply(lambda genotype : "homozygous" if eval(genotype)[0] == eval(genotype)[1] else "heterozygous")
        
        position_low_fws_samples = position_low_fws_samples[position_low_fws_samples["zygosity"] == "heterozygous"]
        
        if len(position_low_fws_samples) == 0: return None
        else:
            
            position_low_fws_samples["0_y"] = position_low_fws_samples["0_y"].apply(lambda sample: [0 if i == -1 else i for i in eval(sample)])
            position_low_fws_samples["total_allele_depth"] = position_low_fws_samples.apply(lambda sample: sum(sample["0_y"]), axis = 1)
            position_low_fws_samples["left_allele_depth"] = position_low_fws_samples.apply(lambda sample: sample["0_y"][eval(sample["0_x"])[0]], axis = 1)
            position_low_fws_samples["right_allele_depth"] = position_low_fws_samples.apply(lambda sample: sample["0_y"][eval(sample["0_x"])[1]], axis = 1)
            position_low_fws_samples["major_allele_depth"] = position_low_fws_samples[["left_allele_depth", "right_allele_depth"]].values.max(1)
            position_low_fws_samples["minor_allele_depth"] = position_low_fws_samples[["left_allele_depth", "right_allele_depth"]].values.min(1)
            position_low_fws_samples["major_allele_depth_ratio"] = position_low_fws_samples["major_allele_depth"] / position_low_fws_samples["total_allele_depth"]
            position_low_fws_samples["minor_allele_depth_ratio"] = position_low_fws_samples["minor_allele_depth"] / position_low_fws_samples["total_allele_depth"]
            
            print(position_low_fws_samples)
            
            return position_low_fws_samples["major_allele_depth_ratio"].mean() - position_low_fws_samples["minor_allele_depth_ratio"].mean()

    def plot_allele_frequency_spectrum(self, allele_list: pd.DataFrame) -> None:
        
        """
        Plots the allele frequency spectrum.

        Parameters:
        allele_list (pd.DataFrame): DataFrame containing allele data.
        """
        
        print("Plotting allele frequency spectrum...")
        
        allele_list = allele_list.reset_index()
        
        if self.variant_focus == "snp":
            
            allele_list = allele_list[
                (allele_list["type"] == "snp") | (allele_list["type"] == "hidden_snp")
                ][["CHROM", "POS", "hom_frequency", "het_frequency"]]
            
        else:
            
            allele_list = allele_list[
                (allele_list["type"] == "insertion") | (allele_list["type"] == "deletion")
                ][["CHROM", "POS", "hom_frequency", "het_frequency"]]
        
        allele_list['CHROM'] = allele_list['CHROM'].str.replace(f"{self.chromosome_prefix}_", "")
        allele_list['CHROM'] = allele_list['CHROM'].str.replace(f"_{self.chromosome_suffix}", "")
        allele_list['POS'] = allele_list['POS'].astype(str)
        allele_list = allele_list.groupby(['CHROM', 'POS']).sum()
        allele_list.index = allele_list.index.map(':'.join)
        allele_list = allele_list.sort_values("hom_frequency", ascending = False).head(self.frequency_spectrum_number_plotted)
        
        plt.figure(figsize=(8, 8))
        plt.bar(allele_list.index, allele_list["hom_frequency"], label = "Homozygous frequency")
        plt.bar(allele_list.index, allele_list["het_frequency"], bottom = allele_list["hom_frequency"] ,label = "Heterozygous frequency")
        plt.xticks(fontsize = 14, rotation = 90)
        plt.grid(True)
        plt.legend(prop={'size': 14})
        
        plt.savefig(f'{self.results_path}/allele_frequency_spectrum.png', dpi=400)
        
    def plot_heuristics(self, allele_list: pd.DataFrame) -> None:
        
        """
        Plots various heuristics for the alleles.

        Parameters:
        allele_list (pd.DataFrame): DataFrame containing allele data.
        """
        
        print("Plotting heuristics...")
        
        figure, axis = plt.subplots(len(self.heuristics_of_interest), len(self.heuristics_of_interest), dpi = 400, figsize=(len(self.heuristics_of_interest) * 2.5, len(self.heuristics_of_interest) * 2.5), layout = "tight")
        
        for category in self.categories_of_interest:
            
            for x_heuristic in self.heuristics_of_interest:
                
                for y_heuristic in self.heuristics_of_interest:
                    
                    axis[self.heuristics_of_interest.index(x_heuristic), self.heuristics_of_interest.index(y_heuristic)].scatter(
                        allele_list[allele_list[category] == True][y_heuristic],
                        allele_list[allele_list[category] == True][x_heuristic],
                        alpha = 0.6,
                        label = category,
                        edgecolors = "none",
                        s = 2.0
                    )
                    axis[self.heuristics_of_interest.index(x_heuristic), self.heuristics_of_interest.index(y_heuristic)].set(
                        xlabel = y_heuristic,
                        ylabel = x_heuristic
                    )
                    if self.heuristics_of_interest.index(x_heuristic) + self.heuristics_of_interest.index(y_heuristic) == 0:
                        
                        axis[self.heuristics_of_interest.index(x_heuristic), self.heuristics_of_interest.index(y_heuristic)].legend()
                    
        figure.savefig(f'{self.results_path}/heuristics_plot.png', dpi = 400)
        
    def plot_heurisitcs_pca(self, allele_list: pd.DataFrame) -> None:
        
        """
        Plots the results of a PCA analysis on the heuristics.

        Parameters:
        allele_list (pd.DataFrame): DataFrame containing allele data.
        """
        
        print("Plotting heuristics PCA...")
        
        #scaler = StandardScaler()
        #pca_x = scaler.fit(allele_list[self.heuristics_of_interest]).transform(allele_list[self.heuristics_of_interest])
        
        pca = PCA(n_components = len(self.heuristics_of_interest))
        pca.fit(allele_list[self.heuristics_of_interest].dropna())
        print(pca.explained_variance_ratio_)
