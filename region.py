import sys
sys.path.append("./../../../env")

import zarr
import numpy as np
import pandas as pd

from config import Config

class Region:
    
    def __init__(self, chromosome: str, start: int, end: int, config: Config, callset: zarr.hierarchy.Group):

        """
        Initialises the Region object and sets up necessary properties and dataframes.

        Parameters:
        chromosome (str): The chromosome identifier.
        start (int): The start position of the region.
        end (int): The end position of the region.
        config (Config): A Config object containing configuration settings.
        callset (zarr.hierarchy.Group): The zarr group containing genomic data.
        """
        
        self.set_config_values(config)
        self.chromosome = self.format_chromosome_string(chromosome)
        self.start = int(start)
        self.end = int(end)
        self.callset = callset
        self.start_coordinate, self.end_coordinate = self.find_coordinates(self.chromosome, self.start, self.end, self.callset)
        self.variants_df = self.generate_variants_df(self.callset, self.chromosome, self.start_coordinate, self.end_coordinate)
        self.metadata_df = self.generate_metadata_df(self.metadata_path)
        self.genotypes_df = self.generate_genotypes_df(self.callset, self.variants_df, self.chromosome, self.start_coordinate, self.end_coordinate)
        self.allele_depths_df = self.generate_allele_depths_df(self.callset, self.chromosome, self.start_coordinate, self.end_coordinate)
        self.genotypes_df = self.apply_quality_control_genotypes(self.genotypes_df, self.variants_df, self.metadata_df)
        self.allele_depths_df = self.apply_quality_control_allele_depths(self.allele_depths_df, self.variants_df, self.metadata_df)
    
    def set_config_values(self, config: Config) -> None:
        
        """
        Reads values from the config object and sets them as attributes of the Region object.

        Parameters:
        config (Config): A Config object containing configuration settings.
        """
        
        print("Setting properties from config...")
        
        for attribute in vars(config).keys():
            
            setattr(self, attribute, getattr(config, attribute))
    
    def format_chromosome_string(self, chromosome: str) -> str:
        
        """
        Transforms the input chromosome into the format used in the Pf dataset.

        Parameters:
        chromosome (str): The input chromosome string.

        Returns:
        str: The formatted chromosome string.
        """
        
        print("Formatting chromosome input...")
        
        if len(chromosome) == 1:
            
            return f"{self.chromosome_prefix}_0{chromosome}_{self.chromosome_suffix}"
        
        elif len(chromosome) == 2:
            
            return f"{self.chromosome_prefix}_{chromosome}_{self.chromosome_suffix}"
        
        else:
            
            return chromosome
    
    def find_coordinates(self, chromosome: int, start: int, end: int, callset: zarr.hierarchy.Group) -> int:
    
        """
        Finds the coordinates of the region within the genomic data.

        Parameters:
        chromosome (int): The chromosome identifier.
        start (int): The start position of the region.
        end (int): The end position of the region.
        callset (zarr.hierarchy.Group): The zarr group containing genomic data.

        Returns:
        tuple: A tuple containing the start and end coordinates within the data.
        """
        
        print("Finding coordinates of region...")
        
        region_mask = (
            (callset['variants']['CHROM'][:] == chromosome) &
            (callset['variants']['POS'][:] >= start) &
            (callset['variants']['POS'][:] <= end)
        )

        start = int(np.argmax(region_mask==True))
        end = start + int(np.argmax(region_mask[start:] == False))
        
        return start, end
    
    def generate_variants_df(self, callset: zarr.hierarchy.Group, chromosome: int, start: int, end: int) -> pd.DataFrame:
        
        """
        Generates a DataFrame of variants for the specified region.

        Parameters:
        callset (zarr.hierarchy.Group): The zarr group containing genomic data.
        chromosome (int): The chromosome identifier.
        start (int): The start coordinate within the data.
        end (int): The end coordinate within the data.

        Returns:
        pd.DataFrame: A DataFrame containing variant information.
        """
        
        print("Generating variants dataframe...")
        
        variants_df = pd.DataFrame({
            'POS' : callset['variants']['POS'][start:end].tolist(),
            'is_snp' : callset['variants']['is_snp'][start:end].tolist(),
            'FILTER_PASS' : callset['variants']['FILTER_PASS'][start:end].tolist(),
            'REF' : callset['variants']['REF'][start:end].tolist(),
            'ALT' : callset['variants']['ALT'][start:end].tolist(),
            "CHROM" : chromosome
        })
        variants_df.set_index(["CHROM", "POS"], inplace = True)
        
        return variants_df
    
    def generate_metadata_df(self, metadata_path: str) -> pd.DataFrame:
        
        """
        Generates a DataFrame of metadata from the specified path.

        Parameters:
        metadata_path (str): The path to the metadata file.

        Returns:
        pd.DataFrame: A DataFrame containing metadata.
        """
        
        print("Generating metadata dataframe...")
        
        return pd.read_csv(metadata_path, sep='\t', header=0, index_col=0).transpose()
    
    def generate_genotypes_df(self, callset: zarr.hierarchy.Group, variants_df: pd.DataFrame, chromosome: int, start: int, end: int) -> pd.DataFrame:
        
        """
        Generates a DataFrame of genotypes for the specified region.

        Parameters:
        callset (zarr.hierarchy.Group): The zarr group containing genomic data.
        variants_df (pd.DataFrame): DataFrame containing variant information.
        chromosome (int): The chromosome identifier.
        start (int): The start coordinate within the data.
        end (int): The end coordinate within the data.

        Returns:
        pd.DataFrame: A DataFrame containing genotype information.
        """
       
        print("Generating genotypes dataframe...")
        
        genotypes_df = pd.DataFrame.from_records(
            data = callset['calldata']['GT'][start:end].tolist(),
            columns = callset['samples']
        )
        
        genotypes_df["POS"] = callset['variants']['POS'][start:end].tolist()
        genotypes_df["CHROM"] = chromosome
        genotypes_df.set_index(["CHROM", "POS"], inplace = True)
        
        return genotypes_df

    def generate_allele_depths_df(self, callset: zarr.hierarchy.Group, chromosome: int, start: int, end: int) -> pd.DataFrame:
        
        """
        Generates a DataFrame of allele depths for the specified region.

        Parameters:
        callset (zarr.hierarchy.Group): The zarr group containing genomic data.
        chromosome (int): The chromosome identifier.
        start (int): The start coordinate within the data.
        end (int): The end coordinate within the data.

        Returns:
        pd.DataFrame: A DataFrame containing allele depth information.
        """
        
        print("Generating allele depths dataframe...")
        
        allele_depths_df = pd.DataFrame.from_records(
            data = callset['calldata']['AD'][start:end].tolist(),
            columns = callset['samples']
        )
        
        allele_depths_df["POS"] = callset['variants']['POS'][start:end].tolist()
        allele_depths_df["CHROM"] = chromosome
        allele_depths_df.set_index(["CHROM", "POS"], inplace = True)
        
        return allele_depths_df
    
    def apply_quality_control_genotypes(self, genotypes_df: pd.DataFrame, variants_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
        
        """
        Applies quality control filters to the genotypes DataFrame using metadata.

        Parameters:
        genotypes_df (pd.DataFrame): DataFrame containing genotype information.
        variants_df (pd.DataFrame): DataFrame containing variant information.
        metadata_df (pd.DataFrame): DataFrame containing metadata.

        Returns:
        pd.DataFrame: A DataFrame containing filtered genotype information.
        """
        
        print("Applying quality control filters to genotypes...")
        
        genotypes_df = genotypes_df[(variants_df['FILTER_PASS'])]
        genotypes_df = genotypes_df.transpose()[(metadata_df.transpose()["QC pass"])].transpose()
        
        return genotypes_df
    
    def apply_quality_control_allele_depths(self, allele_depths_df: pd.DataFrame, variants_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
        
        """
        Applies quality control filters to the allele depths DataFrame using metadata.

        Parameters:
        allele_depths_df (pd.DataFrame): DataFrame containing allele depth information.
        variants_df (pd.DataFrame): DataFrame containing variant information.
        metadata_df (pd.DataFrame): DataFrame containing metadata.

        Returns:
        pd.DataFrame: A DataFrame containing filtered allele depth information.
        """
        
        print("Applying quality control filters to allele depths...")
        
        allele_depths_df = allele_depths_df[(variants_df['FILTER_PASS'])]
        allele_depths_df = allele_depths_df.transpose()[(metadata_df.transpose()["QC pass"])].transpose()
        
        return allele_depths_df
