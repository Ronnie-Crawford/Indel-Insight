import numpy as np
import pandas as pd
import pysam
import pygenometracks

from config import Config

class Bam_handler:
    
    def __init__(self, chromosome: str, position: int, config: Config):
        
        self.set_config_values(config)
        self.chromosome = self.format_chromosome_string(chromosome)
        self.position = int(position)
        self.bam_paths_df = self.read_bam_paths_list_path()
        self.iterate_bams(self.bam_paths_df, self.chromosome, self.position)
        
    def set_config_values(self, config: Config):
        
        """
        Reads values from config object and sets them as self properties.
        """
        
        print("Setting properties from config...")
        
        for attribute in vars(config).keys():
            
            setattr(self, attribute, getattr(config, attribute))
            
    def format_chromosome_string(self, chromosome: str):
        
        """
        Transforms input chromosome into format used in Pf dataset.
        """
        
        print("Formatting chromosome input...")
        
        if len(chromosome) == 1:
            
            return f"{self.chromosome_prefix}_0{chromosome}_{self.chromosome_suffix}"
        
        elif len(chromosome) == 2:
            
            return f"{self.chromosome_prefix}_{chromosome}_{self.chromosome_suffix}"
        
        else:
            
            return int(chromosome) 
            
    def read_bam_paths_list_path(self) -> pd.DataFrame:
        
        """
        
        """
        
        print("Reading bam paths list...")
        
        return pd.read_csv(self.bam_paths_list_path, sep = "\t")
    
    def iterate_bams(self, bam_paths_df, chromosome, position):
        
        for bam_path, samfile in self.load_bams(bam_paths_df):
            
            reads_df = self.generate_reads_df(bam_path, samfile, chromosome, position)
            read_depth = len(reads_df)
            
    
    def load_bams(self, bam_paths_df):
        
        """
        
        """
        
        print("Loading bam files...")
        
        for bam_index in range(0, len(bam_paths_df["new_bam_fn"])):
            
            yield bam_paths_df["new_bam_fn"][bam_index], pysam.AlignmentFile(bam_paths_df["new_bam_fn"][bam_index], "rb")
        
        
    def generate_reads_df(self, bam_path, samfile, chromosome, position):
        
        reads_df = pd.DataFrame({
            "read" : samfile.fetch(chromosome, position, (position + 1))
        })
        
        if reads_df.empty : return None
              
        reads_df["bam_path"] = bam_path
        reads_df["chromosome"] = chromosome
        reads_df["chrom_start"] = reads_df["read"].apply(lambda read : int(str(read).split("\t")[3]))
        reads_df["read_length"] = reads_df["read"].apply(lambda read : int(str(read).split("\t")[8]))
        reads_df["chrom_end"] = reads_df["chrom_start"] + reads_df["read_length"]
        reads_df["flags"] = reads_df["read"].apply(lambda read : bin(int(str(read).split("\t")[1])))
        reads_df["mapping_quality"] = reads_df["read"].apply(lambda read : int(str(read).split("\t")[4]))
        reads_df["cigar_string"] = reads_df["read"].apply(lambda read : str(read).split("\t")[5])
        reads_df["flags"] = reads_df["flags"].apply(lambda flag : list(flag[2:].zfill(12)))
        reads_df[["multiple_segments_flag", "segments_alligned_flag",
                  "segment_unmapped_flag",
                  "next_segment_unmapped_flag",
                  "sequence_reverse_complimented_flag",
                  "next_sequence_reverse_complimented_flag",
                  "first_sequence_flag",
                  "last_sequence_flag",
                  "secondary_allignment_flag",
                  "fail_qc_flag",
                  "PCR_duplicate_flag",
                  "supplementary_alignment_flag"
                  ]] = reads_df["flags"].to_list()
        reads_df["is_hard_clipped"] = reads_df["cigar_string"].apply(lambda string : True if "H" in string else False)
        reads_df["is_soft_clipped"] = reads_df["cigar_string"].apply(lambda string : True if "S" in string else False)
        
        if self.drop_hard_clipped_reads == True: reads_df = reads_df[reads_df["is_hard_clipped"] == False]
        if self.drop_soft_clipped_reads == True: reads_df = reads_df[reads_df["is_soft_clipped"] == False]
        
        print(reads_df)
        
        return reads_df