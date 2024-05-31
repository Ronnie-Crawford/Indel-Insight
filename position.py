import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from config import Config

class Position:
    
    def __init__(self, position_coordinates: str, config: Config):
        
        """
        Initialises the object; collates the genotypic, metadata and
        allele depth data before calculating frequencies and additional data.
        """
        
        self.set_config_values(config)
        self.chromosome = self.format_chromosome_string(
            position_coordinates.split(":")[0]
            )
        self.coordinate = int(
            position_coordinates.split(":")[1]
            )
        self.callset = self.read_callset(
            self.callset_path
            )
        self.position_mask = self.find_position_mask(
            self.callset, self.chromosome, self.coordinate
            )
        self.metadata_df = self.generate_metadata_df(
            self.metadata_path
            )
        self.fws_df = self.read_fws(
            self.fws_path
            )
        self.genotypes_df = self.generate_genotypes_df(
            self.callset, self.metadata_df, self.position_mask, self.fws_df
            )
        self.genotypes_df = self.apply_quality_control_genotypes(
            self.genotypes_df
            )
        self.allele_depths_df = self.generate_allele_depths_df(
            self.callset, self.position_mask
            )
        self.genotypes_df = self.genotypes_df.merge(
            self.allele_depths_df, right_index = True, left_index = True
            )
        self.genotypes_df = self.process_read_depths(
            self.genotypes_df
            )
        self.frequencies_df = self.find_frequencies(
            self.genotypes_df
            )
        self.position_reliabilty = self.calculate_position_reliability(
            self.genotypes_df
        )
        print(f"Position reliability is : {self.position_reliabilty}")
        
    def plots(self) -> None:
    
        """
        This function calls all other functions which generate plots.
        """
    
        self.plot_space_time_groups(self.frequencies_df)
        self.plot_fws_allele_depth_ratio(self.genotypes_df)
        self.investigate_clonal_samples(self.genotypes_df)
    
    def set_config_values(self, config: Config) -> None:
        
        """
        Reads values from config object and sets them as self properties.
        """
        
        print("Setting properties from config...")
        
        for attribute in vars(config).keys():
            
            setattr(self, attribute, getattr(config, attribute))
    
    def format_chromosome_string(self, chromosome: str) -> str:
        
        """
        Transforms input chromosome into format used in Pf dataset.
        """
        
        print("Formatting chromosome input...")
        
        if len(chromosome) == 1:
            
            return (
                f"{self.chromosome_prefix}_0"
                f"{chromosome}_{self.chromosome_suffix}"
            )
        
        elif len(chromosome) == 2:
            
            return (
                f"{self.chromosome_prefix}_{chromosome}_"
                f"{self.chromosome_suffix}"
                )            
        
        else:
            
            return int(chromosome)
    
    def read_callset(self, callset_path: str) -> zarr.hierarchy.Group:
        
        """
        Opens the .zarr file from the given path for the callset.
        """
        
        print("Reading callset...")
        
        return zarr.open(callset_path, "r")
    
    def read_fws(self, fws_path: str) -> pd.DataFrame:
        
        return pd.read_csv(fws_path, sep="\t")
    
    def find_position_mask(
        self, callset: zarr.hierarchy.Group,
        chromosome: int, genome_coordinate: int
        ) -> int:
    
        """
        Converts the chromosome and position on the chromosome
        to a position within the callset data.
        """
    
        print("Finding coordinate of position...")
        
        position_mask = (
            (callset['variants']['CHROM'][:] == chromosome) &
            (callset['variants']['POS'][:] == genome_coordinate)
        )

        return int(np.argmax(position_mask==True))
    
    def generate_metadata_df(self, metadata_path: str) -> pd.DataFrame:
        
        """
        Extract metadata from the config metadata path
        and format as a pandas dataframe.
        """
        
        print("Generating metadata dataframe...")
        
        return pd.read_csv(
            metadata_path, sep='\t', header=0, index_col=0
            ).transpose()
    
    def generate_genotypes_df(
        self, callset: zarr.hierarchy.Group, metadata_df: pd.DataFrame,
        position_mask: int, fws_df: pd.DataFrame
        ) -> pd.DataFrame:
        
        """
        Extract genotypes from the callset for the given position
        and format into a dataframe.
        """
       
        print("Generating genotypes dataframe...")
        
        genotypes_df = pd.DataFrame.from_records(
            data = callset['calldata']['GT'][position_mask].tolist(),
            index = callset["samples"],
            columns = ['primary_genotype', 'secondary_genotype']
        )
        
        genotypes_df['genotype'] = genotypes_df.apply(
            lambda sample : (
                f"[{sample['primary_genotype']}, "
                f"{sample['secondary_genotype']}]"
                ), axis = 1
            )
        
        genotypes_df["zygosity"] = genotypes_df.apply(
            lambda sample : (
                "heterozygous" if sample["primary_genotype"] !=
                sample["secondary_genotype"] else "homozygous"
                ), axis = 1
            )
        genotypes_df = genotypes_df.merge(
            metadata_df.transpose(), right_index = True, left_index = True
            )
        genotypes_df = genotypes_df.merge(
            fws_df.set_index("Sample"), right_index = True, left_index = True
            )
        
        return genotypes_df
    
    def generate_allele_depths_df(
        self, callset: zarr.hierarchy.Group, position_mask: int
        ) -> pd.DataFrame:
        
        """
        From the callset, extract the allele depths for all alleles at each
        position and generate as a dataframe.
        """
        
        print("Generating allele depths dataframe...")
        
        allele_depths_df = pd.DataFrame.from_records(
            data = callset['calldata']['AD'][position_mask].tolist(),
            index = callset['samples'],
            columns = [
                "allele_0_depth",
                "allele_1_depth",
                "allele_2_depth",
                "allele_3_depth",
                "allele_4_depth",
                "allele_5_depth",
                "allele_6_depth"
                ]
        ).replace(-1, 0)
        
        return allele_depths_df
    
    def apply_quality_control_genotypes(
        self, genotypes_df: pd.DataFrame
        ) -> pd.DataFrame:
    
        return genotypes_df[genotypes_df['QC pass'] == True]
    
    def find_frequencies(self, genotypes_df: pd.DataFrame) -> pd.DataFrame:
        
        """
        Takes dataframe of genotypes,
        counts occurence of each genotype in each group,
        then finds frequencies of each genotype compared to total nonmissing
        count in each group.
        """
        
        print("Finding frequencies...")

        genotypes_df["genotype"] = genotypes_df["genotype"].astype('category')
        counts_df = genotypes_df[[
            self.position_frequency_grouping,
            "Year", "Country latitude", "Country longitude", "genotype"
            ]].groupby(
                [self.position_frequency_grouping, "Year"]
                ).value_counts().unstack()
        counts_df["total_nonmissing"] = counts_df.drop(
            "[-1, -1]", axis = 1
            ).sum(axis = 1)
        counts_df = counts_df[
            counts_df["total_nonmissing"] > self.low_sample_threshold
            ]
        
        frequencies_df = counts_df.filter(
            self.genotypes_list
            ).divide(counts_df["total_nonmissing"], axis = 0).fillna(0)
    
        het_genotypes = [
            genotype for genotype in frequencies_df.columns if 
            genotype[1] != genotype[4]
            ]
        frequencies_df["Heterozygous_frequency"] = frequencies_df.filter(
            het_genotypes
            ).sum(axis = 1)
        frequencies_df.drop(het_genotypes, axis = 1, inplace = True)

        return frequencies_df
    
    def process_read_depths(self, genotypes_df: pd.DataFrame):
        
        """
        For each sample, finds: the read depth of all alleles,
        the reads depths of the major and minor alleles,
        and the ratio of these to the total read depth.
        """
        
        print("Processing read depths...")
        
        genotypes_df["total_allele_depth"] = genotypes_df[[
            "allele_0_depth",
            "allele_1_depth",
            "allele_2_depth",
            "allele_3_depth",
            "allele_4_depth",
            "allele_5_depth",
            "allele_6_depth"
            ]].sum(axis = 1)
        genotypes_df["left_allele_depth"] = genotypes_df.apply(
            lambda sample : (
                sample[f"allele_{sample.primary_genotype}_depth"] if 
                sample.primary_genotype in [0, 1, 2, 3, 4, 5, 6] else None
                ),axis = 1)
        genotypes_df["right_allele_depth"] = genotypes_df.apply(
            lambda sample : (
                sample[f"allele_{sample.secondary_genotype}_depth"] if
                sample.secondary_genotype in [0, 1, 2, 3, 4, 5, 6] else
                None
                ), axis = 1)
        
        genotypes_df["major_allele_depth"] = genotypes_df[["left_allele_depth", "right_allele_depth"]].values.max(1)
        genotypes_df["minor_allele_depth"] = genotypes_df[["left_allele_depth", "right_allele_depth"]].values.min(1)
        
        genotypes_df["major_allele_depth_ratio"] = (
            genotypes_df[
                "major_allele_depth"
                ] / genotypes_df["total_allele_depth"]
            )
        genotypes_df["minor_allele_depth_ratio"] = (
            genotypes_df[
                "minor_allele_depth"
                ] / genotypes_df[
                    "total_allele_depth"
                    ])
        
        return genotypes_df
    
    def calculate_position_reliability(self, genotypes_df: pd.DataFrame) -> float:
        
        #clusters = KMeans(n_clusters=2)
        
        #genotypes_df["cluster"] = clusters.fit_predict(genotypes_df[["major_allele_depth"]])
        
        #high_fws_genotypes_df = genotypes_df[genotypes_df["Fws"] > self.clonal_sample_fws_threshold]
        het_genotypes_df = genotypes_df[genotypes_df["zygosity"] == "heterozygous"]
        low_fws_het_genotypes_df = het_genotypes_df[het_genotypes_df["Fws"] < self.low_fws_sample_threshold]
        high_fws_het_genotypes_df = het_genotypes_df[het_genotypes_df["Fws"] > self.clonal_sample_fws_threshold]
        
        print((abs(low_fws_het_genotypes_df["major_allele_depth_ratio"]) - abs(low_fws_het_genotypes_df["minor_allele_depth_ratio"])).mean())
        print((abs(low_fws_het_genotypes_df["major_allele_depth_ratio"]) - abs(low_fws_het_genotypes_df["minor_allele_depth_ratio"])).std())
        
        print(low_fws_het_genotypes_df["major_allele_depth_ratio"].mean())
        print(low_fws_het_genotypes_df["minor_allele_depth_ratio"].mean())
        
        print(low_fws_het_genotypes_df["major_allele_depth_ratio"].kurtosis())
        print(low_fws_het_genotypes_df["minor_allele_depth_ratio"].kurtosis())
        
        print(high_fws_het_genotypes_df["major_allele_depth_ratio"].std())
        print(high_fws_het_genotypes_df["minor_allele_depth_ratio"].std())
        
        print(len(low_fws_het_genotypes_df))
        
        print(low_fws_het_genotypes_df["total_allele_depth"].mean())
        
        return low_fws_het_genotypes_df["minor_allele_depth_ratio"].mean() / low_fws_het_genotypes_df["major_allele_depth_ratio"].mean()
        
        
        #print(high_fws_genotypes_df)
        #print(low_fws_genotypes_df)
        
        #print(kurtosis(high_fws_genotypes_df["primary_allele_depth"]))
        
        #high_fws_genotypes_kurtosis = high_fws_genotypes_df["major_allele_depth"].std() + high_fws_genotypes_df["minor_allele_depth"].std()
        #low_fws_genotypes_kurtosis = low_fws_genotypes_df["major_allele_depth"].std() + low_fws_genotypes_df["minor_allele_depth"].std()
        
        #print(low_fws_genotypes_df["major_allele_depth"].std())
        #print(low_fws_genotypes_df["minor_allele_depth"].std())
        #print(pd.concat([low_fws_genotypes_df["major_allele_depth"], low_fws_genotypes_df["minor_allele_depth"]]).std())
        
        #print(low_fws_genotypes_df["major_allele_depth"].kurtosis())
        #print(low_fws_genotypes_df["minor_allele_depth"].kurtosis())
        #print(pd.concat([low_fws_genotypes_df["major_allele_depth"], low_fws_genotypes_df["minor_allele_depth"]]).kurtosis())
        
        #print(print(len(low_fws_genotypes_df["major_allele_depth"])))
        
        #print(pd.concat([low_fws_genotypes_df["major_allele_depth"], low_fws_genotypes_df["minor_allele_depth"]]).std() / len(low_fws_genotypes_df))
        #print(pd.concat([low_fws_genotypes_df["major_allele_depth"], low_fws_genotypes_df["minor_allele_depth"]]).kurtosis() / len(low_fws_genotypes_df))
        
        #return (pd.concat([low_fws_genotypes_df["major_allele_depth"], low_fws_genotypes_df["minor_allele_depth"]]).kurtosis() / pd.concat([low_fws_genotypes_df["major_allele_depth"], low_fws_genotypes_df["minor_allele_depth"]]).std()) * len(low_fws_genotypes_df)
    
    def plot_space_time_groups(self, frequencies_df: pd.DataFrame) -> None:
        
        """
        Iterates through each group,
        plotting the frequency of each genotype
        within each year within that group.
        """
        
        print("Plotting group frequencies...")
        
        groupings = frequencies_df.index.get_level_values(
            self.position_frequency_grouping
            ).unique().to_list()
        years = frequencies_df.index.get_level_values(
            "Year"
            ).unique().to_list()
        genotypes = self.plotted_position_genotypes
        
        frequencies_df = frequencies_df.filter(genotypes)
        frequencies_df.index = frequencies_df.index.droplevel(
            ["Country latitude", "Country latitude"]
            )
    
        figure, axis = plt.subplots(
            len(groupings),
            dpi = 400,
            figsize=(len(years) * 0.8,len(groupings) * 2.5),
            sharex = True,
            tight_layout = False
            )
        plt.rcParams.update({'font.size': 16})
        
        for grouping in groupings:
        
            bar_plot_bottom = 0
        
            for genotype in frequencies_df.columns:
                
                axis[groupings.index(grouping)].bar(
                    frequencies_df.xs(
                        grouping, level = self.position_frequency_grouping
                        )[genotype].index,
                    frequencies_df.xs(
                        grouping, level = self.position_frequency_grouping
                        )[genotype],
                    bottom = bar_plot_bottom,
                    label=str(genotype)
                    )
                
                bar_plot_bottom += frequencies_df.xs(
                    grouping, level = self.position_frequency_grouping
                    )[genotype]
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc = 'best',
                    prop = {'size': 20}
                    )
                axis[groupings.index(grouping)].set_title(f"{grouping}")
                axis[groupings.index(grouping)].set_yticks(
                    [0.0, 0.5, 1.0],
                    fontsize=50
                    )
                axis[groupings.index(grouping)].grid(True)
                
        figure.suptitle(
            f"Frequency of alleles at {self.chromosome}:{self.coordinate}",
            fontsize = 24
            )
        plt.xticks(years, fontsize = 20, rotation = 90)
        figure.savefig(
            f'{self.results_path}/frequency_over_time_and_space.png',
            dpi=400
            )
        
    def plot_fws_allele_depth_ratio(self, genotypes_df: pd.DataFrame) -> None:
        
        hets_df = genotypes_df[genotypes_df["zygosity"] == "heterozygous"]
        
        plt.figure(
            figsize=(8, 8)
            )
        plt.scatter(
            hets_df["Fws"],
            hets_df["major_allele_depth_ratio"],
            c = "red",
            s = 2.5,
            alpha = 0.4,
            label = "Major allele"
            )
        plt.scatter(
            hets_df["Fws"],
            hets_df["minor_allele_depth_ratio"],
            c = "blue",
            s = 2.5,
            alpha = 0.4,
            label = "Minor allele"
            )
        plt.grid(
            True
            )
        plt.xlabel(
            "Fws"
            )
        plt.ylabel(
            "Allele_depth_ratio"
            )
        plt.legend(
            prop={'size': 14}
            )
        
        plt.savefig(
            f'{self.results_path}/fws_allele_depth_ratio.png',
            dpi = 400
            )
        
    def investigate_clonal_samples(self, genotypes_df: pd.DataFrame) -> None:
        
        """
        This function is used to add analyses needed for clonal samples
        at the given position, but it is still being finalised.
        """
        
        print("Plotting sequencing groups within clonal samples...")
        
        genotypes_df = genotypes_df[genotypes_df["genotype"] != "[-1, -1]"]
        genotypes_df = genotypes_df.drop([
            "primary_genotype",
            "secondary_genotype",
            "Country latitude",
            "Country longitude",
            "All samples same case",
            "QC pass",
            "Exclusion reason",
            "Sample was in Pf6",
            "% callable",
            "Admin level 1 latitude",
            "Population"
            ], axis = 1)
        
        clonal_genotypes_df = genotypes_df[
            genotypes_df["Fws"] >= self.clonal_sample_fws_threshold
            ]
        #clonal_genotypes_df = clonal_genotypes_df[(clonal_genotypes_df["primary_allele_depth_ratio"] > 0.4) & (clonal_genotypes_df["primary_allele_depth_ratio"] < 0.6)]
        clonal_genotypes_df = clonal_genotypes_df[(
            clonal_genotypes_df["major_allele_depth_ratio"] < 0.05
            ) | (
                clonal_genotypes_df["minor_allele_depth_ratio"] < 0.05
                )]
        
        print(clonal_genotypes_df[["genotype", "major_allele_depth", "minor_allele_depth", "major_allele_depth_ratio", "minor_allele_depth_ratio"]].sort_values("major_allele_depth_ratio"))
        #print(clonal_genotypes_df.sort_values("primary_allele_depth_ratio").iloc[0])
        #print(clonal_genotypes_df["Year"].value_counts())
        #print(clonal_genotypes_df["Study"].value_counts())
        
        #print(clonal_genotypes_df[genotypes_df["zygosity"] == "heterozygous"].sort_values("Fws").head(50).iloc[0])
        #print(clonal_genotypes_df[genotypes_df["genotype"] == "[1, 2]"].sort_values("Fws", ascending = False).head(50))
        #print(genotypes_df[["zygosity", "Sample type"]].groupby(["zygosity", "Sample type"]).value_counts().unstack())
        #print(clonal_genotypes_df[["zygosity", "Sample type"]].groupby(["zygosity", "Sample type"]).value_counts().unstack())
        #print(clonal_genotypes_df[["genotype", "Sample type"]].groupby(["genotype", "Sample type"]).value_counts().unstack())
        #print(clonal_genotypes_df[clonal_genotypes_df["zygosity"] == "heterozygous"].sort_values("Fws", ascending = False).head(50))
        
        #het_clonal_genotypes_df = clonal_genotypes_df[genotypes_df["genotype"] == "[2, 3]"].sort_values("Fws", ascending = False)
        
        #print(f"Samples in study 1052-PF-TRAC-WHITE : {len(het_clonal_genotypes_df[het_clonal_genotypes_df['Study'] == '1052-PF-TRAC-WHITE'])} out of {len(het_clonal_genotypes_df['Year'])}")
        #print(het_clonal_genotypes_df["Year"].value_counts())
        
        #print(clonal_genotypes_df[["Study", "Country", "Year", "Sample type", "zygosity"]].value_counts().unstack().reset_index().fillna(0).head(50))
        
        #clonal_genotypes_df[["Study", "Country", "Year", "Sample type", "zygosity"]].value_counts().unstack().reset_index().fillna(0).to_csv(f"{self.results_path}/clonal_samples_distribution.tsv", sep='\t')
        