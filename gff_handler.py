import pandas as pd

from config import Config

class Gff_handler:

    """
    Initializes the Gff_handler object, sets configuration values, and reads the GFF file.

    Parameters:
    config (Config): A Config object containing configuration settings.
    """
    
    def __init__(self, config: Config):
        
        self.set_config_values(config)
        self.regions = self.read_gff(self.gff_path)
        
    def set_config_values(self, config: Config) -> None:
        
        """
        Reads values from the config object and sets them as attributes of the Gff_handler object.

        Parameters:
        config (Config): A Config object containing configuration settings.
        """
        
        print("Setting properties from config...")
        
        for attribute in vars(config).keys():
            
            setattr(self, attribute, getattr(config, attribute))
            
    def read_gff(self, gff_path):
        
        """
        Reads the provided .gff file and returns desired region coordinates.

        Parameters:
        gff_path (str): The path to the .gff file.

        Returns:
        pd.DataFrame: A DataFrame containing the formatted region coordinates.
        """
        
        print("Reading .gff file...")
        
        regions = pd.read_csv(gff_path, header = 18, sep = "\t", usecols = [0, 2, 3, 4], names = ["chromosome", "label", "start", "end"])
        regions = regions[regions["label"] == self.regions_of_focus]
        regions = regions[(regions["chromosome"] != "Pf3D7_API_v3") & (regions["chromosome"] != "Pf3D7_MIT_v3")]
        regions["formatted_region"] = regions.apply(lambda region : f"{region.chromosome}:{str(region.start)}-{str(region.end)}", axis = 1)
        
        return regions
