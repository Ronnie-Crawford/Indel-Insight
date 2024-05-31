import sys
sys.path.append("./../../../env")

import typer
from typing import List

from config import Config
from gff_handler import Gff_handler
from bam_handler import Bam_handler
from multiregion import Multiregion
from position import Position

app = typer.Typer()

def main():
    
    app()

@app.command()
def multiregion(regions : List[str]) -> None:
    
    """
    Gives information about the variants in a certain region.
    The coordinates are 5' to 3' and relative to the start of the chromosome.

    Parameters:
    regions (List[str]): A list of regions to analyze.
    """
    
    config = Config()
    
    if not regions:
        
        gff_handler = Gff_handler(config)
        regions = gff_handler.regions["formatted_region"][0:20]
    
    multiregion = Multiregion(*regions, config = config)
    multiregion.plot_plots()
    multiregion.save_lists()
    
@app.command()
def bams(chromosome: str, position: int) -> None:

    """
    Analyzes BAM files for a specific chromosome and position.

    Parameters:
    chromosome (str): The chromosome to analyze.
    position (int): The position on the chromosome to analyze.
    """
    
    config = Config()
    bam_handler = Bam_handler(chromosome, position, config)
    
@app.command()
def position(position_coordinates: str) -> None:
    
    """
    For the given position, returns the distribution of frequencies of alternative alleles,
    over time and in different populations.

    Parameters:
    position_coordinates (str): The coordinates of the position to analyze.
    """
    
    config = Config()
    position = Position(position_coordinates, config)
    position.plots()
    
@app.command()    
def view_config() -> None:
    
    """
    Prints details of the current config.
    """
    
    config = Config()
    config.view_config_settings()
    
if __name__ == "__main__" : main()
