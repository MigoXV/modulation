import typer

from pathlib import Path
from typing import Optional, List

app = typer.Typer()

# train command
@app.command(name="train")
def train(
    config_path: str = typer.Argument(
        "config/generator_config_test.yml", help="Path to config yaml file."
    ),
    config_set: Optional[List[str]] = typer.Option(
        None,
        help="Specify overrides of config on the command line.",
    ),
):
    from train import train_from_scatch
    train_from_scatch(config_path=config_path, config_set=config_set)
    
# gene wave command
@app.command(name="gene-wave")
def gene_wave(
    config_path: str = typer.Argument(
        "config/generator_config_test.yml", help="Path to config yaml file."
    ),
    config_set: Optional[List[str]] = typer.Option(
        None,
        help="Specify overrides of config on the command line.",
    ),
):
    from wave_generator.wave_generator import gene_waves_main
    gene_waves_main(config_path=config_path, config_set=config_set)
    
if __name__ == "__main__":
    app()
