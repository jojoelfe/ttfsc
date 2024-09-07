import pooch
import pytest
from typer.testing import CliRunner

from ttfsc._cli import cli


@pytest.fixture(scope="module")
def halfmap1():
    fname = pooch.retrieve(
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-37579/other/emd_37579_half_map_1.map.gz",
        known_hash="e1c62f4eb3615d2c29c5cae6de3c10951e5aa828dac2f6aa04b0838dc2b96b6f",
    )
    return fname


@pytest.fixture(scope="module")
def halfmap2():
    fname = pooch.retrieve(
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-37579/other/emd_37579_half_map_2.map.gz",
        known_hash="c076ee990a91715900e081a92a0747078ed5ec4796d35614f0b41025e5c24212",
    )
    return fname


runner = CliRunner()


def test_app(halfmap1, halfmap2):
    result = runner.invoke(cli, [halfmap1, halfmap2])
    print(result.output)
    assert result.exit_code == 0
    assert "Estimated resolution using 0.143 criterion in unmasked map: 3.63 Å" in result.output
