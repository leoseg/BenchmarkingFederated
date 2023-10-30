import pandas as pd
import re
from typing import Union


def genexpr_txt_file_to_csv(
    filepath_txt: str, filepath_csv: Union[None, str] = None
) -> pd.DataFrame:
    """
    Reads the txt file with the genepxression matrix to a csv table
    :param filepath_txt: path to txt file
    :param filepath_csv: if given saves as csv file
    """

    with open(filepath_txt, "r") as file:
        header = file.readline().strip()

    header = header.replace('"', "").split()
    data = pd.read_csv(
        filepath_txt, skiprows=1, engine="python", names=header, sep="\s+"
    )
    data = data.rename(columns={"Unnamed: 0": "Gene"})
    data = data.T
    data = data.rename(columns=data.iloc[0])
    data.index.names = ["Sample"]
    data = data.iloc[1:]
    if filepath_csv:
        data.to_csv(filepath_csv)
    return data


def genexpr_anotation_txt_to_csv(
    filepath_txt: str, filepath_csv: str = None
) -> pd.DataFrame:
    """
    Reads the txt file with the genepxression annotation to a dataframe
    :param filepath_txt: path to txt file
    :param filepath_csv: if given saves as csv file
    """
    with open(filepath_txt, "r") as file:
        header = file.readline().strip()

    header = header.replace('"', "").split()
    data = pd.read_csv(
        filepath_txt,
        skiprows=1,
        engine="python",
        names=header,
        sep="\s+",
        index_col=False,
    )
    data["Sample"] = data["Dataset"].apply(
        lambda x: re.split("_|\.", x)[0].replace('"', "")
    )
    data = data.set_index("Sample")
    if filepath_csv:
        data.to_csv(filepath_csv)
    return data
