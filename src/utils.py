import pandas as pd


def read_data(filename: str = r"data\input\energy.dat") -> pd.DataFrame:
    """
    Pre: Assuming file already exists

    :param filename:
    :return:
    """
    dat = pd.read_csv(filename)
    dat["Date"] = pd.to_datetime(dat["Date"])
    dat["Consumption"] = pd.to_numeric(dat["Consumption"])
    return dat


def missing_vals(df: pd.DataFrame) -> pd.DataFrame:
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'}
    )

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False
    ).round(1)

    # Return the dataframe with missing information
    return mis_val_table_ren_columns
