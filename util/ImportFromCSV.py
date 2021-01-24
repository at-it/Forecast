import pandas as pd


def import_parse_index_by_date(filename: str, date_column_name: str) -> pd.DataFrame:
    """Imports data from current workbook path adding filename in, parse and index by column name provided"""
    result = pd.read_csv(filename, parse_dates=[date_column_name])
    result.set_index(date_column_name,
                     inplace=True)  # put in rows "date" field, inplace=True does not create a new object
    return result
