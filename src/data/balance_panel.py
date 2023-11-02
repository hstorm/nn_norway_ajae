# %%
import pandas as pd
import numpy as np

import sys
sys.path.append("src/data")
sys.path.append("src/features")

import make_dataset as make_dataset

# %%


def prepare_pt_panel():
    """
    Prepare pt to a suitable format to be supplied to balance_panel()
    Delete missing observations, transfrom KGB colume to int
    Argument:
        df -- dataframe

    Returns:
        df -- Transformed dataframe
    """

    # Delete missing observations base on year
    isNanIdx = df[np.isnan(df['year'])].index  # get index of nan observations
    df.drop(isNanIdx, inplace=True)  # get rid of all nan's

    # Transform KGB to int
    df["KGB"] = df["KGB"].map(int)

# %%


def balance_panel(df,
                  vYear='year',
                  vIdx='KGB'):
    """
    Transforms an unbalance panel to a balanced panel

    Argument:
        df -- dataframe with unbalanced panel
        vYear -- variable name to year variable, expects year
                 as in number not a date
        vIdx  -- variable name of id variable
    Returns:
        df -- Transformed dataframe
    """

    # transform year to datatime, add 15/06 (middel of year)
    df[vYear] = pd.to_datetime(
                                ('15/06/' + df[vYear].map(int).map(str)),
                                format='%d/%m/%Y'
                              )

    # set index
    df.reset_index(inplace=True)
    df.set_index([vIdx, vYear], inplace=True)

    # Make unblanced panel balanced
    df.reset_index(inplace=True)
    df.set_index([vYear], inplace=True)

    #  find  first observation which is observed over the entire time
    countGroup = df.groupby(vIdx).count().iloc[:, 1]
    idxFull = (countGroup.where((countGroup == countGroup.max()))
               .dropna()
               .index
               .tolist()
               )[0]

    # get a full year index
    idx = df.groupby(vIdx).groups[idxFull]
    idx

    # expande each group to the full blanced panel
    df = (
            df.groupby(vIdx)
            .apply(lambda x: x.reindex(idx))
            .drop(vIdx, axis=1)
         )

    return df

# %%


def complete_nan(df,
                 vIdx='KGB',
                 staticCols=['knr', 'zoneTAKTL', 'zoneTDISE', 'zoneTDISG',
                             'zoneTDISK', 'zoneTDISM', 'zoneTDMLK',
                             'zoneTPROD'],
                 vAge='age'
                 ):
    """
    Complete nan values
    Fill the once that are static over time,
    recalcualte age for missing values based on birthyear
    and fill remaining nan with zero

    Argument:
        df -- dataframe with missing values
        vIdx  -- variable name of id variable
        staticCols -- a list of variable that should be back and forward filled
                      (i.e. where nan are not replaced by 0)
        vAge -- name a age variable where missing values should be replaced
                be forward/back filling age based on last/first observed age

    Returns:
        df -- Transformed dataframe
    """

    # Check if age in in columns
    # Aim is to fill missing values in age by increasing (decreasing) age
    # base on last (first) observed age
    if vAge in df.columns:
        # Derive the birthyear
        df["birthyear"] = df.index.get_level_values(level=1).year-df[vAge]
        staticCols += ['birthyear']

    # Fill nan for time invariant variables
    df[staticCols] = (df[staticCols].groupby(level=vIdx)
                                    .fillna(method='bfill')
                                    .fillna(method='ffill'))
    # Recalcualte age
    if vAge in df.columns:
        # Recaclulate age based on  birhtyear column (without nan)
        df[vAge] = df.index.get_level_values(level=1).year-df["birthyear"]

    # Fill remaining nan with zero
    df.fillna(0, inplace=True)

    return df

# %%


if __name__ == '__main__':
    #
    df = pd.read_csv(make_dataset.ExtractFeatures().targetFileName,
                     delim_whitespace=True)
    prepare_pt_panel()
    df = balance_panel(df)
    df = complete_nan(df)
    #
    df.to_csv(make_dataset.BlancePanel().targetFileName)
