import pandas as pd
def convertTopGeneDictToDF(
    data_dict,
    gene_list_as_string:bool=True):
    """
    Converts the dictionary containing the top genes and their scores reported by `EmerGene` function into a wide-format DataFrame where each condition has two columns:
    "{condition}_Gene" and "{condition}_EG_score".

    Parameters
    ----------
    data_dict : dict
        Dictionary where keys are conditions.
        - If `gene_list_as_string=True`: values are "gene:score" formatted strings.
        - If `gene_list_as_string=False`: values are DataFrames with 'Gene' and 'EG_score' columns.

    gene_list_as_string : bool, optional (default=True)
        - If True, assumes values in `data_dict` are strings formatted as "gene:score,gene2:score2,...".
        - If False, assumes values in `data_dict` are DataFrames with 'Gene' and 'EG_score' columns.

    Returns
    -------
    pd.DataFrame
        A wide-format DataFrame where each condition has two columns: "{condition}_Gene" and "{condition}_EG_score".
    """
    wide_df = pd.DataFrame()

    for condition, data in data_dict.items():
        if gene_list_as_string:
            # Convert "gene:score" string into a DataFrame
            gene_score_pairs = [pair.split(":") for pair in data.split(",")]
            condition_df = pd.DataFrame(gene_score_pairs, columns=[f"{condition}_Gene", f"{condition}_EG_score"])
            
            # Convert score column to float
            condition_df[f"{condition}_EG_score"] = condition_df[f"{condition}_EG_score"].astype(float)
        else:
            # Assume data is already a DataFrame with 'Gene' and 'EG_score' columns
            condition_df = data.rename(columns={"Gene": f"{condition}_Gene", "EG_score": f"{condition}_EG_score"})

        # Concatenate to the main DataFrame
        if wide_df.empty:
            wide_df = condition_df
        else:
            wide_df = pd.concat([wide_df, condition_df], axis=1)

    return wide_df