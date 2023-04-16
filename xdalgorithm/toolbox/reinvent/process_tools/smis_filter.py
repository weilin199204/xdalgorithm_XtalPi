import pandas as pd


def filtering(input_file_name,output_file_name):
    data = pd.read_csv(input_file_name)
    assert data.shape[0] > 0 and data.shape[1] > 0
    assert 'valid' in list(data.columns)
    assert 'smiles' in list(data.columns)
    assert 'total_score' in list(data.columns)
    data = data[['smiles','valid','total_score']]

    valid_smiles = data.loc[(data['valid']>0)&(data['total_score']>0),['smiles']]['smiles']
    with open(output_file_name,'w') as file_writer:
        content = '\n'.join(list(valid_smiles))
        file_writer.write(content)
