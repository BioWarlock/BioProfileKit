from app import cli

if __name__ == '__main__':
    cli(['--input', 'data/iedb.tsv', '--tax', '-tc','bind_class'])
    #cli(['-i', "test_data/Complete_nitrogenase.csv"])
    #cli(['--input', 'test_data/data/upstream_sd.complete.tsv'])