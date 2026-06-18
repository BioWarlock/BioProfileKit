from src.cli.app import cli

if __name__ == '__main__':
    #cli(['--input', 'data/iedb.tsv', '--tax', '-tc','bind_class'])
   # cli(['-i', "data/Complete_nitrogenase.csv"])
    #cli(['--input', 'data/sorfdb_sub.tsv'])
   # cli(['-i', "data/iedb_sub.tsv", "--tax"])
    #cli(['-i', 'data/winequality-white.csv'])
#    cli(['-i', "data/proteinGroups.tsv"])
    cli(['-i', 'data/bakrep_sagalactiae.tsv'])
    #cli(['-i', 'data/iedb_test.tsv', '-tc', 'bind_class'])