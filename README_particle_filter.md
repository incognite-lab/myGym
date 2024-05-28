## Návod na spouštění vizualizačního skriptu

Po aktivování conda environmentu
`conda activate mygym`
lze spustit ve složce `cd /myGym/`
skript pomocí příkazu:
`filter_test.py --config ./configs/test_filter.py`.

## Evaluace částicových filtrů
Příkaz
`python ./envs/estimators.py` spustí randomized search evaluace filtrů pro náhodné komibnace parametrů na již 
vygenerované sadě dat
