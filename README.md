# Uczenie głębokie - projekt

Projekt zawiera częściową reimplementację algorytmu LPG z pracy "Discovering Reinforcement Learning Algorithms".

W folderze 'envs/' znajdują się środowiska (ostatecznie użyto tylko TabularGrid). Sam algorytm jest zaimplementowany w pliku 'models/lpg.py', używa też modeli z pliku 'tabular.py'. W pliku 'models/a2c.py' jest zaimplementowany algorytm Advantage Actor-Critic użyty jako baseline.

Plik 'train.py' służy do uruchomienia treningu, 'test.py' testuje wytrenowany wcześniej model, a 'plot.py' tworzy wykresy dla danych z testów.
