# -*- coding: utf-8 -*-
from dateparser.search import search_dates

text1 = "Eu comprei isso ontem e vou devolver amanhã."
text2 = "O evento será dia 25 de abril de 2024 às 10h."
print(search_dates(text1, languages=['pt']))
print(search_dates(text2, languages=['pt']))
print(search_dates("25 de abril de 2024", languages=['pt']))
