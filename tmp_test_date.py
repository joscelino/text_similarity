from dateparser.search import search_dates
print(search_dates("Eu comprei isso ontem e vou devolver amanhã.", languages=['pt']))
print(search_dates("O evento será dia 25 de abril de 2024 às 10h.", languages=['pt']))
