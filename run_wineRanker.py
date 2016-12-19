import wineRanker
from wineRanker import WineRanking
wr = wineRanker.WineRanking()
wr.rank_wines_report()
wr.bootstrap(30)
wr.generate_awesome_plots()
wr.rank_wines_pdf_report()
