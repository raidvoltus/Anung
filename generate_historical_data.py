import os
from data_fetcher import StockDataFetcher

# Daftar saham yang ingin diambil
tickers = [
    "ACES.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK", "ANTM.JK", "ARTO.JK", "ASII.JK",
    "BBCA.JK", "BBHI.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BFIN.JK", "BMRI.JK", "BRIS.JK",
    "BSDE.JK", "CPIN.JK", "ELSA.JK", "ERAA.JK", "GOTO.JK", "HRUM.JK", "ICBP.JK", "INCO.JK",
    "INDF.JK", "INDY.JK", "INKP.JK", "INTP.JK", "ITMG.JK", "JPFA.JK", "KLBF.JK", "MDKA.JK",
    "MEDC.JK", "MIKA.JK", "PGAS.JK", "PTBA.JK", "PTPP.JK", "PWON.JK", "SMGR.JK", "SMRA.JK",
    "TBIG.JK", "TBLA.JK", "TINS.JK", "TKIM.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK",
    "WEGE.JK", "WSKT.JK", "WTON.JK"
]

# Path output
output_path = "data/historical_data.csv"

if not os.path.exists(output_path):
    print("\n⚠️ File data belum ditemukan. Menghasilkan sekarang...\n")
    fetcher = StockDataFetcher(tickers)
    fetcher.save_all_to_csv(output_path)
else:
    print("✅ File data sudah tersedia, lewati pengambilan.")
