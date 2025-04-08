import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)

def get_fundamental_data(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "PER": info.get("trailingPE"),
            "PBV": info.get("priceToBook"),
            "ROE": info.get("returnOnEquity"),
            "EPS": info.get("trailingEps"),
            "Market_Cap": info.get("marketCap"),
            "Debt_to_Equity": info.get("debtToEquity")
        }
    except Exception as e:
        logging.error(f"❌ Gagal ambil data fundamental {ticker}: {e}")
        return {}