from kivy.app import App
from kivy.uix.button import Button
from stock_analyzer import run_trading_analysis

class TradingApp(App):
    def build(self):
        return Button(
            text="Mulai Analisis Saham",
            font_size=20,
            on_press=lambda instance: run_trading_analysis()
        )

if __name__ == "__main__":
    TradingApp().run()
