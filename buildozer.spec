[app]
title = StockAnalyzer
package.name = stockanalyzer
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 1.0
requirements = python3,kivy,numpy,pandas,joblib,yfinance,ta,scikit-learn,lightgbm,tensorflow,requests
orientation = portrait
fullscreen = 1

[buildozer]
log_level = 2
warn_on_root = 1

[android]
android.permissions = INTERNET
android.api = 33
android.ndk = 25b
android.gradle_dependencies = com.android.support:appcompat-v7:28.0.0
android.archs = armeabi-v7a, arm64-v8a
android.minapi = 21
