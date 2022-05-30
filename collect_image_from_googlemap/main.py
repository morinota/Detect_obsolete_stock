from lib2to3.pgen2 import driver
from selenium import webdriver
from judge_obsolete_inuse.config import Config


def main():
    # WebDriverインスタンスを生成
    driver = webdriver.Chrome(executable_path=Config.chrome_driver_path)
    # 画面を最大化
    driver.maximize_window()

    # google mapのURLを作成
    query = '北九州市枝光'
    url = f"https://www.google.co.jp/maps/?q={query}"
    url += "&t=k"  # 航空写真

    driver.get(url="https://www.google.com/")  # urlにアクセス
