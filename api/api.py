import requests
import os
import datetime
import time
import csv

figi_list = "C:/Users/fyfft/PycharmProjects/pythonProject/api/figi.txt"
token = "t.mzCLEtZ8Kp-0HRWb-jhP85EMDR5uvspziu-7iUkuNcN6GTEd97s_eNgkVeYT2yTK-mXjaMRxWq1aUHQdBgRFIg"
minimum_year = int(input('год'))
current_year = int(datetime.datetime.now().year)
url = "https://invest-public-api.tinkoff.ru/history-data"


def download(figi, year):
    # выкачиваем все архивы с текущего года
    if year < minimum_year:
        return

    file_name = f"{figi}_{year}.zip"
    print(f"downloading {figi} for year {year}")
    response = requests.get(f"{url}?figi={figi}&year={year}", headers={"Authorization": f"Bearer {token}"})

    # Если превышен лимит запросов в минуту (30) - повторяем запрос.
    if response.status_code == 429:
        print("rate limit exceed. sleep 5")
        time.sleep(5)
        download(figi, year)
        return
    # Если невалидный токен - выходим.
    if response.status_code == 401 or response.status_code == 500:
        print("invalid token")
        exit(1)
    # Если данные по инструменту за указанный год не найдены.
    if response.status_code == 404:
        print(f"data not found for figi={figi}, year={year}, removing empty file")
        # Удаляем пустой архив.
        os.remove(file_name)
    elif response.status_code != 200:
        # В случае другой ошибки - просто напишем ее в консоль и выйдем.
        print(f"unspecified error with code: {response.status_code}")
        exit(1)

    year -= 1
    download(figi, year)

    with open("../teach/" + file_name, "wb") as file:
        file.write(response.content)
    print(f"Downloading {figi} for year {year}")


with open(figi_list, "r") as file:
    for figi in file:
        download(figi.strip(), current_year)






