import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = Path(__file__).resolve().parent / "telco_churn.csv"


def main():
    if OUT.exists() and OUT.stat().st_size > 0:
        print(f"already there: {OUT.name}")
        return
    print("downloading Telco Customer Churn ...")
    urllib.request.urlretrieve(URL, OUT)
    rows = sum(1 for _ in OUT.open(encoding="utf-8")) - 1
    print(f"saved {OUT.name} ({rows} rows)")
    if rows < 7000:
        sys.exit("file looks incomplete, delete it and run again")


if __name__ == "__main__":
    main()
