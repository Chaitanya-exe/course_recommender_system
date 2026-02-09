import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://mduadmission.samarth.edu.in/index.php/site/programme?page={}"

all_courses = []

for page in range(1, 13):  
    print(f"Scraping page {page}...")

    url = BASE_URL.format(page)
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select("table tbody tr")

    for row in rows:
        cols = row.select("td")

        if len(cols) >= 2:
            course_name = cols[1].get_text(strip=True)
            all_courses.append(course_name)

df = pd.DataFrame({"course_name": all_courses})
df.to_csv("mdu_courses.csv", index=False)

print(f"\nSaved {len(all_courses)} courses to mdu_courses.csv")