import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_Internet_top-level_domains#A"

domains_list = []
for table in pd.read_html(url, header=0):
    try:
        domains = table["Name"].tolist()
        domains_list.extend(domains)
    except Exception as e:
        print(e)

outfile = "../../data/helpers/top_level_domains.txt"
with open(outfile, "w") as fp:
    for line in domains_list:
        fp.write(f"{str(line).strip()}\n")
